# Reproducing the results

Commands per experiment, and the paper's published numbers to compare against. This repo ships
**code only** — no recordings, no checkpoints — so every number below is regenerated locally.

Everything except the toy figure needs a π₀.₅ checkpoint fine-tuned on LIBERO and the LIBERO
benchmark installed. Scoring an *existing* recording needs neither.

## Setup

```bash
pip install -e '.[all]'
cp .env.sample .env          # then point PI05_LIBERO_CHECKPOINT at your checkpoint
```

LIBERO is not on PyPI. Install the benchmark from
<https://github.com/Lifelong-Robot-Learning/LIBERO> and make sure `import libero` works; lerobot's
LIBERO env adapter imports it. Place (or symlink) the π₀.₅ checkpoint at
`data/models/pi05-libero-hf/` — a LeRobot/HF directory with `config.json`,
`model.safetensors`, and the `policy_{pre,post}processor*` files.

Hardware: recording a full LIBERO suite is the only expensive step (one GPU per shard; use
`cli/multigpu.py eval`). Everything downstream is CPU-bound except the three capture stages and the
learned-detector fits.

---

## 1. The flow-field figure — no data, no GPU

```bash
python experiments/flowfield_toy.py --device cpu
```

~2 minutes. Trains one 2-D conditional flow net on a ring of observations with chosen
Gaussian-mixture action targets, then reads the learned field at a unimodal (certain), a 2-mode
(aleatoric) and a held-out (epistemic) observation. Writes the figure, `all_fields.png` (every
observation, for inspection), and `summary.json` with per-observation `accel` and posterior spread
to `outputs/flowfield/`.

The full toy configuration — ring size, target shapes, net widths, optimizer, 40 000 steps,
seed 0 — is in `fmaccel/toy/config.py`; see [`method.md`](method.md) for the field-by-field map to
the paper's appendix.

**One honest caveat.** Off-support observations are *not* uniformly diffuse: at some held-out
angles the net extrapolates to a confident-but-wrong field, where both `accel` and the posterior
spread are low. `all_fields.png` shows both behaviours. The figure features the highest-`accel`
held-out observation, and the claim it supports is that **`accel` tracks the posterior spread** —
which stays honest in both cases — not that off-support always implies high `accel`.

The π₀.₅ counterpart (`experiments/flowfield_pi05.py`) projects a real recorded decision's field
onto its first two principal components and needs a recording from step 2.

---

## 2. Record

```bash
python cli/eval.py --model pi05 --dataset libero --task-group libero_object --task-ids 0..9 \
    --record-fm --record-context --disable-compile --run-id my-run

# or shard across GPUs into one run dir:
python cli/multigpu.py eval --gpus 0,1,2,3 --model pi05 --dataset libero \
    --task-group libero_object --task-ids 0..9 --record-fm --record-context
```

`--disable-compile` is **required** with `--record-fm`: `torch.compile` captures the
action-sampling graph before the recording hooks fire, so a compiled path silently records nothing.

`--record-context` is what makes the ground truth possible — it stores the exact conditioning each
decision saw, so the posterior can be resampled at that conditioning without re-running the
environment. Without it, step 3 refuses to run.

π₀.₅ on LIBERO uses 10 denoise steps and an executed window of 10 actions.

## 3. Resample the ground truth

```bash
python cli/divergence.py --run my-run --samples 32 --first-actions 10
```

Draws K=32 chunks per decision at that decision's own reconstructed conditioning and records their
pairwise spread. Watch the reproduce error in the log: this stage feeds the *recorded* noise back
through the FM head and compares against the recorded chunk. Near-zero means the conditioning was
reconstructed exactly; a large value means that episode's ground truth is untrustworthy, and the
stage says so rather than quietly proceeding.

`--first-actions 10` restricts the distance to the executed window, matching the paper.

## 4. Table 1 — does the free proxy track the ground truth?

```bash
python cli/geometry.py --run my-run
python experiments/prefix_rho.py --run my-run --label "pi05 - LIBERO-Object"
```

**Published (π₀.₅ × LIBERO, pooled over 32 647 decisions, T=10):**

| quantity | value |
|---|---|
| `ρ_full` — full-path `accel_T` vs `D_resample` | **0.541** |
| `ρ_best` — best prefix `accel_{p*}` | **0.792** |
| `p*` / T | 5 / 10 |

`ρ_best` > `ρ_full` is expected, not tuning: `accel` sums velocity differences, so the final Euler
steps contribute the most discretization noise and depress the full-path number. The peak sits
around 40% of the way along the denoise path.

Two things will shift your numbers legitimately: **which suite** you record (the paper's 32 647
decisions pool all four LIBERO suites, and pooling heterogeneous suites dilutes ρ — a single suite
scores higher), and **K** (fewer resamples makes the ground truth noisier). The sign, the shape of
the prefix curve, and `ρ_best > ρ_full` should all reproduce regardless.

## 5. Capture stages (only for the learned baselines)

```bash
# <python> <run-path> <shard-id> <n-shards> <first-actions> [micro-batch]
bash experiments/fd_gpu_stages.sh python outputs/runs/my-run 0 1 10

# to spread across 4 GPUs, run one shard per GPU:
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i bash experiments/fd_gpu_stages.sh python outputs/runs/my-run $i 4 10 &
done; wait
```

Runs `cli/divergence.py`, `cli/obs_emb.py`, `cli/hidden_states.py` and `cli/fm_loss.py` in
sequence. Shard `i` owns rollouts `{i, i+N, ...}`, every stage writes per-rollout npz files and
resume-skips ones already on disk, so shards are race-free and an interruption only re-does the
in-flight rollout.

`accel` and `Straightness` need **none** of this — that is the point of them — so skip this step
if you only want the two free scores.

## 6. Table 2 — online failure detection

```bash
python experiments/failure_detection.py score --cell pi05:libero_all --device cuda
python experiments/failure_detection.py aggregate
```

Defaults match the paper: K=32 resamples, 5 repeats, fit seeds 3200–3204, candidate-subset seeds
6400–6404, the OOD family fit on 32 successful rollouts, SAFE on a balanced 16+16 split, with every
fit set held out from scoring. The CUSUM slack is `k = 0.25σ`, and the alarm height is
split-conformally calibrated on M=50 held-out successful episodes at target FPR α=0.1.

**Published (π₀.₅ × LIBERO-all; success rate 96.3%). TPR ± std / median detection lead, in
re-planning steps:**

| Detector | TPR | lead | cost |
|---|---|---|---|
| **Accel** (ours) | **0.85** ±.02 | 19 | free |
| **Straightness** (ours) | 0.78 ±.02 | 18 | free |
| ACE | 0.85 ±.01 | 19 | K resamples |
| STAC | 0.90 ±.02 | 22 | K resamples |
| Diff-DAgger | 0.67 ±.04 | 20 | K forwards |
| FIPER | 0.81 ±.03 | 19 | training |
| RND-OE | 0.38 ±.11 | 18 | training |
| LogpZO | 0.38 ±.08 | 20 | training |
| SAFE | 0.80 ±.09 | 19 | training + failure labels |

The claim is not that `accel` wins outright — STAC edges it on this cell — but that a score costing
*nothing* lands among methods that each cost 32 extra forward passes or a training run, and does so
with markedly lower variance (±.02 for the geometric scores vs ±.09–.11 for the trained ones). Note
the FPR is measured out-of-sample: the calibration episodes are excluded from the scored
population, so the realized false-alarm rate is a result, not the target it was pinned to.

Averaged across the paper's full 8-cell grid, Accel scores 0.66 and Straightness 0.65, against
0.55 (ACE), 0.60 (STAC), 0.52 (Diff-DAgger), 0.55 (FIPER), 0.40 (RND-OE), 0.41 (LogpZO) and 0.68
(SAFE). Reproducing that average needs the other three models and RoboCasa, which this release does
not include — see the scope note in the [README](../README.md).

## 7. Other experiments

```bash
# how the prefix-rho curve moves with the total denoise-step count T
python experiments/denoise_step_sweep.py --gpus 0,1,2,3
python experiments/denoise_step_sweep_figure.py

# per-suite rho instead of the pooled LIBERO number
python experiments/libero_per_suite_rho.py --run my-run

# CUSUM slack sensitivity, and re-scoring accel at a different prefix depth
python experiments/cusum_ksigma_sweep.py
python experiments/accel_prefix_probe.py --mode accel_prefix:1 --cell pi05:libero_all
```

## Adding a model or a benchmark

Every score reads the on-disk recording, not a live policy, so a new policy needs only:

1. a model adapter in `fmaccel/models/` implementing the `ModelAdapter` contract (see
   `models/base.py`; `models/pi05.py` is the worked example) — the essential part is
   `attach_fm_hooks`, which patches the sampler to call back into `FMRecorder`;
2. one line in `fmaccel/registry.py`;
3. for the learned baselines, the three optional capability flags (`supports_obs_emb`,
   `supports_hidden`, `supports_fm_velocity`) and their `embed_context_*` /
   `velocity_fn_for_context` methods.

Then register a detection cell:

```bash
python experiments/failure_detection.py --register-cell mymodel:mybench:my-run:8 score --all
```

Verify a new adapter with the recording invariant the whole pipeline rests on: `x_t[T]` must equal
`chunk_actions` bitwise, and resampling from the captured context must reproduce the recorded chunk.
Both are gated automatically in `cli/divergence.py`.
