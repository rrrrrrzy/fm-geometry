# The Geometric Nature and a Free Proxy for Flow-Matching Uncertainty

Reference implementation of **`accel`** (denoising acceleration) — a cost-free uncertainty proxy
for flow-matching policies, read off a *single* forward pass with no extra model evaluations, no
training, and no resampling — and of the online failure detector built on it.

The idea in one paragraph. For conditional flow matching with a linear interpolant, if the action
posterior collapses to a point the velocity field is an **affine isotropic contraction**, and every
denoising trajectory is a straight line at constant velocity: zero acceleration. Uncertainty is
exactly the departure from that template. `accel` measures it as the normalized total variation of
the denoising velocity, and it tracks the expensive Monte-Carlo resample posterior closely enough
to serve as its free stand-in — including as an online failure signal that fires well before a
rollout ends.

```
record the denoise path  ──►  read accel / Straightness  ──►  CUSUM + split conformal  ──►  alarm
   (free, during eval)          (free, post-hoc)               (calibrated on successes)
                          └──►  resample K chunks  ──►  D_resample     (expensive; the ground
                                (validation only)                       truth accel stands in for)
```

## Scope of this release

This repo ships the **π₀.₅ × LIBERO** reference path: closed-loop recording, the resample ground
truth, both geometric scores, all 14 registered detectors, and the CUSUM + split-conformal
calibration. It reproduces that column of the paper end-to-end.

The paper additionally evaluates **SmolVLA, GR00T N1.7 and VLA-JEPA** on **LIBERO, RoboCasa
Atomic-Seen and D3IL**; those model adapters and benchmark adapters are **not included here**. The
gap is adapters and eval glue, not method: `accel`, `Straightness`, every detector and the whole
calibration layer read the **on-disk recording**, not a live policy, so they are model-agnostic
already. Adding a model is one adapter class plus one line in `fmaccel/registry.py`; adding a
detection cell is one `--register-cell` flag.

**No data ships.** Every table and figure is regenerated locally. The only artifact reproducible
with no checkpoint and no GPU is the toy flow-field figure (below); everything else needs a π₀.₅
checkpoint and a LIBERO install.

## Install

```bash
pip install -e .                  # analysis only: numpy / scipy / matplotlib / tqdm
pip install -e '.[learned]'       # + torch, for the learned baselines (rnd_oe / logpzo / safe)
pip install -e '.[all]'           # + lerobot & gymnasium, to record new rollouts
```

Producing a *new* recording also needs the LIBERO benchmark, which is not on PyPI — see
[`docs/reproduce.md`](docs/reproduce.md). Scoring an *existing* recording never does.

## Quickstart — the flow-field figure, ~2 min, no GPU, no data

```bash
python experiments/flowfield_toy.py --device cpu
```

Trains one 2-D conditional flow net (225k parameters) on a ring of observations whose action
targets we choose ourselves — some unimodal, some 2-mode, plus held-out observations the net never
saw — then draws the learned field at three of them. The certain field contracts to a point with a
near-straight path and low `accel`; the multimodal and off-support fields bend, with high `accel`
and a wide resampled posterior. Writes the figure plus a `summary.json` of per-observation `accel`
and posterior spread to `outputs/flowfield/`.

Because the toy's posterior is known by construction, this is the controlled version of the paper's
central claim, and it needs nothing but this repo.

## The full pipeline

```bash
# 1. record — closed-loop LIBERO eval, capturing every Euler iterate + the exact conditioning.
#    --disable-compile is REQUIRED: torch.compile captures the graph before the hooks fire.
python cli/eval.py --model pi05 --dataset libero --task-group libero_object \
    --record-fm --record-context --disable-compile --run-id my-run

# 2. ground truth — resample K=32 chunks at each decision's own observation.
#    Gated on bitwise reproduction of the recorded chunk: a drifted observation would measure
#    a posterior the policy never faced, so the stage refuses rather than approximates.
python cli/divergence.py --run my-run --samples 32 --first-actions 10

# 3. the free scores + Table 1 — accel, Straightness, and the pooled rank correlation.
python cli/geometry.py --run my-run
python experiments/prefix_rho.py --run my-run --label "pi05 - LIBERO-Object"

# 4. capture stages the learned baselines need (GPU; skip if only scoring accel/Straightness).
#    Args: <python> <run-path> <shard-id> <n-shards> <first-actions>
bash experiments/fd_gpu_stages.sh python outputs/runs/my-run 0 1 10

# 5. Table 2 — the detector battery, CUSUM + split conformal, 5 repeats.
python experiments/failure_detection.py score --cell pi05:libero_all --device cuda
python experiments/failure_detection.py aggregate
```

Every stage writes into one self-describing run directory and records itself in its `run.json`; see
[`outputs/README.md`](outputs/README.md) for the layout and [`docs/formats.md`](docs/formats.md)
for the on-disk recording format.

## Notation: paper ↔ code

| Paper | Code |
|---|---|
| `accel` / denoising acceleration (Algorithm 1) | `whole_chunk_curvature` in `fmaccel/geometry/accel.py` |
| prefix `accel_p`, first `p` of `T` Euler steps | `whole_chunk_prefix_accel`; selected by mode string `accel_prefix:<j>`, `p = j+2` |
| `TV(v)` on the figures | the same quantity as `accel` |
| `Straightness` (chord/arc) | `fmaccel/geometry/straightness.py` |
| `D_resample` — the uncertainty ground truth | `fmaccel/posterior/divergence.py` |
| `ρ_full` / `ρ_best` / `p*` | `run_rho_accel_vs_div` / peak of `run_rho_prefix_accel_vs_div` in `chunk_geometry/meta.json` |
| one-sided CUSUM `S_t`, slack `k = cσ`, `c = 0.25` | `fmaccel/detection/cusum.py` |
| split conformal alarm height, `r = ⌈(M+1)(1−α)⌉`, `M = 50`, `α = 0.1` | same module |
| TPR at target FPR, median detection lead | same module |

Both geometric scores are computed over the **executed action window** only (`n_exec =
min(n_action_steps, chunk_size)`), not the full action horizon — pass it as `--first-actions`.

## Layout

```
fmaccel/
  recording/    FM trajectory capture during eval + a numpy-only reader
  geometry/     accel (Algorithm 1), Straightness, the denoise-phase profile   [no policy needed]
  posterior/    resample the K-chunk posterior — the ground truth              [needs the policy]
  detectors/    14 detectors behind a lazy registry; see docs/baselines.md
  detection/    scoring, CUSUM + conformal calibration, GPU capture stages
  models/       the pi05 adapter (the extension point for other policies)
  datasets/     the LIBERO adapter
  toy/          the 2-D flow net for the flow-field figure
cli/            one thin wrapper per pipeline stage
experiments/    the paper's tables and figures
docs/           method, formats, baselines, reproduce
```

The split that matters: `geometry/`, `detectors/`, `detection/score`, `detection/cusum` and
`recording/loader` are **numpy-only and import without a policy installed**. Only recording and
resampling need the model. That is what makes the scores free — they are read off an artifact, not
computed by a second forward pass.

## Docs

- [`docs/reproduce.md`](docs/reproduce.md) — commands per experiment, and the paper's published
  numbers to compare against.
- [`docs/method.md`](docs/method.md) — the estimators, the calibration, and the toy configuration,
  each mapped to the code.
- [`docs/baselines.md`](docs/baselines.md) — per-detector fidelity notes against each baseline's
  original paper and reference implementation.
- [`docs/formats.md`](docs/formats.md) — the on-disk recording format.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
