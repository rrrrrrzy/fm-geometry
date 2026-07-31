# Method → code

Where each piece of the paper lives, and the choices in the implementation that a reader is likely
to want to check.

## 1. `accel` — the free proxy

`fmaccel/geometry/accel.py`

For conditional flow matching with a linear interpolant, the velocity field is
`v(x,s) = (E[x₁|x_s=x] − x)/(1−s)`. When the endpoint posterior collapses (`Σ = 0`) the Jacobian is
`J = −I/(1−s)`: an affine isotropic contraction, whose integral curves are straight lines traversed
at constant velocity. **Zero acceleration is the signature of certainty**, and a second-order
Tweedie identity makes the departure from that template *equal to* the posterior covariance up to a
schedule factor. So bending is not a heuristic correlate of uncertainty — it is uncertainty, read
in the geometry.

The discrete estimator over `p` of `T` Euler steps:

```
accel_p = p · Σ_{t=1}^{p−1} ‖v_t − v_{t−1}‖  /  Σ_{t=0}^{p−1} ‖v_t‖
```

| paper | code |
|---|---|
| `accel_T` (full path) | `whole_chunk_curvature(x_t, action_dim, fixed_std)["accel"]` |
| `accel_p` (prefix, all depths at once) | `whole_chunk_prefix_accel(...)` → `(N, T−1)`; column `j` is `p = j+2` |
| named-mode entry point (what the detector calls) | `score_chunks(x_t, action_dim, n_exec=..., mode=...)` |

Implementation choices that matter:

- **The whole action chunk is one point.** `x_t` is `(T+1, chunk, act)`; it is flattened to a single
  `chunk·act`-dimensional path, so a chunk yields one `accel`. Action-level variants
  (`action_level_accel`) score each position separately and are reported as a secondary column.
- **Velocity comes from the path, not a separate array.** With a constant Euler step,
  `Δx_t ∝ v_t`, so differencing the recorded iterates gives the velocity up to the shared `dt` —
  which the normalization cancels. Nothing beyond the recording is needed.
- **Per-coordinate standardization with a run-pooled scale.** Each action dimension is divided by
  its winsorized std pooled over the run before the norms (`_resolve_std`, `_pooled_dim_scale`).
  Without it, the largest-scale dimension dominates; winsorizing keeps one diverged rollout from
  hijacking the scale. A *per-chunk* self-normalization instead re-ranks chunks and measurably
  depresses detection, so the pooled scale is the default.
- **Executed window only.** `n_exec = min(n_action_steps, chunk_size)`. Actions past the execution
  horizon are never run, so scoring them adds noise; both geometric scores restrict to this window.
- **The prefix is along the *denoise* axis, not the chunk.** `accel_prefix:<j>` means "stop after
  `j+2` denoise steps", not "the first `j` actions". `window:<n>` is the position-prefix variant.

**Why `ρ_best > ρ_full`.** `accel` sums velocity differences, so the last Euler steps — where
discretization error is largest — contribute most to the total while carrying least signal. The
full-path number is therefore depressed and the peak sits mid-schedule (around 40% of the path).
This is a property of the estimator, not a tuned hyperparameter; the whole prefix curve is
reported, not just its argmax.

**The terminal singularity is avoided, not solved.** `v = (E[x₁|x_s] − x)/(1−s)` blows up as
`s → 1`. Truncating to a prefix sidesteps it; a principled treatment is future work.

## 2. `Straightness` — the saturating sibling

`fmaccel/geometry/straightness.py`, plus `whole_chunk_straightness` in `geometry/accel.py`.

Chord-to-arc ratio of the denoising path: `‖x_T − x_0‖ / Σ‖Δx_t‖`. It reads the same bend and
correlates with the ground truth (`ρ ≈ −0.7`), but for plain CFM it **saturates**: ~97% of chunks
land in `[0.99, 1.0]`, so it can barely resolve them. `accel` drops the saturating denominator and
keeps the correlation with usable dynamic range. Both are reported; the contrast is the point.

## 3. `D_resample` — the ground truth

`fmaccel/posterior/divergence.py`

Fix the observation, resample `K` chunks with `K` independent noise draws, and take the max
pairwise distance over the `k(k−1)/2` pairs (distances per-dim standardized, averaged over the
executed window). The paper uses `K = 32`.

**The fidelity gate is the load-bearing part.** Resampling at a *drifted* observation measures a
posterior the policy never faced, which would silently invalidate every correlation. So the
conditioning is reconstructed from the `--record-context` sidecar (stored tensors, no environment
replay), and at each chunk start the *recorded* noise is fed back through the FM head and compared
against the recorded chunk. Near-zero error proves the reconstruction; a large one is reported as a
failure for that episode. Recordings without captured context are refused outright.

## 4. Online detection: CUSUM + split conformal

`fmaccel/detection/cusum.py`

A per-decision score is not a detector. Two things convert one into an alarm:

**One-sided CUSUM** accumulates sustained upward drift rather than reacting to single spikes:

```
S_t = max(0, S_{t−1} + (z_t − μ₀ − k)),   S_0 = 0,   k = c·σ,   c = 0.25
```

`μ₀` and `σ` are the pooled mean and std of the calibration scores. Alarm at the first `S_t > η`.
Strictly causal — the decision at `t` uses only chunks `≤ t` — which is what makes the reported
lead an online claim rather than hindsight (`cusum_alarm`; `cusum_peak` is a whole-episode
statistic used *only* for calibration).

**Split conformal on the episode CUSUM peak** sets `η`. Take `P⁽ⁱ⁾ = max_t S_t⁽ⁱ⁾` over `M` held-out
*successful* episodes and use the `r`-th smallest, `r = ⌈(M+1)(1−α)⌉` — a finite-sample-valid bound
giving false-alarm rate `≤ α` on exchangeable future successes (`conformal_height`). With `M = 50`,
`α = 0.1` that is the 46th smallest peak.

At small `M` this differs from `np.quantile`, and the difference is the difference between a
guarantee and an optimistic estimate — hence the explicit order statistic. The calibration episodes
are excluded from the scored population, so the realized FPR is *measured*, not pinned by
construction.

**Why calibrate a scalar rather than a time-indexed band.** A time-indexed conformal band needs
calibration data at every timestep, but successful episodes are short and failing ones run to the
limit. On π₀.₅×LIBERO, successes last ~15.5 decisions while failures run to 28–52, so with `M = 50`
the band's support ends around step 38 — beyond which it is pure padding. Across the paper's cells,
26–55% of failing episodes outlive that support and 4–19% of failing decisions fall in the
unsupported region. Calibrating the episode-level CUSUM *peak* is one scalar per episode and has no
horizon problem.

**Metrics.** TPR at target FPR `α = 0.1`, and the median detection lead `L = T_ep − 1 − t_alarm`
counted in re-planning steps. Reported as mean ± sample std over 5 repeats.

## 5. Baselines

`fmaccel/detectors/` — 14 detectors behind a lazy registry, so a baseline's heavy dependency loads
only when that baseline is built. [`baselines.md`](baselines.md) is a per-detector audit against
each original paper and reference implementation, including where this implementation deviates and
why.

`oracle_resample_spread` is the ground truth used as a detector: an upper bound on what any proxy
of the posterior spread could achieve, **not** a baseline.

Fit protocol, held out from scoring in every case: the unsupervised embedding-OOD family on 32
successful rollouts; SAFE on a balanced 16 success + 16 failure split. Training-free detectors are
scored on that same fit-excluded set within each repeat, so comparisons stay paired. Seeds:
fit/model 3200–3204, candidate subsets 6400–6404.

## 6. The toy

`fmaccel/toy/`, driven by `experiments/flowfield_toy.py`

The toy exists because the real-model posterior is only ever estimated. Here we *choose* it, so the
geometric claim can be checked against a known answer.

| paper's appendix | code |
|---|---|
| 6 training observations at angles `2πi/6`, lifted to R⁸ by a seeded QR `Q` (`QᵗQ = I₂`) | `obs_layout="circle"`, `n_obs=6`, `obs_dim=8` (`toy/data.py`) |
| 3 unimodal + 3 bimodal GMM targets | `n_multimodal=3`, `modes_range=(2,2)` |
| means in `[−2.3, 2.3]²`, rejection-sampled with min separation 2.0 | `mode_box=2.3`, `mode_sep=2.0` |
| per-mode std ~ U[0.06, 0.13]; weights `softmax(0.3 z)` | `sigma_range=(0.06,0.13)`, `weight_temp=0.3` |
| 4096 frozen samples per condition | `samples_per_obs=4096` |
| 3 held-out OOD conditions at the gap midpoints | `n_ood=3` |
| 64-d sinusoidal time embedding; obs MLP 8→32→32 SiLU; 4×256 SiLU trunk → linear 2-d; 224 578 params | `time_dim=64`, `cond_dim=32`, `hidden=256`, `depth=4` (`toy/model.py`) |
| CFM MSE, `s ~ U[1e−3, 1−1e−3]`, train-time obs jitter `0.01·ε` | `s_eps=1e-3`, `obs_jitter=0.01` |
| Adam, 40 000 steps, batch 1024, lr 1e−3, wd 0, float32, seed 0, no schedule/EMA | `steps=40000`, `batch_size=1024`, `lr=1e-3`, `weight_decay=0.0`, `seed=0` |
| 40 Euler steps; 512 noise samples per condition | `--num-steps 40`, `--n-traj` |
| streamplot on a 26×26 grid, cells with <5 samples dropped | `--grid`, `--min-count` |
| endpoint spread = mean of the coordinate-wise endpoint stds | `mean_std` in `summary.json` |

`ToyConfig` is the single source of defaults; `build_config()` in the figure script applies the
figure-specific overrides. Generation fidelity (energy distance to the target samples) is logged
during training as a gate: the field reading is only meaningful once the net has actually fit its
targets, so poor fidelity invalidates the reading rather than indicating uncertainty.

Two honest caveats the figure carries:

- **Off-support ≠ diffuse.** At some held-out angles the net extrapolates to a confident-but-wrong
  field, with *low* `accel` and *low* spread. `all_fields.png` shows both behaviours. The figure
  features the highest-`accel` held-out observation for legibility, and the supported claim is that
  `accel` tracks the posterior spread — true in both cases — not that off-support implies high
  `accel`.
- **Euler error looks like curvature.** Too few integration steps inflate `accel`. Raise
  `--num-steps` when reading geometry; the figure uses 40.

## 7. Recording

`fmaccel/recording/recorder.py` + `fmaccel/models/pi05.py`

`FMRecorder` is model-agnostic. The adapter's `attach_fm_hooks` patches the policy's
`sample_actions` / `denoise_step` to call back into it, so recording is additive: the eval loop is
unchanged and the recorded trajectory is the one that actually drove the robot.

Two invariants any new adapter must satisfy, both gated automatically in `cli/divergence.py`:

1. `x_t[T] == chunk_actions` bitwise — the recorded path really ends at the executed action;
2. resampling from the captured context reproduces the recorded chunk.

**`torch.compile` must be off** (`--disable-compile`): graph capture happens before the hooks fire,
so a compiled sampler records nothing — silently. See [`formats.md`](formats.md) for the on-disk
schema.
