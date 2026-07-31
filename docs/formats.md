# Data formats

Every on-disk schema this repo reads or writes. The FM recording is the load-bearing one: it is
the artifact that makes the geometric scores *free* — they are read off it, so no policy is needed
to compute a score, and no second forward pass is spent.

## `run.json`

One per `outputs/runs/<run-id>/`. Written by the producing stage; analysis stages
append to `stages`.

```json
{
  "run_id": "<date>_pi05_libero-object_v1",
  "created": "...", "git_sha": "...",
  "stage": "eval", "parent_run": null,
  "model":   {"name": "pi05", "checkpoint": "...", "config": {...}},
  "dataset": {"name": "libero", "task_group": "libero_object", "task_ids": [0]},
  "args":    {...},
  "stages":  {"chunk_divergence": {"at": "...", "args": {...}}, ...}
}
```

`checkpoint` is the **resolved** path actually loaded (even if defaulted from an env
var), so analysis stages rebuild the exact policy. `parent_run` is set only for
derived runs (`RunDir.child`).

## FM recording — format v3

```
fm/
  manifest.json
  rollouts/
    <prefix>rollout_0000.npz           # one rollout per file, plain array names
    <prefix>rollout_0000_context.npz   # optional (--record-context)
```

One npz **per rollout**, so multi-GPU recording is embarrassingly parallel: each worker writes
its own `<prefix>rollout_*.npz` and the merged manifest just concatenates the rollout list.

`manifest.json`:

```json
{
  "version": 3,
  "dims": {"n_action_steps": 10, "chunk_size": 50, "max_action_dim": 32,
           "action_dim": 7, "num_inference_steps": 10, "dt": -0.1},
  "has_context": true,
  "rollouts": [
    {"file": "rollouts/rollout_0000.npz", "context_file": "...|null",
     "rollout_id": 0, "task_id": 0, "task_group": "libero_object",
     "task_descs": [...], "n_chunks": 8, "batch_size": 1, "n_env_steps": 80}
  ],
  ...run_metadata (source, policy_path, seed, model, dataset)
}
```

### Per-rollout npz arrays

`T = num_inference_steps`, `B = batch`, `K = chunk_size`, `D = max_action_dim`,
`A = action_dim`. `n_env_steps = 0` for teacher-forced recordings (no env feedback).

`v_t` is the whole point: `accel` is the normalized total variation of exactly this array, so the
paper's estimator is a few lines of numpy over a file the eval already wrote.

| key | shape | meaning |
|---|---|---|
| `time` | `(n_chunks, T)` | FM schedule, descends `1.0 → 0.1` (the `timestep` fed to `denoise_step`) |
| `noise` | `(n_chunks, B, K, D)` | initial `x_0` at `t=1` (pre-sampled, bit-exact) |
| `x_t` | `(n_chunks, T+1, B, K, D)` | FM state trajectory; `x_t[c,0]==noise[c]`, `x_t[c,k+1]=x_t[c,k]+dt·v_t[c,k]`, `x_t[c,-1] ≈ chunk_actions[c]` (padded) |
| `v_t` | `(n_chunks, T, B, K, D)` | velocity outputs of `denoise_step` |
| `chunk_actions` | `(n_chunks, B, K, A)` | final chunks, **model-space** (pre env-postprocessor), truncated to `A` |
| `env_step_at_chunk_start` | `(n_chunks,)` int32 | env-step counter at chunk (0, 10, 20, …) |
| `chunk_idx` | `(n_chunks,)` int32 | sequential chunk index |
| `task_descs` | `(n_chunks, B)` object | task string per slot |
| `terminated` / `truncated` | `(n_env_steps, B)` bool | episode boundaries (LIBERO auto-resets on done) |

Context sidecar (`--record-context`): `ctx_images (n_chunks, ncam, B, 3, 224, 224)`,
`ctx_img_masks (n_chunks, ncam, B)`, `ctx_tokens`/`ctx_masks (n_chunks, B, L)`.
`FMRecording.load` attaches these (`.has_context`, `.context(env_idx, chunk_idx)`).

This sidecar is what makes the resample ground truth trustworthy: the posterior must be drawn at
the *exact* conditioning the policy faced, and reconstructing it from stored tensors avoids
re-running the environment (which would drift). Every stage that needs the model
(`chunk_divergence`, the capture stages) reads it, and each gates on reproducing the recorded
chunk from the recorded noise before trusting its own numbers. Recordings without it are refused
rather than approximated.

### Flow-matching conventions (π₀.₅ native time)

π₀.₅ denoises in **reverse time**: `t: 1 → 0` over `T` steps, `dt = -1/T < 0`. `x_t`
starts at the noise (`t=1`) and ends at the action (`t≈0`); `v_t` is the raw
action-expert velocity. Actions are recorded model-space (replay re-runs the same
postprocessor chain, so this is transparent for re-execution).

## Chunk-geometry output

`chunk_geometry/meta.json` — the pooled table this repo's Table 1 is read from:
`run_rho_accel_vs_div` (the full-path ρ), `run_rho_prefix_accel_vs_div` (one ρ per denoise depth,
whose peak is `ρ_best`), `prefix_n_steps`, `prefix_fm_time`, `run_rho_straightness_vs_div`,
`n_pooled_chunks`, and the per-episode breakdown. Plus `chunk_geometry[_roN].npz` per episode and
the pooled `accel_divergence_scatter.png`.

## Chunk-divergence output

`chunk_divergence/chunk_divergence[_roN].npz` — the K resampled candidate chunks per decision, the
max/mean pairwise spread (`D_resample`), the per-action within-chunk profile, and the per-chunk
`reproduce` error from the fidelity gate. A large reproduce error invalidates that episode's GT.

## Posterior metrics

`posterior/posterior_metrics.{npz,csv}` — one row per chunk. Index columns:
`rollout_idx, env_idx, chunk_idx, task_id, task_desc, env_step_at_chunk_start,
episode_success, reproduce_ok`. Metric columns: `mean_std, participation_ratio,
mardia_skew, mardia_kurt_z, bimodality_pc1, frac_dims_bimodal, gmm_delta_bic_pc1,
straightness_mean, straightness_min`. `meta_json` holds the run config +
`standardize_mean/std` (consumed by `flow_viz`).

## Resample output

`resample/r<R>_e<E>_c<C>_n<N>_s<seed>.npz` — self-contained: `stored_*` (the recorded
slice), `reproduce_*` (re-run from stored noise, with `reproduce_max_abs_err` — 0 means
exact), `resample_*` (N new noise draws), and a `meta_json` blob.
