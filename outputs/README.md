# `outputs/`

Everything the pipeline produces. Nothing here is tracked by git — this repo ships **code only**,
so every number and figure is regenerated locally.

Point `FMACCEL_OUTPUT_DIR` elsewhere to read and write somewhere else (useful for scoring
recordings that live on another disk).

```
outputs/
├── runs/<run-id>/            one self-describing run per closed-loop eval
│   ├── run.json              model, dataset, args, and the stages that have run
│   ├── eval/                 eval_info.json (per-episode success), trajectories
│   ├── fm/                   the FM recording: every Euler iterate + the captured context
│   ├── videos/               rollout videos, if rendered
│   ├── chunk_divergence/     K resampled candidates per decision (the GT), + reproduce error
│   ├── chunk_geometry/       accel / Straightness / the pooled rho table (meta.json)
│   ├── obs_emb/              observation embeddings   (capture stage, for the OOD baselines)
│   ├── hidden_states/        action-expert features   (capture stage, for SAFE)
│   ├── fm_loss/              Diff-DAgger loss         (capture stage)
│   └── detectors/            per-episode score streams per detector
├── failure_detection/        the assembled Table 2 + figure, per repeat
├── prefix_rho/               the assembled Table 1 + prefix-rho figure
├── per_suite_rho/            per-LIBERO-suite rho breakdown
├── flowfield/{toy,pi05}/      the flow-field figures
└── denoise_step_sweep/       the denoise-depth sweep
```

`run.json` is the contract: a stage records itself there when it completes, so a partially-scored
run is self-describing rather than ambiguous. The on-disk recording format is documented in
[`docs/formats.md`](../docs/formats.md).
