# `data/`

Model weights and datasets live here. Nothing in this directory is tracked by git.

## `data/models/pi05-libero-hf/`

The π₀.₅ checkpoint fine-tuned on LIBERO, in LeRobot/HF format — a directory containing
`config.json`, `model.safetensors`, and the `policy_{pre,post}processor*` files. This is the
default `--checkpoint` (also settable as `PI05_LIBERO_CHECKPOINT` in `.env`).

Any LeRobot-loadable π₀.₅ checkpoint works; `lerobot/pi05_base` fine-tuned on LIBERO is what the
paper evaluates. Nothing about the analysis is checkpoint-specific — the scores are read off
whatever denoising trajectory the recorded policy produced.

## `data/datasets/`

Only needed for the teacher-forced execution mode, which reads a LeRobot-format LIBERO dataset.
Closed-loop eval does not need it: it builds the LIBERO environments directly, which requires the
LIBERO benchmark itself to be installed (not on PyPI — see `docs/reproduce.md`).
