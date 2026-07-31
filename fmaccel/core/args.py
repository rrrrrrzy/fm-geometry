"""Reusable argparse fragments — ONE canonical flag name per concept.

Thin scripts compose a parser from these fragments instead of redeclaring flags,
which is how the refactor kills the old ``--k`` vs ``--n-samples`` vs ``--n-vis``
(and ``--obs-size`` vs ``--img-size`` vs ``--video-size``) drift. Each ``add_*``
mutates and returns the parser so calls chain.

Canonical names:
  --model/--dataset/--checkpoint, --run, --samples, --micro-batch, --render-size,
  --device, --seed, --tag, --gpus, --num-inference-steps, --n-action-steps,
  --disable-compile, --record-fm/--record-context.

Importing this module is lerobot-free: ``--model``/``--dataset`` choices come from
the registry's string tables, which import nothing heavy.
"""

from __future__ import annotations

import argparse

from fmaccel.registry import list_datasets, list_models


def add_model(p: argparse.ArgumentParser, *, required: bool = True) -> argparse.ArgumentParser:
    g = p.add_argument_group("model")
    g.add_argument("--model", choices=list_models(), required=required,
                   help="model adapter (registry name)")
    g.add_argument("--checkpoint", default=None,
                   help="checkpoint path/dir (default: adapter's env-var/default)")
    return p


def add_dataset(p: argparse.ArgumentParser, *, required: bool = True) -> argparse.ArgumentParser:
    g = p.add_argument_group("dataset")
    g.add_argument("--dataset", choices=list_datasets(), required=required,
                   help="dataset/env adapter (registry name)")
    g.add_argument("--task-group", default=None,
                   help="task suite within the dataset (e.g. libero_spatial)")
    g.add_argument("--task-ids", type=int, nargs="+", default=None,
                   help="task indices within the suite (default: adapter default)")
    g.add_argument("--n-episodes", type=int, default=1, help="episodes per task")
    g.add_argument("--batch-size", type=int, default=1, help="parallel envs per task")
    return p


def add_run(p: argparse.ArgumentParser, *, required: bool = True) -> argparse.ArgumentParser:
    """Downstream stages: resolve an existing run by id (under outputs/runs/) or path."""
    p.add_argument("--run", required=required,
                   help="run-id or path of the producing run (reads its run.json)")
    return p


def add_common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--device", default="cuda", help="torch device")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--tag", default=None, help="run-id suffix (else HHMMSS is used)")
    return p


def add_sampling(p: argparse.ArgumentParser, *, default_samples: int = 2048) -> argparse.ArgumentParser:
    g = p.add_argument_group("sampling")
    g.add_argument("--samples", type=int, default=default_samples,
                   help="number of FM resamples / noise draws (was --k/--n-samples/--n-vis)")
    g.add_argument("--micro-batch", type=int, default=None,
                   help="cap per-forward batch size (default: --samples)")
    return p


def add_render(p: argparse.ArgumentParser, *, default: int = 360) -> argparse.ArgumentParser:
    p.add_argument("--render-size", type=int, default=default,
                   help="offscreen render resolution (was --obs-size/--img-size/--video-size)")
    return p


def add_fm_inference(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = p.add_argument_group("fm inference")
    g.add_argument("--num-inference-steps", type=int, default=None,
                   help="override FM denoise step count")
    g.add_argument("--n-action-steps", type=int, default=None,
                   help="actions executed before re-querying the policy")
    g.add_argument("--disable-compile", action="store_true",
                   help="disable torch.compile (REQUIRED when recording — hooks break under compile)")
    return p


def add_recording(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = p.add_argument_group("fm recording")
    g.add_argument("--record-fm", action="store_true", help="record FM denoising trajectories")
    g.add_argument("--record-context", action="store_true",
                   help="also store the exact FM-head prefix inputs per chunk")
    return p


def add_gpus(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--gpus", default=None, help='comma/space-separated GPU ids, e.g. "0,1,2,3"')
    return p
