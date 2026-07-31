#!/usr/bin/env python
"""Materialize per-chunk action-expert last-hidden features from a context-capturing run (GPU).

Unlocks the SAFE supervised failure probe (detectors/safe.py): reconstructs each recorded chunk's
prefix conditioning from the --record-context sidecar and re-runs the FM head (torch.compile OFF)
to capture the action-expert last-layer hidden (suffix_out), reducing it to one (d,) vector per
decision (SAFE mean-over-horizon-then-flow-step). Writes <run>/hidden_states/hidden_ro<id>.npz.
The lerobot-free detector_score then loads these into ChunkRecord.hidden and runs SAFE's
FAILURE-labeled fit.

    python cli/hidden_states.py --run <run-id> --device cuda

Needs an adapter with supports_hidden=True + a --record-context recording.
See docs/baselines.md §4.2.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from fmaccel.core import args as A


def _int_list(s: "str | list[str] | None") -> list[int] | None:
    if not s:
        return None
    toks = " ".join(s) if isinstance(s, (list, tuple)) else s   # nargs='+' gives a list of tokens
    return [int(x) for x in toks.replace(",", " ").split()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    A.add_run(p)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rollouts", nargs="+", default=None,
                   help="comma/space list of rollout indices (default: all)")
    p.add_argument("--max-rollouts", type=int, default=None, help="cap number of rollouts")
    p.add_argument("--horizon-reduce", default="mean", choices=("mean", "first", "last"),
                   help="SAFE horizon-axis aggregation (default mean = PizeroDatasetConfig)")
    p.add_argument("--diff-reduce", default="mean", choices=("mean", "first", "last"),
                   help="SAFE flow-step-axis aggregation (default mean = PizeroDatasetConfig)")
    p.add_argument("--no-progress", dest="progress", action="store_false")
    args = p.parse_args()

    from fmaccel.detection.captures import run_hidden_states

    rd, summary = run_hidden_states(
        args.run, rollouts=_int_list(args.rollouts), max_rollouts=args.max_rollouts,
        horizon_reduce=args.horizon_reduce, diff_reduce=args.diff_reduce,
        device=args.device, progress=args.progress,
    )
    print(f"hidden_states: {summary['n_rollouts']} rollouts, d={summary['hidden_dim']} "
          f"-> {rd.stage_dir('hidden_states')}")


if __name__ == "__main__":
    main()
