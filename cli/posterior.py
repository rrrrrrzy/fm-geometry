#!/usr/bin/env python
"""Posterior-scatter sweep: resample every chunk K times + compute metrics.

    python cli/posterior.py --run <run-id> --samples 2048

Writes <run>/posterior/posterior_metrics.{parquet,npz,csv}. Shard across GPUs
with --num-shards / --shard-index (see cli/multigpu.py).
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from fmaccel.core import args as A
from fmaccel.posterior.scatter import run_posterior


def _int_list(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x) for x in s.replace(",", " ").split()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    A.add_run(p)
    A.add_common(p)
    A.add_sampling(p, default_samples=2048)
    p.add_argument("--rollouts", default=None, help="comma/space list (default: all)")
    p.add_argument("--envs", default=None, help="comma/space list (default: all)")
    p.add_argument("--chunks", default=None, help="comma/space list (default: all)")
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--stats-file", default=None, help="npz with frozen mean/std (default: compute)")
    args = p.parse_args()

    out_dir, summary = run_posterior(
        args.run, samples=args.samples, micro_batch=args.micro_batch,
        rollouts=_int_list(args.rollouts), envs=_int_list(args.envs), chunks=_int_list(args.chunks),
        max_chunks=args.max_chunks, num_shards=args.num_shards, shard_index=args.shard_index,
        seed=args.seed, device=args.device, stats_file=args.stats_file,
    )
    print("out_dir:", out_dir)
    print("summary:", summary)


if __name__ == "__main__":
    main()
