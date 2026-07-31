#!/usr/bin/env python
"""Materialize per-chunk observation embeddings from a context-capturing run (GPU).

Unlocks the embedding-OOD detector family (rnd_oe/logpzo/pca_kmeans/knn/mahalanobis/fiper):
reconstructs each recorded chunk's prefix conditioning from the --record-context sidecar and
runs the model's prefix embedder, writing <run>/obs_emb/obs_emb_ro<id>.npz. The lerobot-free
detector_score then loads these into ChunkRecord.obs_emb.

    python cli/obs_emb.py --run <run-id> --device cuda

Needs an adapter with supports_obs_emb=True + a --record-context recording.
See docs/baselines.md §4.
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
    p.add_argument("--no-progress", dest="progress", action="store_false")
    args = p.parse_args()

    from fmaccel.detection.captures import run_obs_emb

    rd, summary = run_obs_emb(
        args.run, rollouts=_int_list(args.rollouts), max_rollouts=args.max_rollouts,
        device=args.device, progress=args.progress,
    )
    print(f"obs_emb: {summary['n_rollouts']} rollouts, d={summary['emb_dim']} -> {rd.stage_dir('obs_emb')}")


if __name__ == "__main__":
    main()
