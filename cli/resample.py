#!/usr/bin/env python
"""Resample one recorded FM chunk N times by varying the FM-head noise.

    python cli/resample.py --run <run-id> --chunk r0/e0/c3 --samples 64

Writes a self-contained npz (stored slice + reproduce check + resamples) to
<run>/resample/. Use --samples 0 for the reproduce check only.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from fmaccel.core import args as A
from fmaccel.posterior.stage import parse_chunk_spec, run_resample


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    A.add_run(p)
    A.add_common(p)
    A.add_sampling(p, default_samples=8)
    p.add_argument("--chunk", required=True, help="chunk spec r<R>/e<E>/c<C>, e.g. r0/e0/c3")
    p.add_argument("--no-trajectory", action="store_true", help="skip per-step (x_t,v_t) — ~17× smaller")
    p.add_argument("--num-inference-steps", type=int, default=None)
    args = p.parse_args()

    r, e, c = parse_chunk_spec(args.chunk)
    out, res = run_resample(
        args.run, rollout_idx=r, env_idx=e, chunk_idx=c, samples=args.samples,
        seed=args.seed if args.seed else None, micro_batch=args.micro_batch,
        capture_trajectory=not args.no_trajectory, num_inference_steps=args.num_inference_steps,
        device=args.device,
    )
    print("wrote:", out)
    print("meta:", res["meta"])


if __name__ == "__main__":
    main()
