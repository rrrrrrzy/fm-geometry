#!/usr/bin/env python
"""Materialize the per-chunk fm_loss (Diff-DAgger) score from a context-capturing run (GPU).

fm_loss is model-in-the-loop (many re-noised velocity forwards per decision), so it can't run on
the lerobot-free detector_score path. This stage precomputes it: builds each chunk's velocity
callable from the --record-context sidecar and runs FmLossDetector, writing
<run>/fm_loss/fm_loss_ro<id>.npz. detector_score then loads it as a precomputed fm_loss_scores
stream. Includes the in-distribution low-loss smoke gate (validates the -v_θ(x_τ,1-τ) contract).

    python cli/fm_loss.py --run <run-id> --device cuda

Needs an adapter with supports_fm_velocity=True + a --record-context recording.
See docs/baselines.md §3.
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
    p.add_argument("--m-t", type=int, default=16, help="# t samples (Diff-DAgger uses 16)")
    p.add_argument("--m-noise", type=int, default=2, help="# noise draws per t (Diff-DAgger uses 32)")
    p.add_argument("--n-exec", type=int, default=None, help="executed window (default min(n_action_steps,chunk))")
    p.add_argument("--contract-tol", type=float, default=0.12,
                   help="velocity-contract gate tol (max|v_closure - v_recorded|; ~0.1 = bf16 tower "
                        "drift floor, a real wiring bug is O(1))")
    p.add_argument("--rollouts", nargs="+", default=None, help="comma/space rollout indices (default: all)")
    p.add_argument("--max-rollouts", type=int, default=None)
    p.add_argument("--no-progress", dest="progress", action="store_false")
    args = p.parse_args()

    from fmaccel.detection.captures import run_fm_loss_score

    rd, summary = run_fm_loss_score(
        args.run, n_exec=args.n_exec, m_t=args.m_t, m_noise=args.m_noise,
        rollouts=_int_list(args.rollouts), max_rollouts=args.max_rollouts,
        device=args.device, contract_tol=args.contract_tol, progress=args.progress,
    )
    print(f"fm_loss: {summary['n_rollouts']} rollouts -> {rd.stage_dir('fm_loss')}")
    print(f"  velocity-contract gate: max|v_closure - v_recorded|={summary['contract_err']} "
          f"(tol {summary['contract_tol']}) -> {'OK' if summary['contract_ok'] else 'FAILED — numbers untrustworthy'}")


if __name__ == "__main__":
    main()
