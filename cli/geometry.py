#!/usr/bin/env python
"""Whole-chunk geometry: straightness + accel vs chunk-divergence with the *entire
action chunk as one unit*.

    python cli/geometry.py --run <run-id> --rollouts 0,7

Pure analysis (numpy/matplotlib, no GPU): reads the run's fm/ recording for
straightness/accel and the chunk_divergence/*.npz for divergence. Writes
<run>/chunk_geometry/ (per-episode geometry.png + npz + meta.json). Reports the
free proxies' rank-correlation against divergence both per-episode and pooled over
the whole run. Run cli/divergence.py first so the k-candidate chunks exist.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from fmaccel.core import args as A
from fmaccel.geometry.accel import run_chunk_geometry


def _int_list(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x) for x in s.replace(",", " ").split()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    A.add_run(p)
    p.add_argument("--rollouts", default=None, help="comma/space list of rollout ids (default: first --max-rollouts)")
    p.add_argument("--max-rollouts", type=int, default=3, help="cap when --rollouts is unset")
    args = p.parse_args()

    rd, summary = run_chunk_geometry(
        args.run, rollouts=_int_list(args.rollouts), max_rollouts=args.max_rollouts,
    )
    print("out_dir:", rd.chunk_geometry_dir)
    for ep in summary["episodes"]:
        print(f"  ro{ep['rollout_id']}: straight min={ep['straightness_full_min']:.3f} "
              f"accel mean={ep['accel_mean']:.3f} | div peak="
              f"{ep['divergence_peak'] if ep['divergence_peak'] is None else round(ep['divergence_peak'],1)} "
              f"| ρ(accel,div) chunk={ep['rho_accel_vs_div']:+.2f} action={ep['rho_accel_vs_div_action']:+.2f} "
              f"| ρ(straight,div)={ep['rho_straightness_vs_div']:+.2f}")
    rra = summary.get("run_rho_accel_vs_div")
    rrs = summary.get("run_rho_straightness_vs_div")
    rraa = summary.get("run_rho_accel_vs_div_action")
    n_pool = summary.get("n_pooled_chunks", 0)
    n_pool_a = summary.get("n_pooled_actions", 0)
    if rra is not None and n_pool:
        print(f"RUN ρ(accel,div)  chunk (n={n_pool}) = {rra:+.3f}   "
              f"action (n={n_pool_a}) = {rraa:+.3f}")
        print(f"RUN ρ(straight,div) chunk (n={n_pool}) = {rrs:+.3f}")
    ra = summary.get("mean_within_episode_rho_accel_vs_div")
    rs = summary.get("mean_within_episode_rho_straightness_vs_div")
    if ra is not None:
        print(f"mean within-episode ρ vs divergence:  accel={ra:+.3f}   straightness={rs:+.3f}")
    # prefix-accel ρ(t): how early along the denoise the free proxy tracks divergence.
    prt = summary.get("prefix_fm_time") or []
    prr = summary.get("run_rho_prefix_accel_vs_div") or []
    prn = summary.get("prefix_n_steps") or []
    if prt and any(r is not None for r in prr):
        print(f"prefix-accel ρ(noise→t cumulative accel, divergence)  [n={n_pool} chunks]:")
        for ns, t, r in zip(prn, prt, prr):
            rtxt = "  n/a" if r is None else f"{r:+.3f}"
            bar = "" if r is None else "█" * int(round(max(r, 0.0) * 30))
            print(f"  steps≤{ns:<2} (FM t={t:.2f}): ρ={rtxt}  {bar}")
    print("scatter:", rd.chunk_geometry_dir / "accel_divergence_scatter.png")
    if prt and any(r is not None for r in prr):
        print("prefix-ρ curve:", rd.chunk_geometry_dir / "prefix_accel_rho.png")


if __name__ == "__main__":
    main()
