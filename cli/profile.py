#!/usr/bin/env python
"""accel_profile: WHERE along the denoise schedule a chunk's ``accel`` lives.

    python cli/profile.py --run <run-id> [--window full|exec]
    python cli/profile.py --compare RUN_A,RUN_B --labels spatial,object

Pure analysis (numpy/matplotlib, no GPU): reads only the run's fm/ recording. ``accel``
is already Σ over denoise steps, so this keeps the per-step contribution
c_t = ‖Δv_{t+1}−Δv_t‖/⟨‖Δv‖⟩ (Σ_t c_t == accel, asserted), normalizes within chunk, and
plots ĉ_t vs FM time grouped by accel level — answering where high-accel chunks bend.
Writes <run>/accel_profile/ (profile.png + cm_vs_accel.png + npz + meta.json). The
--compare mode overlays the high/low group profiles of several runs on one FM-time axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from fmaccel.core import args as A
from fmaccel.core import runs
from fmaccel.geometry.profile import NPZ_NAME, run_accel_profile


def _int_list(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x) for x in s.replace(",", " ").split()]


def _compare(run_specs: list[str], labels: list[str] | None, out: str | None) -> None:
    """Overlay the high vs low accel-group mean profiles of several runs (read from each
    run's accel_profile.npz — run the per-run stage first) on one FM-time axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = labels or run_specs
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (spec, lab) in enumerate(zip(run_specs, labels)):
        rd = runs.resolve_run(spec)
        npz = rd.accel_profile_dir / NPZ_NAME
        if not npz.exists():
            raise FileNotFoundError(f"{npz} missing — run accel_profile on {spec} first")
        d = np.load(npz, allow_pickle=True)
        tm = d["time_mid"]
        gl = list(d["group_labels"])
        gm = d["group_means_chunk"]                       # (n_groups, T-1)
        hi = next(k for k, l in enumerate(gl) if l.startswith("high"))
        lo = next(k for k, l in enumerate(gl) if l.startswith("low"))
        c = colors[i % len(colors)]
        ax.plot(tm, gm[hi], "-", lw=2.2, color=c, label=f"{lab} · high accel")
        ax.plot(tm, gm[lo], "--", lw=1.4, color=c, alpha=0.7, label=f"{lab} · low accel")
    ax.invert_xaxis()
    ax.set_xlabel("FM time  (1.0 = noise  →  0.1 = action)")
    ax.set_ylabel("normalized curvature  ĉ_t = c_t / accel")
    ax.set_title("Where the denoise bend lives: high vs low accel across datasets")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out) if out else Path("accel_profile_compare.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("compare ->", out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    A.add_run(p, required=False)
    p.add_argument("--window", choices=("full", "exec"), default="full",
                   help="action sub-chunk the denoise path is read over (default: full chunk)")
    p.add_argument("--rollouts", default=None, help="comma/space list of rollout ids (default: all)")
    p.add_argument("--max-rollouts", type=int, default=None, help="cap when --rollouts is unset")
    p.add_argument("--compare", default=None, help="comma list of run-ids/paths to overlay (reads each npz)")
    p.add_argument("--labels", default=None, help="comma list of legend labels for --compare")
    p.add_argument("--out", default=None, help="output path for the --compare figure")
    args = p.parse_args()

    if args.compare:
        _compare([s.strip() for s in args.compare.split(",")],
                 [s.strip() for s in args.labels.split(",")] if args.labels else None,
                 args.out)
        return

    if not args.run:
        p.error("--run is required (or use --compare)")
    rd, summary = run_accel_profile(
        args.run, rollouts=_int_list(args.rollouts), max_rollouts=args.max_rollouts,
        window=args.window,
    )
    print("out_dir:", rd.accel_profile_dir)
    print(f"pooled {summary['n_chunks']} chunks / {summary['n_actions']} action-paths | "
          f"max|Σc_t−accel| = {summary['max_sum_err']:.2e} | "
          f"{summary['n_denoise_jerk_terms']} jerk terms")
    print(f"ρ(τ_cm, accel)  chunk={summary['rho_cm_accel_chunk']:+.3f}  "
          f"action={summary['rho_cm_accel_action']:+.3f}")
    for g in summary["groups_chunk"]:
        print(f"  {g['label']:<16} n={g['n']:<6} τ_cm={g['tau_cm_mean']:+.3f}  "
              f"early(noise-side) frac={g['early_frac_mean']:.3f}")
    print("profile:", rd.accel_profile_dir / "profile.png")


if __name__ == "__main__":
    main()
