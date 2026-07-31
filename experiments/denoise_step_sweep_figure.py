#!/usr/bin/env python
"""Denoise-step sweep prefix-ρ figure — 1:1 square (π₀.₅ / LIBERO).

Re-plots ``outputs/denoise_step_sweep/denoise_step_sweep_prefix_curves``
from ``denoise_step_sweep.json``.

Design:
  - drops T=2 (single point);
  - one line per remaining T (4, 6, 8, 10, 15, 20) coloured by a smooth
    LinearSegmentedColormap (dark navy -> teal -> pink);
  - x-axis = k (denoise steps folded into the accel prefix), k = 2..T;
  - y-axis label reads "resample divergence" (not "resample-GT chunk divergence");
  - legend shows only ``T=n``; axis ticks/limits tightened to the data;
  - square canvas with ``set_box_aspect(1.0)``, same PNG+PDF outputs.

Pure numpy/matplotlib; env-agnostic.

    python experiments/denoise_step_sweep_figure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PALETTE = ["#364F6B", "#3FC1C9", "#FC5185"]   # dark navy -> teal -> pink
DROP_TS = {2}


def _nice_ylim(vmin: float, vmax: float, pad_frac: float = 0.10) -> tuple[float, float]:
    span = max(vmax - vmin, 1e-6)
    lo = vmin - pad_frac * span
    hi = vmax + pad_frac * span
    lo = np.floor(lo * 20.0) / 20.0
    hi = np.ceil(hi * 20.0) / 20.0
    return float(lo), float(hi)


def _plot(records: list[dict], out_base: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    import matplotlib.ticker as mticker
    from matplotlib.ticker import MultipleLocator

    records = [r for r in records if int(r["T"]) not in DROP_TS]
    records = sorted(records, key=lambda r: int(r["T"]))
    if not records:
        raise SystemExit("no records to plot after dropping T=2")

    cmap = LinearSegmentedColormap.from_list("color_plan_grad", PALETTE, N=256)
    n = len(records)
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_box_aspect(1.0)

    xs_all: list[float] = []
    ys_all: list[float] = []
    for r, c in zip(records, colors):
        ns = np.asarray(r["prefix_n_steps"], int)
        T = int(r["T"])
        rho = np.asarray(
            [np.nan if x is None else x for x in r["run_rho_prefix_accel_vs_div"]],
            float,
        )
        if not len(ns) or T <= int(ns.min()):
            continue
        k_min = float(ns.min())
        depth = (ns.astype(float) - k_min) / (float(T) - k_min)
        ax.plot(
            depth,
            rho,
            marker="o",
            ms=6.0,
            lw=2.2,
            color=c,
            label=f"T={T}",
            zorder=3,
        )
        xs_all.extend(depth.tolist())
        ys_all.extend([v for v in rho.tolist() if not np.isnan(v)])

    if xs_all and ys_all:
        ax.set_xlim(-0.10, 1.06)
        y_lo, y_hi = _nice_ylim(min(ys_all), max(ys_all), pad_frac=0.15)
        ax.set_ylim(y_lo, y_hi)

    ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))

    ax.set_xlabel(r"prefix depth  $(k-2)\,/\,(T-2)$")
    ax.set_ylabel(r"Spearman $\rho$  (prefix-accel vs resample divergence)")
    ax.set_title(
        r"$\pi_{0.5}$ · LIBERO-all — prefix-accel $\rho$ vs denoise depth",
        fontsize=11, pad=8,
    )
    ax.grid(True, which="major", alpha=0.35, zorder=0)
    ax.grid(True, which="minor", alpha=0.15, zorder=0)

    handles = [
        Line2D([0], [0], color=c, lw=2.5, marker="o", ms=6.0, label=f"T={int(r['T'])}")
        for r, c in zip(records, colors)
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=10,
        framealpha=0.9,
        ncol=1,
    )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--summary",
        default="outputs/denoise_step_sweep/denoise_step_sweep.json",
        help="path to the sweep aggregate JSON",
    )
    p.add_argument(
        "--out",
        default="outputs/denoise_step_sweep/denoise_step_sweep_prefix_curves",
        help="output basename (without extension)",
    )
    args = p.parse_args()

    data = json.loads(Path(args.summary).read_text())
    records = data.get("records", [])
    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    _plot(records, out_base)

    kept = [r for r in records if int(r["T"]) not in DROP_TS]
    print(f"plotted {len(kept)} tracks (dropped T={sorted(DROP_TS)}):")
    for r in sorted(kept, key=lambda r: int(r["T"])):
        print(
            f"  T={int(r['T']):>2}  n={r['n_pooled_chunks']:>5}  "
            f"peak ρ={r['best_prefix_rho']:+.3f} @ k={r['best_prefix_k']}"
        )
    print(f"wrote: {out_base}.png (+ .pdf)")


if __name__ == "__main__":
    main()
