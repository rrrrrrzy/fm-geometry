#!/usr/bin/env python
"""Table 1 / prefix-ρ figure: does the FREE ``accel`` proxy rank decisions like the EXPENSIVE
Monte-Carlo resample posterior?

For one or more scored runs, pools every decision of every episode and reports the Spearman rank
correlation between

  * ``accel`` — read off the recorded denoise path at zero extra cost, and
  * ``D_resample`` — the divergence of ``K`` action chunks resampled at the *same* observation
    with different noise seeds (the paper's uncertainty ground truth),

both for the full-path ``accel_T`` and for the prefix ``accel_p`` over the first ``p`` of ``T``
Euler steps, one ρ per depth. The headline is the **peak of the prefix curve**: ``accel`` is a sum
of velocity differences, so it accumulates last-step Euler discretization noise and the full-path
number is depressed, while the best-aligned readout sits a few steps in (``p*``). That is the
paper's ``ρ_full`` / ``ρ_best`` / ``p*/T`` triple.

Because different policies denoise at different depths, the curve is plotted against the
*fraction* of the path folded in (``p / T``) rather than the raw step index, so runs with
different ``T`` stay comparable.

Input: any run that has been through ``cli/divergence.py`` then ``cli/geometry.py``. This script
only reads each run's ``chunk_geometry/meta.json``, so it is pure numpy/matplotlib — no GPU, no
policy.

    # one run
    python experiments/prefix_rho.py --run <run-id>
    # several, with labels for the legend and table
    python experiments/prefix_rho.py \\
        --run <run-a> --label "pi05 - LIBERO-Spatial" \\
        --run <run-b> --label "pi05 - LIBERO-Object"

The paper's published values for this table are listed in ``docs/reproduce.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from fmaccel.core import runs  # noqa: E402

PALETTE = ["#364F6B", "#3FC1C9", "#FC5185", "#7A9E7E", "#C08552", "#5C6B73"]
MARKERS = ["o", "s", "D", "^", "P", "v"]


def _best_prefix(prefix_n_steps: list[int], prr: list) -> tuple[float | None, int | None]:
    """Peak prefix-accel ρ and the denoise depth (number of steps) it occurs at.

    The full-path accel ρ (the last prefix cutoff) is depressed by the final Euler steps'
    discretization noise — ``accel`` is a sum of velocity differences, so the noisiest steps
    contribute most. The peak of the prefix curve, usually well before the final step, is the
    accel readout best aligned with the resample GT: the paper's ``ρ_best`` at ``p*``."""
    pairs = [(int(n), float(r)) for n, r in zip(prefix_n_steps, prr) if r is not None]
    if not pairs:
        return None, None
    k, r = max(pairs, key=lambda p: p[1])
    return r, k


def _load_run(run_id: str, label: str | None) -> dict:
    """Read one run's pooled ρ table from ``<run>/chunk_geometry/meta.json``."""
    rd = runs.resolve_run(run_id)
    meta_path = rd.chunk_geometry_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{rd.run_id}: no chunk_geometry/meta.json — run `python cli/divergence.py --run "
            f"{rd.run_id} --samples 32` then `python cli/geometry.py --run {rd.run_id}` first")
    meta = json.loads(meta_path.read_text())
    prr = [None if r is None else float(r) for r in (meta.get("run_rho_prefix_accel_vs_div") or [])]
    ns = [int(x) for x in (meta.get("prefix_n_steps") or [])]
    best_rho, best_k = _best_prefix(ns, prr)
    T = len(ns) + 1                                    # denoise steps (prefix cutoffs are 2..T)
    return {
        "run_id": rd.run_id,
        "label": label or rd.run_id,
        "n_pooled_chunks": int(meta.get("n_pooled_chunks", 0)),
        "chunk_size": int(meta.get("chunk_size", 0)),
        "first_actions": int(meta.get("first_actions", 0)),
        "n_action_steps": int(meta.get("n_action_steps", 0)),
        "num_inference_steps": T,
        "rho_full": meta.get("run_rho_accel_vs_div"),           # accel_T (whole denoise path)
        "rho_action": meta.get("run_rho_accel_vs_div_action"),
        "rho_straightness": meta.get("run_rho_straightness_vs_div"),
        "rho_best": best_rho,                                   # accel_{p*} — the headline
        "best_p": best_k,
        "prefix_n_steps": ns,
        "prefix_depth": [float(k) / T for k in ns] if T else [],
        "prefix_fm_time": [float(x) for x in (meta.get("prefix_fm_time") or [])],
        "rho_prefix": prr,
        "n_rollouts": int(meta.get("n_rollouts", 0)),
    }


def _figure(tracks: list[dict], out_base: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, t in enumerate(tracks):
        depth = np.asarray(t["prefix_depth"], float)
        rho = np.asarray([np.nan if r is None else r for r in t["rho_prefix"]], float)
        if not len(depth):
            continue
        col = PALETTE[i % len(PALETTE)]
        ax.plot(depth, rho, marker=MARKERS[i % len(MARKERS)], ms=5, lw=1.9, color=col,
                label=f"{t['label']}  (n={t['n_pooled_chunks']}, T={t['num_inference_steps']})")
        if t["rho_best"] is not None and t["best_p"] is not None:
            ax.plot([t["best_p"] / t["num_inference_steps"]], [t["rho_best"]], marker="*",
                    ms=13, color=col, mec="k", mew=0.5, zorder=5)
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.set_xlabel("fraction of the denoise path folded into the accel prefix  (p / T)")
    ax.set_ylabel("Spearman rho  (accel_p vs resample divergence)")
    ax.set_title("The free accel proxy tracks the expensive resample posterior\n"
                 "star = peak prefix rho", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150)
    plt.close(fig)


def _write_markdown(tracks: list[dict], out_path: Path) -> None:
    def f(x):
        return "—" if x is None else f"{x:+.3f}"

    L = [f"# accel vs resample-divergence rank correlation ({len(tracks)} run(s))\n",
         "Each run was produced once with `--record-fm --record-context`; the free `accel` proxy is",
         "rank-correlated against the Monte-Carlo resample divergence (`K` candidate chunks per",
         "re-plan), pooled over every decision of every episode.\n",
         "**Headline = `rho_best`**, the peak of the prefix curve. The full-path `accel_T` is",
         "depressed by last-step Euler discretization noise, so the best-aligned readout sits a few",
         "denoise steps in. `T` = number of denoise (Euler) steps.\n",
         "| Run | n chunks | T | **rho_best** | p* | rho_full | rho (action-level) | rho (Straightness) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in tracks:
        bp = "—" if t["best_p"] is None else str(t["best_p"])
        L.append(f"| {t['label']} | {t['n_pooled_chunks']} | {t['num_inference_steps']} | "
                 f"**{f(t['rho_best'])}** | {bp} | {f(t['rho_full'])} | "
                 f"{f(t['rho_action'])} | {f(t['rho_straightness'])} |")
    L.append("\n## rho per denoise depth\n")
    L.append("rho of the accel computed from only the first *p* of the *T* denoise steps. "
             "Bold = each run's peak.\n")
    for t in tracks:
        cells = []
        for k, r in zip(t["prefix_n_steps"], t["rho_prefix"]):
            s = "—" if r is None else (f"**{r:+.3f}**" if k == t["best_p"] else f"{r:+.3f}")
            cells.append(f"p={k}: {s}")
        L.append(f"- **{t['label']}** (T={t['num_inference_steps']}): " + "  ".join(cells))
    L.append("\n## Provenance\n")
    for t in tracks:
        L.append(f"- **{t['label']}** — run `{t['run_id']}`, num_inference_steps="
                 f"{t['num_inference_steps']}, n_action_steps={t['n_action_steps']}, "
                 f"first_actions={t['first_actions']}/{t['chunk_size']}, "
                 f"n_rollouts={t['n_rollouts']}.")
    L.append("")
    out_path.write_text("\n".join(L))


def _copy_artifacts(tracks: list[dict], out: Path) -> None:
    """Mirror each run's per-episode chunk_geometry artifacts next to the consolidated figure."""
    import shutil
    for t in tracks:
        geom = runs.resolve_run(t["run_id"]).chunk_geometry_dir
        dst = out / t["run_id"]
        dst.mkdir(parents=True, exist_ok=True)
        if (geom / "meta.json").exists():
            shutil.copy2(geom / "meta.json", dst / "chunk_geometry_meta.json")
        for name in ("accel_divergence_scatter.png", "prefix_accel_rho.png"):
            if (geom / name).exists():
                shutil.copy2(geom / name, dst / name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="a run that has been through cli/divergence.py + cli/geometry.py "
                        "(repeatable)")
    p.add_argument("--label", action="append", default=[],
                   help="legend/table label for the corresponding --run (repeatable, optional)")
    p.add_argument("--out", default="outputs/prefix_rho",
                   help="output directory for the table + figure")
    args = p.parse_args()

    labels = list(args.label) + [None] * (len(args.run) - len(args.label))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tracks = []
    for run_id, label in zip(args.run, labels):
        try:
            tracks.append(_load_run(run_id, label))
        except Exception as ex:  # noqa: BLE001
            print(f"[prefix_rho] skipping {run_id}: {ex}", file=sys.stderr)
    if not tracks:
        raise SystemExit("no run had a chunk_geometry/meta.json")

    _figure(tracks, out / "prefix_rho")
    (out / "prefix_rho.json").write_text(json.dumps({"runs": tracks}, indent=2))
    _write_markdown(tracks, out / "prefix_rho.md")
    _copy_artifacts(tracks, out)

    print(f"=== accel vs divergence rho ({len(tracks)} run(s)) ===")
    for t in tracks:
        rb = "  n/a" if t["rho_best"] is None else f"{t['rho_best']:+.3f}"
        rf = "  n/a" if t["rho_full"] is None else f"{t['rho_full']:+.3f}"
        print(f"  {t['label']:<34} rho_best={rb} @ p*={t['best_p']}/{t['num_inference_steps']}"
              f"   (rho_full {rf}; n={t['n_pooled_chunks']})")
    print(f"\nwrote: {out/'prefix_rho.png'} (+ .pdf), {out/'prefix_rho.md'}, "
          f"{out/'prefix_rho.json'}")


if __name__ == "__main__":
    main()
