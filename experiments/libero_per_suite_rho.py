#!/usr/bin/env python
"""LIBERO per-suite decomposition of the crossbench accel↔resample-GT ρ.

The pooled `libero_all` ρ (in `crossbench_summary.md`) mixes 4 heterogeneous suites
(spatial / object / goal / long); when a policy's success is unbalanced across suites
the pooled ρ is diluted by the worst suite's noise. This script rebuilds the same
Spearman ρ **per suite**, from the same per-rollout `chunk_geometry_ro*.npz` +
`chunk_divergence_ro*.npz` (already on disk — no GPU work).

Provenance-safe: reads `eval/eval_info_gpu*.json` (each worker records the actual
suite order it evaluated — spatial → object → goal → libero_10, 20 ep each) to fix
a recording whose fm/manifest.json labelled every rollout
as `libero_spatial`. Falls back to fm/manifest.json for other runs (pi05 records its
groups correctly).

Outputs to `outputs/per_suite_rho/<cell_key>/per_suite/`:
  * `per_suite_summary.md` — per-suite ρ table (best-prefix, full-path, straightness, action).
  * `per_suite_summary.json` — machine-readable version + per-prefix ρ grid.
  * `per_suite_prefix_rho.{png,pdf}` — 4 suites on one axes (colored by suite).

Usage:
    python experiments/libero_per_suite_rho.py --run <run-id>
    python experiments/libero_per_suite_rho.py --run outputs/runs/<id>

By default runs on every LIBERO cell in the crossbench summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401,E402

from fmaccel.core import runs as _runs  # noqa: E402

SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_COLOR = {  # match crossbench_prefix_rho_square palette + one extra
    "libero_spatial": "#364F6B",  # dark navy
    "libero_object":  "#3FC1C9",  # teal
    "libero_goal":    "#F5B841",  # amber
    "libero_10":      "#FC5185",  # pink
}
SUITE_MARKER = {"libero_spatial": "o", "libero_object": "s",
                "libero_goal": "D", "libero_10": "^"}


def _load_manifest(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "fm" / "manifest.json").read_text())["rollouts"]


def _suite_map_from_manifest(rollouts: list[dict]) -> dict[int, str]:
    """Trust fm/manifest.json's task_group when it actually varies across rollouts
    Returns rollout_id -> suite name."""
    return {int(r["rollout_id"]): str(r["task_group"]) for r in rollouts}


def _suite_map_from_eval_info(run_dir: Path, rollouts: list[dict]) -> dict[int, str] | None:
    """Reconstruct rollout_id → suite from per-worker eval_info_gpu*r*.json.

    Each per-worker file has ``per_task`` (a list of {task_group, task_id, metrics:{successes:[…]}}).
    Concatenating in that order (20 ep/suite, spatial→object→goal→long) yields the worker's
    rollout stream, matching `worker_rollout_id`. Rows in fm/manifest.json carry the same
    (file-prefix, worker_rollout_id) → we look up the suite in that stream."""
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        return None
    per_worker: dict[str, list[str]] = {}
    for pth in sorted(eval_dir.glob("eval_info_gpu*.json")):
        m = re.match(r"eval_info_(gpu\d+r\d+)\.json$", pth.name)
        if not m:
            continue
        wk = m.group(1)
        d = json.loads(pth.read_text())
        pt = d.get("per_task") or []
        if not isinstance(pt, list) or not pt:
            continue
        # concat ep-labels in the order the worker actually ran (spatial→object→goal→10)
        stream: list[str] = []
        for t in pt:
            g = t.get("task_group")
            n = len(t.get("metrics", {}).get("successes", []))
            stream.extend([g] * n)
        per_worker[wk] = stream
    if not per_worker:
        return None
    out: dict[int, str] = {}
    for r in rollouts:
        file_str = str(r.get("file", ""))
        wk = file_str.split("_rollout")[0].split("/")[-1]
        wid = int(r.get("worker_rollout_id", -1))
        stream = per_worker.get(wk)
        if stream is None or not (0 <= wid < len(stream)):
            return None  # incomplete → fall back
        out[int(r["rollout_id"])] = stream[wid]
    return out


def _load_geometry(run_dir: Path) -> dict[int, dict]:
    """Load every per-rollout chunk_geometry_ro*.npz + matching chunk_divergence_ro*.npz.

    Returns {rollout_id → dict of arrays}. Each rollout contributes n_chunks rows to the
    pooled Spearman. The chunk_divergence file provides the resample-GT (max_pairwise_dist
    aggregated to executed sub-chunk) that chunk_geometry has already aligned via
    ``divergence_chunk_step``."""
    geom_dir = run_dir / "chunk_geometry"
    out: dict[int, dict] = {}
    for pth in sorted(geom_dir.glob("chunk_geometry_ro*.npz")):
        m = re.match(r"chunk_geometry_ro(\d+)\.npz$", pth.name)
        if not m:
            continue
        rid = int(m.group(1))
        z = np.load(pth, allow_pickle=True)
        out[rid] = {k: np.asarray(z[k]) for k in z.files}
    return out


def _pooled_stack(rollout_ids: list[int], geom: dict[int, dict], key: str) -> np.ndarray:
    """Concatenate arrays 'key' across rollouts (each rollout: shape (n_chunks_i, ...))."""
    parts = []
    for rid in rollout_ids:
        if rid not in geom or key not in geom[rid]:
            continue
        parts.append(np.asarray(geom[rid][key]))
    if not parts:
        return np.zeros(0, dtype=float)
    return np.concatenate(parts, axis=0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr = _rankdata(x[mask])
    yr = _rankdata(y[mask])
    # Pearson on ranks == Spearman
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank tiebreaker (matches scipy.stats.rankdata default)."""
    a = np.asarray(a).ravel()
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # break ties: replace each run of equal values with the mean rank
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts), dtype=float)
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        ranks = avg[inv]
    return ranks


def _analyze_run(run_dir: Path, out_dir: Path, label: str) -> dict:
    rollouts = _load_manifest(run_dir)
    # 1. rollout_id → suite: manifest first, else reconstruct from per-worker eval_info.
    suite_by_rid = _suite_map_from_manifest(rollouts)
    unique_groups = set(suite_by_rid.values())
    if unique_groups <= {"libero_spatial"} or len(unique_groups) < 2:
        alt = _suite_map_from_eval_info(run_dir, rollouts)
        if alt is not None and len(set(alt.values())) >= 2:
            suite_by_rid = alt
            print(f"[{label}] recovered suite labels from eval_info_gpu*r*.json "
                  f"(manifest task_group was uniform)")
        else:
            raise SystemExit(f"[{label}] cannot recover per-suite mapping: "
                             f"manifest is uniform and eval_info doesn't disambiguate")
    groups = defaultdict(list)
    for rid, suite in suite_by_rid.items():
        groups[suite].append(rid)
    # 2. per-rollout geometry
    geom = _load_geometry(run_dir)
    if not geom:
        raise SystemExit(f"[{label}] no chunk_geometry_ro*.npz files under {run_dir/'chunk_geometry'}")
    # 3. pooled meta (T, action_dim) — needed for the prefix-rho grid.
    meta = json.loads((run_dir / "chunk_geometry" / "meta.json").read_text())
    T = len(meta.get("prefix_n_steps") or []) + 1  # prefix cutoffs = 2..T so len = T-1
    prefix_ns = list(meta.get("prefix_n_steps") or [])
    prefix_ft = list(meta.get("prefix_fm_time") or [])
    act_dim = int(meta.get("action_dim", 0))
    # 4. compute per-suite ρ against the same divergence GT.
    suite_records: dict[str, dict] = {}
    for suite in SUITE_ORDER:
        rids = sorted(groups.get(suite, []))
        if not rids:
            continue
        accel = _pooled_stack(rids, geom, "accel")
        accel_full = _pooled_stack(rids, geom, "accel_full")
        straight = _pooled_stack(rids, geom, "straightness_full")
        div_max = _pooled_stack(rids, geom, "divergence_max_full")
        prefix_accel = _pooled_stack(rids, geom, "prefix_accel")  # (n, T-1)
        accel_action = _pooled_stack(rids, geom, "accel_action")  # (n, first_actions)
        div_action = _pooled_stack(rids, geom, "divergence_action_max")  # (n, first_actions)
        n_chunks = accel.shape[0]
        # headline: rho(accel_exec, div_max) at last prefix cutoff == full accel (accel_full is
        # the whole 0..T path — legacy field). We use `accel` which is the "executed sub-chunk
        # first-fa exec-window" number, matching the pooled meta's run_rho_accel_vs_div.
        rho_full = _spearman(accel, div_max)
        rho_straight = _spearman(straight, div_max)
        # prefix curve
        rho_prefix = []
        if prefix_accel.ndim == 2 and prefix_accel.shape[1] == len(prefix_ns):
            for j in range(prefix_accel.shape[1]):
                rho_prefix.append(_spearman(prefix_accel[:, j], div_max))
        # per-action ρ (chunk × action-position → its own unit)
        rho_action = float("nan")
        if accel_action.size and div_action.size:
            rho_action = _spearman(accel_action.ravel(), div_action.ravel())
        best_prefix_rho = max((r for r in rho_prefix if np.isfinite(r)), default=float("nan"))
        best_prefix_k = None
        if rho_prefix:
            for k, r in zip(prefix_ns, rho_prefix):
                if np.isfinite(r) and r == best_prefix_rho:
                    best_prefix_k = int(k); break
        suite_records[suite] = {
            "n_rollouts": len(rids),
            "n_chunks": int(n_chunks),
            "rho_full": rho_full,
            "rho_straight": rho_straight,
            "rho_action": rho_action,
            "prefix_ns": prefix_ns,
            "prefix_fm_time": prefix_ft,
            "prefix_rho": rho_prefix,
            "best_prefix_rho": best_prefix_rho,
            "best_prefix_k": best_prefix_k,
        }
    # 5. write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_suite_summary.json").write_text(json.dumps({
        "run": str(run_dir), "label": label, "T": T, "action_dim": act_dim,
        "prefix_ns": prefix_ns, "prefix_fm_time": prefix_ft,
        "suites": suite_records,
    }, indent=2))
    _write_md(suite_records, out_dir / "per_suite_summary.md", label=label, T=T, act_dim=act_dim)
    _plot(suite_records, out_dir / "per_suite_prefix_rho", label=label, T=T)
    return suite_records


def _write_md(suite_records: dict, out_path: Path, *, label: str, T: int, act_dim: int) -> None:
    lines: list[str] = []
    lines.append(f"# LIBERO per-suite accel↔divergence ρ — **{label}**  (T={T}, act_dim={act_dim})\n")
    lines.append("Pooled Spearman ρ, one row per suite (same chunk_geometry_ro*.npz — no re-record).")
    lines.append("| Suite | n rollouts | n chunks | **ρ best-prefix** | @ k / T | ρ full-path | ρ action | ρ(straight, div) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for suite in SUITE_ORDER:
        if suite not in suite_records:
            continue
        r = suite_records[suite]
        bpk = "—" if r["best_prefix_k"] is None else f"≤{r['best_prefix_k']}"
        f = lambda x: "—" if not np.isfinite(x) else f"{x:+.3f}"  # noqa: E731
        lines.append(f"| {suite} | {r['n_rollouts']} | {r['n_chunks']} | "
                     f"**{f(r['best_prefix_rho'])}** | {bpk} | "
                     f"{f(r['rho_full'])} | {f(r['rho_action'])} | {f(r['rho_straight'])} |")
    # mean over suites
    rho_bps = [r["best_prefix_rho"] for r in suite_records.values() if np.isfinite(r["best_prefix_rho"])]
    if rho_bps:
        lines.append(f"\n**mean best-prefix ρ over suites: {np.mean(rho_bps):+.3f}** "
                     f"(over {len(rho_bps)} suites)")
    # prefix curve per suite
    lines.append("\n## prefix-k ρ vs divergence (per denoise depth)\n")
    for suite in SUITE_ORDER:
        if suite not in suite_records:
            continue
        r = suite_records[suite]
        cells = []
        bk = r["best_prefix_k"]
        for k, rho in zip(r["prefix_ns"], r["prefix_rho"]):
            s = "—" if not np.isfinite(rho) else (f"**{rho:+.3f}**" if k == bk else f"{rho:+.3f}")
            cells.append(f"k≤{k}: {s}")
        lines.append(f"- **{suite}** (T={T}): " + "  ".join(cells))
    out_path.write_text("\n".join(lines) + "\n")


def _plot(suite_records: dict, out_base: Path, *, label: str, T: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for suite in SUITE_ORDER:
        if suite not in suite_records:
            continue
        r = suite_records[suite]
        ns = np.asarray(r["prefix_ns"], float)
        rho = np.asarray([np.nan if not np.isfinite(x) else x for x in r["prefix_rho"]], float)
        if not len(ns):
            continue
        ax.plot(ns, rho, marker=SUITE_MARKER[suite], color=SUITE_COLOR[suite],
                lw=2.0, ms=6, label=f"{suite} (n={r['n_chunks']}, ρ*={r['best_prefix_rho']:+.3f})")
        if r["best_prefix_k"] is not None:
            ax.plot([r["best_prefix_k"]], [r["best_prefix_rho"]], "*", color=SUITE_COLOR[suite],
                    ms=13, mec="k", mew=0.5, zorder=5)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel(f"denoise steps folded into accel prefix (k, of T={T})")
    ax.set_ylabel(r"Spearman $\rho$ (prefix-accel vs resample divergence)")
    ax.set_title(f"{label} — LIBERO per-suite prefix-accel ρ  (★ = peak)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150)
    plt.close(fig)


# Default cell: the shipped pi05 x LIBERO recording. Pass --run <path-or-id> for any other.
DEFAULT_CELLS = [
    ("pi05_libero", "pi05 - LIBERO", "pi05_libero_all"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="a cell key or a run path/id; default = the pi05 LIBERO cell")
    p.add_argument("--out-root", default="outputs/per_suite_rho",
                   help="root under which each cell's per_suite/ subdir is written")
    p.add_argument("--label", default=None, help="override display label when --run is a raw path")
    args = p.parse_args()

    cells: list[tuple[str, str, str]]
    if args.run is None:
        cells = list(DEFAULT_CELLS)
    else:
        # exact key match?
        by_key = {k: (k, lb, run) for (k, lb, run) in DEFAULT_CELLS}
        if args.run in by_key:
            cells = [by_key[args.run]]
        else:
            lb = args.label or args.run.split("/")[-1]
            cells = [(re.sub(r"[^A-Za-z0-9_]+", "_", lb).strip("_"), lb, args.run)]

    out_root = Path(args.out_root)
    for key, lb, run in cells:
        try:
            rd = _runs.resolve_run(run)
            run_dir = rd.root
        except Exception:
            run_dir = Path(run)
            if not (run_dir / "chunk_geometry" / "meta.json").exists():
                print(f"[{key}] SKIP {run}: no chunk_geometry/meta.json")
                continue
        out_dir = out_root / key / "per_suite"
        print(f"\n[{key}] {run_dir}")
        rec = _analyze_run(run_dir, out_dir, label=lb)
        print(f"  wrote {out_dir/'per_suite_summary.md'}")
        for suite in SUITE_ORDER:
            if suite not in rec:
                continue
            r = rec[suite]
            print(f"    {suite:<18} n={r['n_chunks']:>6}  "
                  f"best-prefix ρ={r['best_prefix_rho']:+.3f} @ k≤{r['best_prefix_k']}  "
                  f"full-path ρ={r['rho_full']:+.3f}")


if __name__ == "__main__":
    main()
