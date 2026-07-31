#!/usr/bin/env python
"""main_exp FD: recompute the online-CUSUM headline at different slack ``k_sigma`` values, from the
already-scored per-episode streams — NO model reload, NO re-record.

Each scored cell's ``detectors/<split>/<Task>.json`` stores every detector's per-chunk score stream
per episode (the fit-held-out episodes are already excluded — this is the exact scored set the
headline table used). CUSUM's only free knob is the slack ``k = k_sigma·σ`` (reference = μ0+k); the
threshold ``h`` is re-calibrated on the SUCCESS peaks for every k, so the target success-FPR stays
matched (no operating-point cheat, no test-label leak). This sweeps ``k_sigma ∈ {0.5, 0.25, 0.1}``
and writes a table + json per k, plus a side-by-side accel comparison.

Usage::
    python experiments/cusum_ksigma_sweep.py
    python experiments/cusum_ksigma_sweep.py --ksigma 0.25 0.1 --fpr 0.1
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap  # noqa: F401,E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("fd_driver", str(Path(__file__).with_name("failure_detection.py")))
_fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fd)
CELLS = _fd.CELLS
cell_key = _fd.cell_key
RUNS_DIR = _fd.RUNS_DIR
DETECTORS = _fd.DETECTORS
OUT_DIR = _fd.OUT_DIR

from fmaccel.detection.cusum import cusum_online  # noqa: E402


def load_streams(det_dir: Path) -> dict[str, dict[str, list[np.ndarray]]]:
    """{detector_name: {'succ':[stream,...], 'fail':[...]}} from every per-task JSON of a cell."""
    out: dict[str, dict[str, list[np.ndarray]]] = {}
    for f in glob.glob(str(det_dir / "*" / "*.json")):
        if f.endswith("meta.json"):
            continue
        d = json.loads(Path(f).read_text())
        for t in d["tasks"].values():
            for e in t["episodes"]:
                ok = bool(e["success"])
                for key, v in e.items():
                    if not key.endswith("_scores"):
                        continue
                    s = np.asarray(v, float)
                    if s.size == 0:
                        continue
                    dn = key[: -len("_scores")]
                    slot = out.setdefault(dn, {"succ": [], "fail": []})
                    slot["succ" if ok else "fail"].append(s)
    return out


def sweep_cell(cell: dict, ksigmas: list[float], fpr: float) -> dict[float, dict[str, dict]]:
    det_dir = RUNS_DIR / cell["run"] / "detectors"
    if not det_dir.exists():
        return {}
    streams = load_streams(det_dir)
    per_k: dict[float, dict[str, dict]] = {}
    for ks in ksigmas:
        per_det: dict[str, dict] = {}
        for dn, slot in streams.items():
            cu = cusum_online(slot["succ"], slot["fail"], fpr=fpr, k_sigma=ks)
            per_det[dn] = cu
        per_k[ks] = per_det
    return per_k


def write_table(ksigma: float, fpr: float, results: dict[str, dict[str, dict]],
                col_keys: list[str], n_info: dict[str, dict]) -> None:
    ordered = [d for d in DETECTORS if d in results] + [d for d in sorted(results) if d not in DETECTORS]

    def _cell(d, ck) -> str:
        cu = results.get(d, {}).get(ck)
        if not cu:
            return "—"
        t, l = cu.get("tpr"), cu.get("median_lead")
        if not isinstance(t, (int, float)) or t != t:
            return "—"
        return f"{t:.2f} / {l:.0f}" if isinstance(l, (int, float)) and l == l else f"{t:.2f}"

    lines = [f"# Failure detection — CUSUM @ FPR={fpr}, **slack k_sigma={ksigma}**", ""]
    lines.append(f"cell = `TPR / median-lead(chunks)`. `h` re-calibrated to success-FPR≈{fpr} at THIS "
                 f"k_sigma (FPR-matched — the only change vs the headline is the CUSUM slack "
                 f"`k = {ksigma}·σ`). accel = method under test; `oracle_resample_spread` = GT upper bound.")
    lines.append("")
    lines.append("**Cell sizes + realized success-FPR (accel):**")
    for ck in col_keys:
        i = n_info[ck]
        rf = results.get("accel", {}).get(ck, {}).get("succ_fire_rate")
        lines.append(f"- `{ck}`: {i['n_episodes']} eps ({i['n_success']} succ / {i['n_fail']} fail)"
                     + (f", realized FPR≈{rf:.3f}" if isinstance(rf, (int, float)) and rf == rf else ""))
    lines.append("")
    lines += ["| detector (TPR / lead) | " + " | ".join(col_keys) + " |",
              "|" + "---|" * (len(col_keys) + 1)]
    for d in ordered:
        mark = " **(ours)**" if d == "accel" else (" [oracle]" if d.startswith("oracle_") else "")
        lines.append(f"| {d}{mark} | " + " | ".join(_cell(d, ck) for ck in col_keys) + " |")
    md = "\n".join(lines) + "\n"
    tag = str(ksigma).replace(".", "p")
    (OUT_DIR / f"fd_table_cusum_k{tag}.md").write_text(md)
    payload = {
        "headline_metric": f"cusum_online@FPR={fpr},k_sigma={ksigma}",
        "cells": {ck: n_info[ck] for ck in col_keys},
        "cusum_tpr": {d: {ck: results.get(d, {}).get(ck, {}).get("tpr") for ck in col_keys} for d in ordered},
        "cusum_median_lead": {d: {ck: results.get(d, {}).get(ck, {}).get("median_lead") for ck in col_keys} for d in ordered},
        "cusum_realized_fpr": {d: {ck: results.get(d, {}).get(ck, {}).get("succ_fire_rate") for ck in col_keys} for d in ordered},
    }
    (OUT_DIR / f"fd_table_cusum_k{tag}.json").write_text(json.dumps(payload, indent=2, default=str))
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ksigma", type=float, nargs="+", default=[0.5, 0.25, 0.1])
    ap.add_argument("--fpr", type=float, default=0.1)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # per detector -> per k -> per cell
    tpr_by_k: dict[float, dict[str, dict[str, dict]]] = {ks: {} for ks in args.ksigma}
    n_info: dict[str, dict] = {}
    col_keys: list[str] = []
    for c in CELLS:
        ck = cell_key(c)
        det_dir = RUNS_DIR / c["run"] / "detectors"
        if not det_dir.exists():
            print(f"[skip] {ck}: not scored")
            continue
        col_keys.append(ck)
        m = json.loads((det_dir / "meta.json").read_text())
        n_info[ck] = {"n_episodes": m.get("n_episodes"), "n_success": m.get("n_success"), "n_fail": m.get("n_fail")}
        per_k = sweep_cell(c, args.ksigma, args.fpr)
        for ks in args.ksigma:
            for dn, cu in per_k.get(ks, {}).items():
                tpr_by_k[ks].setdefault(dn, {})[ck] = cu
        acc = per_k.get(args.ksigma[0], {}).get("accel", {})
        print(f"[{ck}] loaded ({n_info[ck]['n_success']}s/{n_info[ck]['n_fail']}f); "
              f"accel@k{args.ksigma[0]} TPR={acc.get('tpr', float('nan')):.3f}", flush=True)

    for ks in args.ksigma:
        write_table(ks, args.fpr, tpr_by_k[ks], col_keys, n_info)

    # side-by-side accel comparison across k
    print("\n" + "=" * 92)
    print(f"accel CUSUM TPR / lead @ FPR={args.fpr}  across k_sigma  (all FPR-matched, h re-fit per k)")
    print("=" * 92)
    hdr = f"{'cell':<22}{'succ%':>7} | " + " | ".join(f"k={ks:<4}".ljust(14) for ks in args.ksigma)
    print(hdr)
    print("-" * 92)
    means = {ks: [] for ks in args.ksigma}
    for ck in col_keys:
        i = n_info[ck]
        sr = i["n_success"] / i["n_episodes"] if i["n_episodes"] else float("nan")
        row = f"{ck:<22}{sr*100:6.0f}% | "
        cells = []
        for ks in args.ksigma:
            cu = tpr_by_k[ks].get("accel", {}).get(ck, {})
            t, l, fp = cu.get("tpr"), cu.get("median_lead"), cu.get("succ_fire_rate")
            if isinstance(t, (int, float)) and t == t:
                means[ks].append(t)
            cells.append(f"{(t if t is not None else float('nan')):.3f}/{(l if l is not None else float('nan')):.0f}".ljust(13))
        print(row + " | ".join(cells))
    print("-" * 92)
    mrow = f"{'MEAN TPR':<22}{'':>7} | "
    print(mrow + " | ".join(f"{np.mean(means[ks]):.3f}".ljust(13) for ks in args.ksigma))
    print("=" * 92)
    print(f"[written] {OUT_DIR}/fd_table_cusum_k*.{{md,json}} for k_sigma in {args.ksigma}")


if __name__ == "__main__":
    main()
