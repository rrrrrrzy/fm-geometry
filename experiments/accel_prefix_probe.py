#!/usr/bin/env python
"""main_exp FD probe: re-score **only accel** at a chosen denoise-depth prefix cut and compare
its online-CUSUM headline against the validated default (``accel_prefix:7`` → auto ``:-1`` on
shallow models).

Motivation: in this repo "prefix" is a DENOISE-depth cut of the growing noise→t path
(``accel_prefix:<j>`` = accel of the first ``j+2`` flow steps of the executed window, each cut
self-contained). The headline detector uses the deep default ``j=7`` (falls back to the last
layer ``:-1`` == ``accel_exec``). The prefix-ρ
curve found ρ(prefix-accel, resample-divergence) peaks EARLY (near the noise end), so a shallower
cut may be a better uncertainty proxy — this probes the **2nd cut** ``accel_prefix:1`` (accel over
the first 3 denoise iterates) on all 6 cells.

accel is training-free (no fit families, no GPU) → this reads each cell's stored ``fm/`` x_t and
re-scores in seconds, writing a NON-DESTRUCTIVE ``detectors_<tag>/`` dir per run (the validated
``detectors/`` is untouched). Prints a side-by-side CUSUM TPR@FPR / median-lead table.

Usage::

    python experiments/accel_prefix_probe.py --mode accel_prefix:1
    python experiments/accel_prefix_probe.py --mode accel_prefix:1 --cell pi05:libero_all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap  # noqa: F401,E402

# Reuse the headline driver's CELLS table + gold-label recovery so labels/windows stay identical.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("fd_driver", str(Path(__file__).with_name("failure_detection.py")))
_fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fd)

CELLS = _fd.CELLS
cell_key = _fd.cell_key
gold_labels = _fd.gold_labels
resolve_cell = _fd.resolve_cell
RUNS_DIR = _fd.RUNS_DIR


def _mode_tag(mode: str) -> str:
    return mode.replace(":", "").replace("-", "m")  # accel_prefix:1 -> accel_prefix1 ; :-1 -> :-1 -> ...m1


def rescore_cell(cell: dict, *, mode: str, device: str) -> dict | None:
    from fmaccel.detection.score import run_detector_score

    run_dir = RUNS_DIR / cell["run"]
    if not run_dir.exists():
        print(f"[skip] {cell_key(cell)}: run dir missing {run_dir}")
        return None
    succ = gold_labels(cell, run_dir)  # None for LIBERO (built-in terminated proxy, verified == eval_info)
    out = run_dir / f"detectors_{_mode_tag(mode)}"
    _rd, summary = run_detector_score(
        str(run_dir), detectors=["accel"], n_exec=cell["first_actions"],
        success_by_rollout=succ, out_dir=out, device=device,
        accel_fixed_std=True, accel_mode=mode,
        label=f"{cell_key(cell)}: accel @ {mode}")
    if not summary or "accel_scores" not in summary["detectors"]:
        print(f"[warn] {cell_key(cell)}: accel produced no score at {mode}")
        return None
    r = summary["detectors"]["accel_scores"]
    cu = r.get("cusum_online", {}).get("0.1", {})
    print(f"[{cell_key(cell)}] accel@{mode}: "
          f"CUSUM TPR={cu.get('tpr', float('nan')):.3f} @FPR={cu.get('succ_fire_rate', float('nan')):.2f} "
          f"lead={cu.get('median_lead', float('nan')):.0f}ch  (AUROC={r['whole_ep_auroc']:.3f}, "
          f"realized-mode={r.get('agg')})", flush=True)
    return {"cell": cell_key(cell), "auroc": r["whole_ep_auroc"], "cohen_d": r["whole_ep_cohen_d"],
            "cusum": cu, "n_success": summary["n_success"], "n_fail": summary["n_fail"]}


def _default_accel_cusum(cell: dict) -> dict | None:
    """The validated default-accel CUSUM already stored under detectors/meta.json (for the compare)."""
    p = RUNS_DIR / cell["run"] / "detectors" / "meta.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    r = m["detectors"].get("accel_scores")
    if not r:
        return None
    return {"auroc": r["whole_ep_auroc"], "cohen_d": r.get("whole_ep_cohen_d"),
            "cusum": r.get("cusum_online", {}).get("0.1", {})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="accel_prefix:1",
                    help="accel denoise-depth prefix mode to probe (default accel_prefix:1 = 2nd cut)")
    ap.add_argument("--cell", help="only this cell (model:bench); default all 6")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fpr", default="0.1", help="CUSUM target FPR key to report")
    args = ap.parse_args()

    cells = [resolve_cell(args.cell)] if args.cell else CELLS
    rows = []
    for c in cells:
        probe = rescore_cell(c, mode=args.mode, device=args.device)
        if probe is None:
            continue
        base = _default_accel_cusum(c)
        rows.append((c, probe, base))

    # side-by-side table: default accel_prefix:7 (auto :-1 on shallow) vs the probed cut
    k = args.fpr
    print("\n" + "=" * 96)
    print(f"accel CUSUM @ FPR={k}   —   default (prefix:7/auto) vs probe ({args.mode})")
    print("=" * 96)
    hdr = f"{'cell':<22}{'succ%':>7} | {'def TPR':>8}{'def lead':>9} | {'new TPR':>8}{'new lead':>9}  ΔTPR"
    print(hdr)
    print("-" * 96)
    for c, probe, base in rows:
        sr = probe["n_success"] / (probe["n_success"] + probe["n_fail"])
        bt = (base or {}).get("cusum", {}).get("tpr")
        bl = (base or {}).get("cusum", {}).get("median_lead")
        nt = probe["cusum"].get("tpr")
        nl = probe["cusum"].get("median_lead")
        dstr = f"{(nt - bt):+.3f}" if (bt is not None and nt is not None) else "  n/a"
        print(f"{cell_key(c):<22}{sr*100:6.0f}% | "
              f"{(bt if bt is not None else float('nan')):8.3f}{(bl if bl is not None else float('nan')):9.0f} | "
              f"{(nt if nt is not None else float('nan')):8.3f}{(nl if nl is not None else float('nan')):9.0f}  {dstr}")
    print("=" * 96)

    outp = REPO / "outputs" / "main_exp" / "failure_detection" / f"accel_probe_{_mode_tag(args.mode)}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "probe_mode": args.mode, "fpr": k,
        "rows": [{"cell": cell_key(c), "probe": probe, "default": base} for c, probe, base in rows],
    }, indent=2))
    print(f"[written] {outp}")


if __name__ == "__main__":
    main()
