#!/usr/bin/env python
"""Score per-chunk failure detectors on closed-loop eval JSONs — one harness for all.

    # accel reference (catA closed-loop; reproduces ~0.82 pooled / ~0.88 within-task):
    python cli/detect.py \
        --data-root outputs/runs/<run-id>/detector_score \
        --score-key chunk_accels --out outputs/failure_detection/catA_accel

    # compare several detectors written into the same episodes' JSON:
    python cli/detect.py --data-root <dir> \
        --score-key chunk_accels --score-key stac_scores --score-key fm_loss_scores

Pure analysis (numpy + matplotlib, no GPU): reads ``<root>/<split>/<Task>.json`` with
``tasks.<name>.episodes[].{success, n_chunks, <score_key>}`` and reports, per score key,
whole-episode / within-task / early-window AUROC, TPR@FPR, and the length confound. See
``fmaccel.detection.cusum`` and ``docs/baselines.md``.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from fmaccel.detection.cusum import run_failure_detection_eval


def _float_list(s: str | None) -> list[float] | None:
    if not s:
        return None
    return [float(x) for x in s.replace(",", " ").split()]


def _int_list(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x) for x in s.replace(",", " ").split()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", action="append", required=True, dest="data_roots",
                   help="dir containing <split>/<Task>.json (repeatable to pool runs)")
    p.add_argument("--score-key", action="append", dest="score_keys", default=None,
                   help="per-chunk stream key to score (repeatable; default chunk_accels)")
    p.add_argument("--splits", default="target,pretrain", help="comma list of split subdirs")
    p.add_argument("--agg", default="mean", choices=("mean", "max", "last"),
                   help="whole-episode aggregation of the per-chunk stream")
    p.add_argument("--early-ks", default="1,2,3,5,8,10,15", help="early-window sizes (chunks)")
    p.add_argument("--fpr", default="0.1", help="comma list of target FPRs for TPR@FPR")
    p.add_argument("--out", default=None, help="output dir for meta.json + compare plot")
    p.add_argument("--label", default=None, help="title/label for the summary + plot")
    p.add_argument("--no-plot", action="store_true", help="skip the comparison plot")
    args = p.parse_args()

    score_keys = args.score_keys or ["chunk_accels"]
    summary = run_failure_detection_eval(
        args.data_roots,
        score_keys=score_keys,
        splits=[s for s in args.splits.replace(",", " ").split()],
        agg=args.agg,
        early_ks=_int_list(args.early_ks) or (1, 2, 3, 5, 8, 10, 15),
        fpr_targets=_float_list(args.fpr) or (0.1,),
        out_dir=args.out,
        label=args.label,
        plot=not args.no_plot,
    )

    print(f"\n{'='*78}")
    print(f"failure-detection eval: {summary['n_episodes']} eps "
          f"({summary['n_success']} succ / {summary['n_fail']} fail) "
          f"over splits {summary['splits']}")
    print('='*78)
    for key, r in summary["detectors"].items():
        print(f"\n[{key}]  n={r['n_episodes']} (sr={r['success_rate']:.3f})  agg={r['agg']}")
        print(f"  [A] whole-ep AUROC = {r['whole_ep_auroc']:.3f}   Cohen_d = {r['whole_ep_cohen_d']:+.3f}   "
              f"(succ {r['succ_mean']:.4f} vs fail {r['fail_mean']:.4f})")
        print(f"  [B] n_chunks median: succ {r['succ_median_chunks']:.0f}  fail {r['fail_median_chunks']:.0f}  "
              f"({'failures longer → whole-ep is partly hindsight' if r['fail_median_chunks'] > r['succ_median_chunks'] else 'comparable'})")
        print(f"  [C] within-task AUROC = {r['within_task_auroc_pairweighted']:.3f} (pair-weighted)  "
              f"mean {r['within_task_auroc_mean']:.3f}  "
              f"[{r['n_cells_pointing']}/{r['n_cells']} cells point the right way]")
        tpr = "  ".join(f"TPR@{fp}FPR={v:.3f}" for fp, v in r["tpr_at_fpr"].items())
        print(f"      {tpr}")
        if r["early_window"]:
            print(f"  [D] early-window (survival-conditioned):  "
                  + "  ".join(f"k={row['k']}:{row['auroc']:.3f}" for row in r["early_window"]))
    if summary.get("plot"):
        print(f"\n[plot] {summary['plot']}")
    if summary.get("meta"):
        print(f"[meta] {summary['meta']}")


if __name__ == "__main__":
    main()
