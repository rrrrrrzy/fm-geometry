#!/usr/bin/env python
"""Post-hoc: score failure-detectors over a recorded run's fm/ (+ resample) into the FD format.

    # accel + geometry from x_t, and ace/oracle from the resample candidates:
    python cli/score.py --run <run-id> \
        --detectors accel,straightness,sparc,ace,oracle_resample_spread

Reads ``<run>/fm/`` (x_t) and ``<run>/chunk_divergence/`` (the k resample candidates; run
``cli/divergence.py`` first for ace/oracle), scores each detector per chunk, writes
``<run>/detectors/<task_group>/<Task>.json`` + ``meta.json`` + ``compare_detectors.png`` and runs
the shared failure_detection_eval battery. Success is a recording proxy (terminated vs truncated);
the resample-based detectors are skipped on rollouts lacking a chunk_divergence npz. Pure analysis
(numpy + the detectors' deps), no GPU / model load. See ``docs/baselines.md``.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from fmaccel.core import args as A
from fmaccel.detection.score import run_detector_score


def _int_list(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x) for x in s.replace(",", " ").split()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    A.add_run(p)
    p.add_argument("--detectors", default="accel,straightness,sparc",
                   help="comma list of detector registry names (accel/straightness/sparc from x_t; "
                        "ace/oracle_resample_spread need <run>/chunk_divergence)")
    p.add_argument("--rollouts", default=None, help="comma/space list of rollout ids (default: all)")
    p.add_argument("--max-rollouts", type=int, default=None, help="cap when --rollouts unset")
    p.add_argument("--n-exec", type=int, default=None,
                   help="executed window for the score (default: min(n_action_steps, chunk_size))")
    p.add_argument("--out", default=None, help="output dir (default: <run>/detectors/)")
    p.add_argument("--device", default="cpu",
                   help="torch device for the LEARNED detectors' fit (rnd_oe/logpzo/safe/fiper); "
                        "use cuda to avoid the multi-hour CPU-Adam. numpy detectors ignore it.")
    p.add_argument("--fit-n-unsup", type=int, default=5,
                   help="ABSOLUTE #rollouts held out to fit the UNSUPERVISED embedding-OOD family "
                        "(rnd_oe/logpzo/pca_kmeans/knn/mahalanobis/fiper), drawn from SUCCESS-only "
                        "rollouts carrying obs_emb and EXCLUDED from scoring (default 5; keeps almost "
                        "every rollout in the scored set). Pass a negative value to fall back to "
                        "round(--fit-frac · #success-with-obs_emb).")
    p.add_argument("--fit-n-sup", type=int, default=10,
                   help="ABSOLUTE #rollouts held out to fit SUPERVISED SAFE, drawn from rollouts "
                        "carrying 'hidden' (spans BOTH classes) and EXCLUDED from scoring (default "
                        "10). SAFE's usable size is bounded by the (scarce) failure count, not this "
                        "number. Pass a negative value to fall back to round(--fit-frac · #hidden). "
                        "IGNORED when --fit-n-sup-succ/--fit-n-sup-fail set a balanced per-class draw.")
    p.add_argument("--fit-n-sup-succ", type=int, default=None,
                   help="EXACT #SUCCESS rollouts for a BALANCED SAFE fit (paper protocol). Setting "
                        "this or --fit-n-sup-fail switches SAFE to a balanced per-class draw; an unset "
                        "side defaults to the other (then 4). Both held out from scoring.")
    p.add_argument("--fit-n-sup-fail", type=int, default=None,
                   help="EXACT #FAILURE rollouts for a BALANCED SAFE fit (paper protocol, e.g. 4).")
    p.add_argument("--fit-frac", type=float, default=0.5,
                   help="fallback held-out fraction per fit family when --fit-n-unsup/--fit-n-sup "
                        "are negative (legacy default 0.5).")
    p.add_argument("--fit-seed", type=int, default=0,
                   help="seed for the held-out fit sample (SAFE uses fit-seed+1 so the two families "
                        "draw independently).")
    p.add_argument("--resample-k", type=int, default=4,
                   help="MC candidates used by resample detectors, selected from the saved "
                        "chunk_divergence candidate axis (default 4; <=0 uses all saved candidates).")
    p.add_argument("--resample-seed", type=int, default=0,
                   help="seed for the deterministic saved-candidate subset (default 0).")
    p.add_argument("--cusum-calib-n", type=int, default=10,
                   help="SUCCESS episodes held out to calibrate the CUSUM threshold (default 10, "
                        "Sentinel's M~10-50 deployment protocol). They are excluded from the CUSUM "
                        "eval population, so the reported success fire-rate is a MEASURED "
                        "out-of-sample FPR. 0 = legacy in-sample calibration (fire-rate == target "
                        "by construction).")
    p.add_argument("--cusum-calib-draws", type=int, default=20,
                   help="independent M-episode calibration draws to average the CUSUM layer over "
                        "(default 20). A single M=10 threshold is high-variance (TPR +-0.09, "
                        "realized FPR spanning [0.00,0.31]); the draw is independent of scoring so "
                        "averaging is free. 1 = a single draw.")
    p.add_argument("--cusum-calib-seed", type=int, default=None,
                   help="seed for the held-out CUSUM calibration draw (default: follow --fit-seed, "
                        "so each repeat gets a different calibration set and the error bars "
                        "absorb calibration variance).")
    p.add_argument("--accel-fixed-std", default="auto",
                   help="accel z-score reference scale: 'auto' (default) = run-pooled std over this "
                        "recording's chunks (matches the validated label scale; self-norm depresses "
                        "AUROC ~0.08); 'self' = legacy per-chunk self-normalization; or a path to an "
                        "external std npz (e.g. accel_fixed_std_catA.npz).")
    args = p.parse_args()

    if args.accel_fixed_std == "auto":
        accel_fixed_std: object = True
    elif args.accel_fixed_std in ("self", "none", "off"):
        accel_fixed_std = None
    else:
        accel_fixed_std = args.accel_fixed_std  # path to an external npz

    detectors = [d for d in args.detectors.replace(",", " ").split()]
    # negative -> None sentinel = fall back to the round(fit_frac · pool) fraction path
    fit_n_unsup = None if args.fit_n_unsup < 0 else args.fit_n_unsup
    fit_n_sup = None if args.fit_n_sup < 0 else args.fit_n_sup
    rd, summary = run_detector_score(
        args.run, detectors=detectors, n_exec=args.n_exec,
        rollouts=_int_list(args.rollouts), max_rollouts=args.max_rollouts, out_dir=args.out,
        device=args.device, accel_fixed_std=accel_fixed_std,
        fit_n_unsup=fit_n_unsup, fit_n_sup=fit_n_sup,
        fit_n_sup_succ=args.fit_n_sup_succ, fit_n_sup_fail=args.fit_n_sup_fail,
        fit_frac=args.fit_frac, fit_seed=args.fit_seed,
        resample_k=None if args.resample_k <= 0 else args.resample_k,
        resample_seed=args.resample_seed,
        cusum_calib_n=args.cusum_calib_n, cusum_calib_seed=args.cusum_calib_seed,
        cusum_calib_draws=args.cusum_calib_draws,
    )
    print("out_dir:", args.out or (rd.root / "detectors"))
    if not summary:
        print("(no scoreable detectors / no episodes — check fm/ and chunk_divergence/)")
        return
    print(f"\n{summary['n_episodes']} eps ({summary['n_success']} succ / {summary['n_fail']} fail) "
          f"over groups {summary['splits']}")
    for key, r in summary["detectors"].items():
        oracle = " [ORACLE/upper-bound, NOT a baseline]" if key.startswith("oracle_") else ""
        cu = r.get("cusum_online", {}).get("0.1", {})
        cu_str = ""
        if cu:
            cu_str = (f"  | CUSUM TPR={cu.get('tpr', float('nan')):.3f}@FPR{cu.get('fpr', 0.1):g} "
                      f"lead={cu.get('median_lead', float('nan')):.0f}ch")
        print(f"  {key:<28} whole-ep AUROC={r['whole_ep_auroc']:.3f}  "
              f"within-task={r['within_task_auroc_pairweighted']:.3f}  "
              f"TPR@0.1={r['tpr_at_fpr'].get('0.1', float('nan')):.3f}{cu_str}{oracle}")
    if summary.get("plot"):
        print("plot:", summary["plot"])


if __name__ == "__main__":
    main()
