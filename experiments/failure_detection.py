#!/usr/bin/env python
"""**Online failure detection** (Table 2): the free geometric scores vs the literature.

Runs the detector battery over a closed-loop ``--record-fm --record-context`` recording, turns
each detector's per-decision score stream into alarms with a one-sided CUSUM whose height is
calibrated by split conformal prediction on held-out *successful* episodes, and reports
TPR at a target FPR together with the median detection lead — then assembles the table and the
comparison figure.

Detectors compared (14 registered; the paper reports 9):

  * **ours, free** — ``accel`` (the method under test) and ``straightness``;
  * **resample-based** — ``ace`` (action-chunk entropy), ``stac`` (temporal MMD), ``fm_loss``
    (Diff-DAgger re-noised flow loss);
  * **training-based** — ``rnd_oe``, ``logpzo``, ``fiper``, and the supervised ``safe`` probe;
  * ``oracle_resample_spread`` is the ground-truth upper bound, **not** a baseline.
  * ``sparc`` / ``pca_kmeans`` / ``knn`` / ``mahalanobis`` are additional ablation surface.

Scoring is the lerobot-free post-hoc :func:`fmaccel.detection.score.run_detector_score`: it reads
each run's ``fm/`` (the recorded Euler iterates), ``chunk_divergence/`` (the K resample
candidates), and the ``obs_emb/`` / ``hidden_states/`` / ``fm_loss/`` capture-stage outputs, and
needs torch only for the learned-detector fits. So a cell scores without the policy installed —
only the capture stages that produced those directories need the model.

**Fit protocol (CLI-configurable).** The unsupervised embedding-OOD family
(``rnd_oe``/``logpzo``/``pca_kmeans``/``knn``/``mahalanobis``/``fiper``) is calibrated on
successful rollouts; supervised ``safe`` on a balanced success+failure split. Every fit set is
held out from scoring, and the training-free detectors are scored on that same
fit-excluded episode set within each repeat, so all method comparisons stay paired.

**Success labels.** LIBERO cells map the true per-episode ``successes`` from
``eval/eval_info.json`` onto rollouts by (task_group, task_id, worker order), asserting the count
matches the recording's ``terminated`` flags. A cell may instead supply a ``labels.json`` sidecar
(``{"<rollout_id>": true/false}``) for recordings whose harness never sees env ``terminated``.

This release ships the **π₀.₅ × LIBERO** cell. The paper additionally reports SmolVLA, GR00T N1.7
and VLA-JEPA on LIBERO and RoboCasa Atomic-Seen; add such a cell with ``--register-cell`` (or an
entry in ``CELLS``) once the corresponding recording exists.

Usage::

    # score the default cell with five independent repeat seeds:
    python experiments/failure_detection.py score --cell pi05:libero_all --device cuda
    python experiments/failure_detection.py score --all --device cuda

    # assemble mean ± sample-std tables + figure from every repeat/cell:
    python experiments/failure_detection.py aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: F401,E402

# The 14 detectors of the battery (accel = method under test; oracle = GT upper bound, not a
# baseline). Order here == the row order in the assembled table.
DETECTORS = [
    "accel", "straightness", "sparc",            # free geometric proxies (accel = ours)
    "ace", "stac",                               # resample-posterior baselines
    "fm_loss",                                   # Diff-DAgger (model-in-the-loop, precomputed)
    "rnd_oe", "logpzo", "pca_kmeans", "knn", "mahalanobis",  # embedding-OOD family
    "fiper", "safe",                             # assembled + supervised probe
    "oracle_resample_spread",                    # ORACLE / upper bound (NOT a baseline)
]

_OUT_ROOT = Path(os.environ.get("FMACCEL_OUTPUT_DIR", REPO / "outputs"))
RUNS_DIR = _OUT_ROOT / "runs"
OUT_DIR = _OUT_ROOT / "failure_detection"
# Resampled candidate chunks per decision, consumed from the chunk_divergence stage. The paper
# reports K=32; override with --resample-k.
DEFAULT_RESAMPLE_K = 32
DEFAULT_REPEATS = 5
# SUCCESS episodes held out per cell to calibrate the CUSUM alarm height by split conformal
# prediction on the episode-level CUSUM peak. Excluded from the eval population, so the reported
# success fire-rate is a MEASURED out-of-sample FPR, not the in-sample target it was pinned to.
# The paper uses M=50.
DEFAULT_CUSUM_CALIB_N = 50
# Independent M-episode draws averaged per repeat (free: the draw never touches scoring).
DEFAULT_CUSUM_CALIB_DRAWS = 20

# One row per (model, benchmark) cell. `first_actions` = the executed action window
# (== n_action_steps of the recording; both geometric scores are computed over this window only,
# not the full action horizon). `label_src` picks the gold-label recovery path: "eval_info" reads
# eval/eval_info.json, "sidecar" reads a labels.json written next to the run.
#
# This release ships the π₀.₅ × LIBERO cell. Add more with --register-cell
# model:bench:run[:first_actions[:label_src]] , or append to this list.
CELLS: list[dict] = [
    dict(model="pi05", bench="libero_all",
         run="pi05_libero_all",
         first_actions=10, label_src="eval_info"),
]

MODEL_LABEL = {"pi05": "pi05"}
BENCH_LABEL = {"libero_all": "LIBERO"}


def cell_key(c: dict) -> str:
    return f"{c['model']}:{c['bench']}"


def resolve_cell(name: str) -> dict:
    for c in CELLS:
        if cell_key(c) == name:
            return c
    raise SystemExit(f"unknown cell {name!r}; available: {[cell_key(c) for c in CELLS]}")


def _cell_slug(cell: dict) -> str:
    return cell_key(cell).replace(":", "__")


def register_cell(spec: str) -> dict:
    """Add a cell from ``model:bench:run[:first_actions[:label_src]]``.

    The extension point for the model×benchmark cells this release does not ship: point it at a
    recording that has been through the record + capture stages and it scores like any other cell
    (every detector reads the on-disk recording, not a live policy)."""
    parts = spec.split(":")
    if len(parts) < 3:
        raise SystemExit(
            f"--register-cell needs model:bench:run[:first_actions[:label_src]], got {spec!r}")
    model, bench, run = parts[0], parts[1], parts[2]
    cell = dict(model=model, bench=bench, run=run,
                first_actions=int(parts[3]) if len(parts) > 3 and parts[3] else 0,
                label_src=parts[4] if len(parts) > 4 and parts[4] else "eval_info")
    CELLS.append(cell)
    MODEL_LABEL.setdefault(model, model)
    BENCH_LABEL.setdefault(bench, bench)
    return cell


def _tree_name(resample_k: int, calib_n: int | None = None) -> str:
    """Result-tree directory name. A non-default CUSUM calibration size gets its OWN tree
    (``k32_M20``) so a re-calibration can never overwrite the canonical table's inputs."""
    base = f"k{int(resample_k)}"
    if calib_n is not None and int(calib_n) != int(DEFAULT_CUSUM_CALIB_N):
        base += f"_M{int(calib_n)}"
    return base


def _repeat_dir(cell: dict, *, resample_k: int, repeat: int,
                calib_n: int | None = None) -> Path:
    """Self-contained score output for one cell/repeat, always under the requested result root."""
    return (OUT_DIR / "repeats" / _tree_name(resample_k, calib_n)
            / f"repeat_{int(repeat):02d}" / _cell_slug(cell))


def _repeat_complete(cell: dict, *, resample_k: int, repeat: int,
                     detectors: list[str]) -> bool:
    """Whether an existing repeat is complete for the requested detector set and protocol."""
    p = _repeat_dir(cell, resample_k=resample_k, repeat=repeat) / "meta.json"
    if not p.exists():
        return False
    try:
        meta = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    protocol = meta.get("detector_score_protocol", {})
    if protocol.get("resample_k") != int(resample_k):
        return False
    if protocol.get("cusum_calib_n") != int(DEFAULT_CUSUM_CALIB_N):
        return False   # pre-held-out-calibration repeats are stale; re-score rather than mix
    produced = set(meta.get("detectors", {}))
    return all(f"{name}_scores" in produced for name in detectors)


# --------------------------------------------------------------------------------------------
# gold-label recovery
# --------------------------------------------------------------------------------------------
def _labels_from_eval_info(run_dir: Path) -> dict[int, bool] | None:
    """LIBERO gold labels. LIBERO terminates the episode on task success, so the per-rollout
    ``terminated`` flag the recorder already stored IS the gold label — detector_score's built-in
    ``_episode_success`` proxy reads exactly that. We therefore return ``None`` (let the built-in
    proxy label) AFTER asserting the run's terminated-success COUNT matches ``eval_info.json``'s
    ``successes`` total, so the gold labels are verified without depending on the (8-GPU-merge-
    scrambled) per-rollout ``task_group``/``task_id`` ordering. A count mismatch aborts."""
    import numpy as np

    man = json.loads((run_dir / "fm" / "manifest.json").read_text())
    ei = json.loads((run_dir / "eval" / "eval_info.json").read_text())
    ei_succ = sum(sum(bool(x) for x in t["metrics"]["successes"]) for t in ei["per_task"])
    ei_tot = sum(len(t["metrics"]["successes"]) for t in ei["per_task"])
    term_succ = 0
    n_recorded_eps = 0
    for r in man["rollouts"]:
        with np.load(run_dir / "fm" / r["file"]) as d:
            t = np.asarray(d["terminated"], bool)
            batch = int(r.get("batch_size", t.shape[1] if t.ndim > 1 else 1))
            if t.ndim == 1:
                t = t[:, None]
            term_succ += sum(bool(t[:, env_idx].any()) for env_idx in range(batch))
            n_recorded_eps += batch
    if term_succ != ei_succ or n_recorded_eps != ei_tot:
        raise SystemExit(
            f"{run_dir.name}: terminated-proxy success {term_succ}/{n_recorded_eps} != "
            f"eval_info {ei_succ}/{ei_tot} — the built-in per-rollout `terminated` label does NOT "
            "match the eval loop; refusing to score with an unverified label.")
    print(f"[labels] {run_dir.name}: gold verified — terminated-success {term_succ}/{ei_tot} "
          f"== eval_info; using the built-in per-rollout terminated label.")
    return None  # None -> detector_score uses its built-in terminated proxy (== gold, just verified)


def _labels_from_sidecar(run_dir: Path) -> dict[int, bool]:
    """Gold labels from a ``<run>/labels.json`` sidecar
    (``{rollout_id: success}``), written by the atomic re-record launcher after mapping each
    shard's sim-client success by (shard, worker order) with the n_chunks checksum."""
    p = run_dir / "labels.json"
    if not p.exists():
        raise SystemExit(
            f"{run_dir.name}: no gold success labels (expected {p}). A two-process "
            "recorder never sees env `terminated`, so a re-record with the sim-client success "
            "sidecar is required — run the atomic re-record launcher first. (No noisy proxy in "
            "the paper path.)")
    raw = json.loads(p.read_text())
    return {int(k): bool(v) for k, v in raw.items()}


def gold_labels(cell: dict, run_dir: Path) -> dict[int, bool] | None:
    """Gold success labels for a cell, or ``None`` when the recording's own ``terminated`` flags
    are authoritative (:func:`_labels_from_eval_info` verifies they match ``eval_info.json``
    before returning ``None``)."""
    if cell["label_src"] == "eval_info":
        return _labels_from_eval_info(run_dir)
    return _labels_from_sidecar(run_dir)


# --------------------------------------------------------------------------------------------
# score one cell
# --------------------------------------------------------------------------------------------
def score_cell(cell: dict, *, device: str, detectors: list[str], repeat: int,
               resample_k: int, resample_seed: int, fit_seed: int,
               fit_n_unsup: int, fit_n_sup_succ: int, fit_n_sup_fail: int) -> dict:
    from fmaccel.detection.score import run_detector_score

    run_dir = RUNS_DIR / cell["run"]
    if not run_dir.exists():
        raise SystemExit(f"cell {cell_key(cell)}: run dir missing: {run_dir}")
    succ = gold_labels(cell, run_dir)   # None => use the recording's terminated flags (verified above)
    if succ is not None:
        n_s = sum(succ.values())
        print(f"[{cell_key(cell)}] {len(succ)} rollouts labeled (sidecar), {n_s} success / "
              f"{len(succ)-n_s} fail ({n_s/len(succ):.3f})", flush=True)

    out = _repeat_dir(cell, resample_k=resample_k, repeat=repeat)
    # Pass the FULL run path rather than the bare run-id, so the capture stages
    # (obs_emb / hidden_states / fm_loss) are read from this exact recording. resolve_run accepts
    # either; see experiments/fd_gpu_stages.sh, which produces them in place.
    rd, summary = run_detector_score(
        str(run_dir), detectors=detectors, n_exec=cell["first_actions"],
        success_by_rollout=succ, out_dir=out, device=device,
        accel_fixed_std=True,                                   # auto run-pooled std (validated scale)
        fit_n_unsup=fit_n_unsup, fit_n_sup_succ=fit_n_sup_succ, fit_n_sup_fail=fit_n_sup_fail,
        fit_seed=fit_seed, resample_k=resample_k, resample_seed=resample_seed,
        cusum_calib_n=DEFAULT_CUSUM_CALIB_N, cusum_calib_draws=DEFAULT_CUSUM_CALIB_DRAWS,
        label=(f"{cell_key(cell)} repeat={repeat}: detector battery, resample k={resample_k} "
               f"({fit_n_unsup}-succ unsup / {fit_n_sup_succ}+{fit_n_sup_fail} SAFE, "
               f"{DEFAULT_CUSUM_CALIB_N}-succ held-out CUSUM calibration)"))
    if not summary:
        raise SystemExit(f"cell {cell_key(cell)}: no scoreable detectors — check the capture stages.")
    print(f"[{cell_key(cell)} r{repeat}] {summary['n_episodes']} eps "
          f"({summary['n_success']} succ / {summary['n_fail']} fail) over {summary['splits']}", flush=True)
    # Print the CUSUM headline (the reported metric); AUROC de-emphasized (appendix only).
    for key, r in sorted(summary["detectors"].items(),
                         key=lambda kv: -(kv[1].get("cusum_online", {}).get("0.1", {}).get("tpr") or -1)):
        cu = r.get("cusum_online", {}).get("0.1", {})
        oracle = " [ORACLE]" if key.startswith("oracle_") else ""
        if cu:
            print(f"    {key:<30} CUSUM TPR={cu.get('tpr', float('nan')):.3f} @FPR={cu.get('succ_fire_rate', float('nan')):.2f} "
                  f"lead={cu.get('median_lead', float('nan')):.0f}ch  (AUROC={r['whole_ep_auroc']:.3f}){oracle}", flush=True)
        else:
            print(f"    {key:<30} (no CUSUM)  (AUROC={r['whole_ep_auroc']:.3f}){oracle}", flush=True)
    return summary


# --------------------------------------------------------------------------------------------
# recalibrate: re-run ONLY the CUSUM layer of an existing tree at a different calibration size
# --------------------------------------------------------------------------------------------
def recalibrate(*, calib_n: int, repeats: int = DEFAULT_REPEATS,
                resample_k: int = DEFAULT_RESAMPLE_K,
                src_calib_n: int = DEFAULT_CUSUM_CALIB_N) -> None:
    """Re-derive the CUSUM layer of the ``src_calib_n`` tree at ``calib_n`` held-out successes.

    Detector scoring is the expensive part (it re-reads each run's whole ``fm/`` directory); the
    calibration split touches none of it — it only decides which already-computed per-episode score
    streams estimate the threshold. So an M-ablation reuses the scored tree verbatim and costs
    seconds, and, because the streams are literally identical, the comparison isolates M with no
    scoring noise mixed in.

    Writes a parallel tree ``repeats/k<k>_M<n>/`` plus ``fd_table_cusum_M<n>.{md,json}``; the
    canonical M=10 artifacts are never touched.
    """
    from fmaccel.detection.cusum import run_failure_detection_eval

    if calib_n == src_calib_n:
        raise SystemExit(f"--calib-n {calib_n} equals the source tree's M; nothing to re-derive")
    n_done = 0
    for repeat in range(repeats):
        for c in CELLS:
            src = _repeat_dir(c, resample_k=resample_k, repeat=repeat, calib_n=src_calib_n)
            meta_p = src / "meta.json"
            if not meta_p.exists():
                raise SystemExit(f"source scoring missing: {meta_p} — run `score` first")
            src_meta = json.loads(meta_p.read_text())
            dst = _repeat_dir(c, resample_k=resample_k, repeat=repeat, calib_n=calib_n)
            dst.mkdir(parents=True, exist_ok=True)
            summary = run_failure_detection_eval(
                [src], score_keys=sorted(src_meta.get("detectors", {})),
                splits=src_meta.get("splits", []), out_dir=dst, plot=False,
                cusum_calib_n=int(calib_n),
                # Same seed as the source repeat, so the M=10 and M=50 draws are nested in
                # distribution rather than an independent roll -> the delta is M, not luck.
                cusum_calib_seed=int(src_meta.get("detector_score_protocol", {})
                                     .get("cusum_calib_seed", repeat)),
                cusum_calib_draws=DEFAULT_CUSUM_CALIB_DRAWS,
                label=f"{cell_key(c)} repeat={repeat}: recalibrated M={calib_n} (streams from M={src_calib_n})")
            # Carry the scoring protocol forward verbatim (aggregate validates resample_k against
            # it) and stamp what this pass actually changed.
            proto = dict(src_meta.get("detector_score_protocol", {}))
            proto["cusum_calib_n"] = int(calib_n)
            proto["recalibrated_from"] = str(src.relative_to(OUT_DIR))
            summary["detector_score_protocol"] = proto
            (dst / "meta.json").write_text(json.dumps(summary, indent=2, default=str))
            n_done += 1
            cu = summary["detectors"].get("accel_scores", {}).get("cusum_online", {}).get("0.1", {})
            print(f"[{cell_key(c)} r{repeat}] M={calib_n}: accel TPR={cu.get('tpr', float('nan')):.3f}"
                  f" (draw-std {cu.get('tpr_std', float('nan')):.3f})"
                  f" realizedFPR={cu.get('succ_fire_rate', float('nan')):.3f}", flush=True)
    print(f"\nrecalibrated {n_done} cell-repeats -> repeats/{_tree_name(resample_k, calib_n)}/")


# --------------------------------------------------------------------------------------------
# aggregate every scored cell -> cross-model x cross-benchmark tables + figure
# --------------------------------------------------------------------------------------------
def _read_cell_meta(cell: dict, *, resample_k: int, repeat: int,
                    calib_n: int | None = None) -> dict | None:
    p = _repeat_dir(cell, resample_k=resample_k, repeat=repeat, calib_n=calib_n) / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _det_name(score_key: str) -> str:
    return score_key[:-len("_scores")] if score_key.endswith("_scores") else score_key


def _repeat_stat(values) -> dict:
    """JSON-safe mean and sample std (ddof=1) over finite repeat values."""
    import numpy as np

    raw = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        raw.append(v if np.isfinite(v) else None)
    finite = np.asarray([v for v in raw if v is not None], float)
    return {
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std(ddof=1)) if finite.size > 1 else (0.0 if finite.size == 1 else None),
        "n": int(finite.size),
        "values": raw,
    }


def aggregate(fpr: str = "0.1", *, repeats: int = DEFAULT_REPEATS,
              resample_k: int = DEFAULT_RESAMPLE_K,
              calib_n: int = DEFAULT_CUSUM_CALIB_N) -> None:
    """Aggregate every cell/repeat and report CUSUM mean ± sample standard deviation."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if repeats < 1:
        raise SystemExit(f"repeats must be >=1, got {repeats}")
    if resample_k < 1:
        raise SystemExit(f"resample-k must be >=1, got {resample_k}")

    by_cell: dict[str, list[dict]] = {}
    missing: list[str] = []
    for c in CELLS:
        ck = cell_key(c)
        metas = []
        for repeat in range(repeats):
            m = _read_cell_meta(c, resample_k=resample_k, repeat=repeat, calib_n=calib_n)
            if m is None:
                missing.append(f"{ck}/repeat_{repeat:02d}")
            else:
                protocol = m.get("detector_score_protocol", {})
                if protocol.get("resample_k") != resample_k:
                    raise SystemExit(
                        f"{ck}/repeat_{repeat:02d}: meta resample_k={protocol.get('resample_k')} "
                        f"!= requested {resample_k}")
                metas.append(m)
        if len(metas) == repeats:
            by_cell[ck] = metas
    if missing:
        raise SystemExit("missing repeat results — run `score` first:\n  " + "\n  ".join(missing))

    # The aggregate is protocol-driven: never stamp the legacy 5 / 4+4 fit sizes onto a
    # differently configured sweep. All cell/repeats must agree before we publish one table.
    def _fit_protocol(m: dict) -> tuple[int | None, int | None, int | None]:
        p = m.get("detector_score_protocol", {})
        n_unsup = p.get("fit_n_unsup_requested")
        n_succ = p.get("fit_n_sup_succ_requested")
        n_fail = p.get("fit_n_sup_fail_requested")
        # Backward compatibility for result trees written before the requested counts were
        # explicit protocol fields: those sweeps were balanced, so the saved selected-ID lists
        # recover the effective sizes exactly.
        if n_unsup is None and "fit_unsup_rollout_ids" in p:
            n_unsup = len(p["fit_unsup_rollout_ids"])
        if n_succ is None and n_fail is None and "fit_sup_rollout_ids" in p:
            n_succ = n_fail = len(p["fit_sup_rollout_ids"]) // 2
        return n_unsup, n_succ, n_fail

    fit_protocols = {
        _fit_protocol(m) for metas in by_cell.values() for m in metas
    }
    if len(fit_protocols) != 1:
        raise SystemExit(f"mixed fit protocols in result tree: {sorted(fit_protocols, key=str)}")
    fit_n_unsup, fit_n_sup_succ, fit_n_sup_fail = next(iter(fit_protocols))

    col_keys = [cell_key(c) for c in CELLS]
    det_names = {
        _det_name(sk) for metas in by_cell.values() for m in metas
        for sk in m.get("detectors", {})
    }
    ordered_dets = [d for d in DETECTORS if d in det_names] + sorted(det_names - set(DETECTORS))

    def _metric(extract) -> dict[str, dict[str, dict]]:
        return {
            d: {
                ck: _repeat_stat(extract(m.get("detectors", {}).get(f"{d}_scores", {}))
                                 for m in by_cell[ck])
                for ck in col_keys
            }
            for d in ordered_dets
        }

    def _fpr_span(realized: dict, ck: str) -> str:
        """Median [min, max] realized FPR ACROSS detectors for one cell.

        Under held-out calibration each detector gets its own threshold sample, so the achieved
        specificity is a per-detector measurement — quoting a single detector's number (as the
        in-sample table could, where every value equalled the target by construction) would hide
        exactly the spread a reader needs to judge whether the operating points are matched.
        """
        vals = [realized.get(d, {}).get(ck, {}).get("mean") for d in ordered_dets]
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        import statistics
        return (f"median {statistics.median(vals):.3f} "
                f"[{min(vals):.3f}, {max(vals):.3f}] over {len(vals)} detectors")

    tpr = _metric(lambda r: r.get("cusum_online", {}).get(fpr, {}).get("tpr"))
    lead = _metric(lambda r: r.get("cusum_online", {}).get(fpr, {}).get("median_lead"))
    realized_fpr = _metric(lambda r: r.get("cusum_online", {}).get(fpr, {}).get("succ_fire_rate"))
    # Two independent variance sources; keep them separate rather than pooled. The table's ±
    # is the REPEAT std (fit-split + resample-subset draw). This one is the CALIBRATION-DRAW std
    # within a repeat, already averaged over `cusum_calib_draws` — recorded so a reader can see
    # how much of the operating point rests on which 10 successes happened to be held out.
    tpr_calib_std = _metric(lambda r: r.get("cusum_online", {}).get(fpr, {}).get("tpr_std"))
    auroc = _metric(lambda r: r.get("whole_ep_auroc"))
    n_info = {
        ck: {
            "n_episodes": _repeat_stat(m.get("n_episodes") for m in metas),
            "n_success": _repeat_stat(m.get("n_success") for m in metas),
            "n_fail": _repeat_stat(m.get("n_fail") for m in metas),
            "splits": metas[0].get("splits"),
        }
        for ck, metas in by_cell.items()
    }

    def _pm(stat: dict | None, precision: int = 2) -> str:
        if not stat or stat.get("mean") is None:
            return "—"
        return f"{stat['mean']:.{precision}f}±{stat['std']:.{precision}f}"

    def _table_cell(detector: str, ck: str) -> str:
        ts, ls = tpr[detector][ck], lead[detector][ck]
        if ts["mean"] is None:
            return "—"
        return f"{_pm(ts, 2)} / {_pm(ls, 1)}"

    resample_description = (
        f"**k={resample_k}**, consuming all saved candidates (no model re-resampling)."
        if int(resample_k) == 32 else
        f"**k={resample_k}**, drawn as a repeat-seeded subset of the saved k=32 candidates "
        "(no model re-resampling)."
    )
    resample_source = (
        "all saved k=32 chunk_divergence candidates"
        if int(resample_k) == 32 else
        "repeat-seeded subsets of saved k=32 chunk_divergence candidates"
    )

    lines = [
        f"# Failure detection — k={resample_k}, M={calib_n}, {repeats} repeats "
        f"(online CUSUM @ target FPR={fpr})",
        "",
        f"Each cell reports `TPR mean±std / median-lead mean±std (chunks)` across **{repeats}** "
        f"repeats; std is the sample standard deviation (`ddof=1`). Resample methods use "
        + resample_description,
        "",
        "Each repeat draws different held-out fit rollouts: the unsupervised OOD family fits on "
        f"**{fit_n_unsup} success** rollouts and SAFE on a balanced "
        f"**{fit_n_sup_succ} success + {fit_n_sup_fail} failure** split. Their union "
        "is excluded from every method's scored set, preserving within-repeat method comparability. "
        "`oracle_resample_spread` is an upper bound, not a baseline.",
        "",
        f"The CUSUM alarm height is calibrated on a further **{calib_n} held-out "
        "success episodes** per cell (Sentinel's `M≈10–50` split-conformal deployment protocol), "
        "drawn per repeat and shared by every detector, then excluded from the CUSUM evaluation "
        "population. The realized FPR below is therefore a **measured out-of-sample specificity**, "
        "not a target pinned by construction — expect it to scatter around "
        f"{fpr} rather than equal it.",
        "",
        "**Scored cell sizes (mean±std across repeats) and realized success-FPR:**",
    ]
    for ck in col_keys:
        ni = n_info[ck]
        lines.append(
            f"- `{ck}`: {_pm(ni['n_episodes'], 1)} eps "
            f"({_pm(ni['n_success'], 1)} succ / {_pm(ni['n_fail'], 1)} fail), "
            f"realized FPR across detectors: {_fpr_span(realized_fpr, ck)}")
    lines += ["", "| detector (TPR mean±std / lead-ch mean±std) | " + " | ".join(col_keys) + " |",
              "|" + "---|" * (len(col_keys) + 1)]
    for d in ordered_dets:
        mark = " **(ours)**" if d == "accel" else (" [oracle]" if d.startswith("oracle_") else "")
        lines.append(f"| {d}{mark} | " + " | ".join(_table_cell(d, ck) for ck in col_keys) + " |")
    md = "\n".join(lines) + "\n"
    suffix = "" if int(calib_n) == int(DEFAULT_CUSUM_CALIB_N) else f"_M{int(calib_n)}"
    (OUT_DIR / f"fd_table_cusum{suffix}.md").write_text(md)

    protocols = {
        ck: [
            {"repeat": repeat, **m.get("detector_score_protocol", {})}
            for repeat, m in enumerate(metas)
        ]
        for ck, metas in by_cell.items()
    }
    payload = {
        "headline_metric": f"cusum_online@FPR={fpr}",
        "repeat_protocol": {
            "n_repeats": int(repeats), "std_ddof": 1, "resample_k": int(resample_k),
            "resample_source": resample_source,
            "fit_protocol": {"unsup_n_success": fit_n_unsup,
                             "safe_balanced": [fit_n_sup_succ, fit_n_sup_fail]},
            "cusum_calibration": {
                "n_success_held_out": int(calib_n),
                "draws_averaged": int(DEFAULT_CUSUM_CALIB_DRAWS),
                "estimator": "split-conformal ceil((M+1)(1-fpr)) order statistic",
                "shared_across_detectors": True,
                "fpr_is": "measured out-of-sample",
            },
        },
        "cells": {
            cell_key(c): {"run": c["run"], "model": c["model"], "bench": c["bench"],
                          **n_info[cell_key(c)], "repeat_protocols": protocols[cell_key(c)]}
            for c in CELLS
        },
        "cusum_tpr": tpr,
        "cusum_median_lead": lead,
        "cusum_realized_fpr": realized_fpr,
        "cusum_tpr_calibration_draw_std": tpr_calib_std,
        "appendix_whole_ep_auroc": auroc,
    }
    (OUT_DIR / f"fd_table_cusum{suffix}.json").write_text(json.dumps(payload, indent=2, default=str))

    plot_written = False
    try:
        _plot_cusum(ordered_dets, col_keys, tpr, fpr, repeats=repeats, resample_k=resample_k,
                    calib_n=calib_n, suffix=suffix)
        plot_written = True
    except ImportError as exc:
        print(f"[aggregate] plot skipped (optional matplotlib unavailable: {exc})", flush=True)
    products = (f"fd_table_cusum{suffix}.{{md,json}}"
                + (f" + fd_cusum{suffix}.{{png,pdf}}" if plot_written else ""))
    print(f"[aggregate] wrote {OUT_DIR}/{products} over {len(col_keys)} cells × {repeats} repeats")
    print(md)


def _plot_cusum(dets: list[str], cols: list[str], tpr: dict, fpr: str, *,
                repeats: int, resample_k: int, calib_n: int = DEFAULT_CUSUM_CALIB_N,
                suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ACCENT = "#FC5185"
    dets_plot = [d for d in dets if not d.startswith("oracle_")] + [d for d in dets if d.startswith("oracle_")]
    x = np.arange(len(dets_plot))
    n = len(cols)
    w = 0.8 / max(1, n)
    fig, ax = plt.subplots(figsize=(max(11, 1.15 * len(dets_plot)), 5.4))
    for j, ck in enumerate(cols):
        vals = [tpr[d].get(ck, {}).get("mean") for d in dets_plot]
        errs = [tpr[d].get(ck, {}).get("std") for d in dets_plot]
        vals = [np.nan if v is None else v for v in vals]
        errs = [0.0 if v is None else v for v in errs]
        bars = ax.bar(x + (j - (n - 1) / 2) * w, vals, w, yerr=errs,
                      error_kw={"elinewidth": 0.55, "capsize": 1.2},
                      label=ck, edgecolor="white", linewidth=0.4)
        for bi, d in enumerate(dets_plot):
            if d == "accel":
                bars[bi].set_edgecolor(ACCENT); bars[bi].set_linewidth(1.8)
    ax.axhline(float(fpr), color="gray", lw=0.8, ls="--", zorder=0)  # a detector at chance fires ~FPR
    ax.set_xticks(x)
    ax.set_xticklabels([("accel\n(ours)" if d == "accel" else ("oracle" if d.startswith("oracle_") else d))
                        for d in dets_plot], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"CUSUM TPR @ success-FPR≈{fpr}")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Online failure detection across FM-VLAs × benchmarks — resample k={resample_k}\n"
                 f"CUSUM TPR @ FPR={fpr} (M={calib_n} held-out conformal), "
                 f"mean ± sample std over {repeats} repeats")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9, title="model:benchmark")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fd_cusum{suffix}.{ext}", dpi=150)
    plt.close(fig)


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help=("independent result root (default: outputs/failure_detection); use a new "
              "directory for a new protocol so existing tables are never overwritten"))
    sub = ap.add_subparsers(dest="mode", required=True)

    ap.add_argument("--register-cell", action="append", default=[], metavar="SPEC",
                    help="add a cell not shipped in CELLS: model:bench:run[:first_actions"
                         "[:label_src]] (repeatable). label_src is eval_info (default) or sidecar")

    ps = sub.add_parser("score", help="run the detector battery over one/all cells")
    g = ps.add_mutually_exclusive_group(required=True)
    g.add_argument("--cell", help="model:bench, e.g. pi05:libero_all")
    g.add_argument("--all", action="store_true", help="score every cell whose run is on disk")
    ps.add_argument("--device", default="cuda", help="torch device for the learned-detector fits")
    ps.add_argument("--detectors", default=",".join(DETECTORS),
                    help="comma list (default: the full battery of 14)")
    ps.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                    help=f"number of independent fit/resample repeats (default {DEFAULT_REPEATS})")
    ps.add_argument("--repeat-start", type=int, default=0,
                    help="first repeat index; seeds and output directory use this absolute index (default 0)")
    ps.add_argument("--skip-existing", action="store_true",
                    help="skip repeat/cell outputs whose meta already contains every requested detector")
    ps.add_argument("--resample-k", type=int, default=DEFAULT_RESAMPLE_K,
                    help=f"saved MC candidates used by resample methods (default {DEFAULT_RESAMPLE_K})")
    # Seeds and fit sizes as reported in the paper: fit/learned-model seeds 3200..3204 and
    # candidate-subset seeds 6400..6404 over the 5 repeats; the OOD family is fit on 32 successful
    # rollouts and SAFE on a balanced 16 + 16, with the union of fit sets excluded from scoring.
    ps.add_argument("--fit-seed", type=int, default=3200,
                    help="base fit split/model RNG seed; repeat r uses fit-seed+r (default 3200)")
    ps.add_argument("--resample-seed", type=int, default=6400,
                    help="base saved-candidate subset seed; repeat r uses resample-seed+r (default 6400)")
    ps.add_argument("--fit-n-unsup", type=int, default=32,
                    help="#success rollouts for the OOD family fit (default 32)")
    ps.add_argument("--fit-n-sup-succ", type=int, default=16,
                    help="#success rollouts for the balanced SAFE fit (default 16)")
    ps.add_argument("--fit-n-sup-fail", type=int, default=16,
                    help="#failure rollouts for the balanced SAFE fit (default 16)")

    pr = sub.add_parser("recalibrate",
                        help="re-run ONLY the CUSUM layer of a scored tree at a different held-out "
                             "calibration size M (cheap: reuses the cached per-episode streams)")
    pr.add_argument("--calib-n", type=int, required=True,
                    help="held-out SUCCESS episodes for the new CUSUM threshold (e.g. 50)")
    pr.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                    help=f"repeat directories to re-derive (default {DEFAULT_REPEATS})")
    pr.add_argument("--resample-k", type=int, default=DEFAULT_RESAMPLE_K,
                    help=f"source resample-k tree (default {DEFAULT_RESAMPLE_K})")
    pr.add_argument("--src-calib-n", type=int, default=DEFAULT_CUSUM_CALIB_N,
                    help=f"source tree's M (default {DEFAULT_CUSUM_CALIB_N})")

    pa = sub.add_parser("aggregate", help="assemble the cross-cell CUSUM table + figure from scored cells")
    pa.add_argument("--fpr", default="0.1", help="target success-FPR operating point (default 0.1)")
    pa.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                    help=f"number of repeat directories to aggregate (default {DEFAULT_REPEATS})")
    pa.add_argument("--resample-k", type=int, default=DEFAULT_RESAMPLE_K,
                    help=f"resample-k result tree to aggregate (default {DEFAULT_RESAMPLE_K})")
    pa.add_argument("--calib-n", type=int, default=DEFAULT_CUSUM_CALIB_N,
                    help=f"CUSUM held-out calibration size whose tree to aggregate (default "
                         f"{DEFAULT_CUSUM_CALIB_N}); a non-default M reads repeats/k<k>_M<n>/ and "
                         f"writes fd_table_cusum_M<n>.*")

    args = ap.parse_args()
    for spec in args.register_cell:
        register_cell(spec)
    OUT_DIR = args.out_dir.expanduser().resolve()
    if args.mode == "score":
        if args.repeats < 1:
            raise SystemExit(f"--repeats must be >=1, got {args.repeats}")
        if args.repeat_start < 0:
            raise SystemExit(f"--repeat-start must be >=0, got {args.repeat_start}")
        if args.resample_k < 1:
            raise SystemExit(f"--resample-k must be >=1, got {args.resample_k}")
        detectors = args.detectors.replace(",", " ").split()
        cells = CELLS if args.all else [resolve_cell(args.cell)]
        failures = []
        for repeat in range(args.repeat_start, args.repeat_start + args.repeats):
            for c in cells:
                if args.skip_existing and _repeat_complete(
                        c, resample_k=args.resample_k, repeat=repeat, detectors=detectors):
                    print(f"[score] SKIP complete {cell_key(c)}/repeat_{repeat:02d}", flush=True)
                    continue
                try:
                    score_cell(
                        c, device=args.device, detectors=detectors, repeat=repeat,
                        resample_k=args.resample_k,
                        resample_seed=args.resample_seed + repeat,
                        fit_seed=args.fit_seed + repeat,
                        fit_n_unsup=args.fit_n_unsup, fit_n_sup_succ=args.fit_n_sup_succ,
                        fit_n_sup_fail=args.fit_n_sup_fail)
                except (SystemExit, Exception) as exc:
                    failures.append(f"{cell_key(c)}/repeat_{repeat:02d}: {exc}")
                    print(f"[score] {failures[-1]} FAILED", flush=True)
        if failures:
            raise SystemExit("one or more repeat cells failed:\n  " + "\n  ".join(failures))
    elif args.mode == "recalibrate":
        if args.calib_n < 1:
            raise SystemExit(f"--calib-n must be >=1, got {args.calib_n}")
        recalibrate(calib_n=args.calib_n, repeats=args.repeats,
                    resample_k=args.resample_k, src_calib_n=args.src_calib_n)
    else:
        aggregate(fpr=args.fpr, repeats=args.repeats, resample_k=args.resample_k,
                  calib_n=args.calib_n)


if __name__ == "__main__":
    main()
