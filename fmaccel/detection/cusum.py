"""Turning per-decision scores into calibrated alarms — the harness every detector is scored through.

A per-decision score is not a detector. Two things convert one into an online alarm, and both are
implemented here (paper §4; ``docs/method.md`` §4):

  * **one-sided CUSUM** — ``S_t = max(0, S_{t-1} + (z_t - mu0 - k))``, ``S_0 = 0``, slack
    ``k = c*sigma`` with ``c = 0.25``; alarm at the first ``S_t > eta``. It accumulates *sustained*
    drift instead of reacting to single spikes, and is strictly causal: the decision at ``t`` uses
    only chunks ``<= t``. That causality is what makes the reported detection lead an online claim
    rather than hindsight. (:func:`cusum_alarm`; :func:`cusum_peak` is a whole-episode statistic
    used ONLY for calibration, never reported as a score.)
  * **split conformal on the episode CUSUM peak** — ``eta`` is the ``r``-th smallest peak over ``M``
    held-out SUCCESSFUL episodes, ``r = ceil((M+1)(1-alpha))``, which bounds the false-alarm rate at
    ``<= alpha`` on exchangeable future successes. With ``M = 50``, ``alpha = 0.1`` that is the 46th
    smallest. (:func:`conformal_height`.)

Calibrating a *scalar* per episode, rather than a time-indexed conformal band, is deliberate: a band
needs calibration data at every timestep, but successes are short while failures run to the limit,
so the band's support ends early and becomes pure padding exactly where the failures live. The
episode-level peak has no horizon problem.

The calibration episodes are held out from the scored population, so the realized false-alarm rate
is a MEASURED out-of-sample number, not the target it was pinned to in-sample. Read
``calibration`` in the returned dict before quoting an FPR.

Alongside the online CUSUM the module reports diagnostic views used in the analysis:

  [A] whole-episode AUROC + Cohen's d   — does the episode-level score separate outcome at all?
  [B] length confound                   — failures time out (longer), so a whole-episode mean is
                                          partly hindsight; [D] and [E] are the honest claims.
  [C] within-(task, split) AUROC        — difficulty-controlled (pair-weighted across cells).
  [D] early-window AUROC @ k, survival-conditioned — does the score over the FIRST k chunks predict
                                          eventual failure, among episodes surviving >= k?
  [E] causal online CUSUM               — the streaming detector that actually fires: failure TPR,
                                          median lead in re-planning steps, alarm/onset position.
  TPR @ fixed FPR                       — the reported operating point.

Pure numpy (+ optional matplotlib for the comparison plot); no torch, no policy, so it imports and
runs without lerobot installed and needs no GPU. The convention everywhere is **lower score = more
confident**, so a *higher* score should mark failure — see
:class:`fmaccel.detectors.base.Detector`.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Collection, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- metrics
def auroc(fail_vals: np.ndarray, succ_vals: np.ndarray) -> tuple[float, int]:
    """P(random failure score > random success score). 0.5 = no signal; >0.5 = higher → fail.

    Mann-Whitney U with tie-averaged ranks. Returns ``(auroc, n_pairs)``; ``(nan, 0)`` if
    either group is empty. Verbatim from the published accel analysis.
    """
    fail_vals, succ_vals = np.asarray(fail_vals, float), np.asarray(succ_vals, float)
    if len(fail_vals) == 0 or len(succ_vals) == 0:
        return float("nan"), 0
    allv = np.concatenate([fail_vals, succ_vals])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    avg = sums / cnt
    ranks = avg[inv]
    Rf = ranks[: len(fail_vals)].sum()
    U = Rf - len(fail_vals) * (len(fail_vals) + 1) / 2
    return U / (len(fail_vals) * len(succ_vals)), len(fail_vals) * len(succ_vals)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference ``(mean(a) − mean(b)) / pooled_sd``. nan if <2 in either."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def tpr_at_fpr(fail_vals: np.ndarray, succ_vals: np.ndarray, fpr: float) -> float:
    """TPR at a threshold giving the target FPR on successes (higher score ⇒ alarm).

    τ = the ``(1−fpr)`` quantile of success scores (so a fraction ``fpr`` of successes
    exceed it); TPR = fraction of failures exceeding τ. nan if either group is empty.
    """
    fail_vals, succ_vals = np.asarray(fail_vals, float), np.asarray(succ_vals, float)
    if len(fail_vals) == 0 or len(succ_vals) == 0:
        return float("nan")
    tau = float(np.quantile(succ_vals, 1.0 - float(fpr)))
    return float(np.mean(fail_vals > tau))


# ------------------------------------------------------------------ CUSUM (online)
def cusum_peak(stream: np.ndarray, mu0: float, k: float) -> float:
    """Peak of the one-sided upper CUSUM ``S_t = max(0, S_{t-1} + (x_t − μ0 − k))``.

    Used (on SUCCESS episodes) to calibrate the alarm height ``h`` to a target false-
    alarm rate. The peak is a whole-episode (hindsight) statistic — it is used only for
    threshold calibration, never reported as a detection score.
    """
    a = np.asarray(stream, float)
    S = m = 0.0
    for x in a:
        S = max(0.0, S + (x - mu0 - k))
        if S > m:
            m = S
    return m


def cusum_alarm(stream: np.ndarray, h: float, mu0: float, k: float) -> tuple[int | None, int | None]:
    """Causal one-sided CUSUM. Returns ``(alarm_idx, onset_idx)`` or ``(None, None)``.

    ``S_t = max(0, S_{t-1} + (x_t − μ0 − k))`` accumulates upward shifts above the
    reference ``μ0 + k``; the FIRST ``t`` with ``S_t > h`` is the alarm, and the last
    chunk where ``S`` reset to 0 before it is the maximum-likelihood change-point (onset).
    Strictly causal — the decision at ``t`` uses only chunks ``<= t`` — so this is the honest
    online claim, unlike a whole-episode AUROC, which is hindsight.
    """
    a = np.asarray(stream, float)
    S, last_zero = 0.0, 0
    for t, x in enumerate(a):
        S = max(0.0, S + (x - mu0 - k))
        if S == 0.0:
            last_zero = t
        if S > h:
            return t, last_zero
    return None, None


def conformal_height(peaks: np.ndarray, fpr: float) -> float:
    """Split-conformal alarm height from ``M`` SUCCESS-episode CUSUM peaks (Sentinel's protocol).

    The finite-sample-valid threshold is the ``⌈(M+1)(1−δ)⌉``-th smallest peak, which guarantees
    false-alarm rate ``≤ δ`` on exchangeable future successes. It differs from ``np.quantile`` at
    the small ``M`` a held-out calibration set implies (``M=10, δ=0.1 → ⌈9.9⌉ = 10`` = the max,
    vs ``np.quantile``'s interpolated ~9th) — and at small ``M`` that gap is the difference between
    a real guarantee and an optimistic one. If ``⌈(M+1)(1−δ)⌉ > M`` the sample cannot certify the
    level; we return ``+inf`` only when that is *also* unattainable, else the max.
    """
    p = np.sort(np.asarray(peaks, float))
    M = p.size
    if M == 0:
        return float("nan")
    rank = int(np.ceil((M + 1) * (1.0 - float(fpr))))   # 1-indexed order statistic
    return float(p[min(rank, M) - 1])


def cusum_online(
    succ_streams: Sequence[np.ndarray],
    fail_streams: Sequence[np.ndarray],
    *,
    fpr: float = 0.1,
    k_sigma: float = 0.25,
    calib_streams: Sequence[np.ndarray] | None = None,
    conformal: bool | None = None,
) -> dict:
    """Calibrate a causal CUSUM on SUCCESS episodes and evaluate it on failures.

    Detector-agnostic, so it works at any score scale (accel ~2, straightness ~0.15):
    the reference ``μ0`` = pooled mean of calibration chunk-scores, slack ``k = k_sigma·σ``
    (``σ`` = pooled calibration std), and the alarm height ``h`` = the ``(1−fpr)`` upper point of
    the calibration episodes' CUSUM peaks. Failures are then scored online; we report the fraction
    that alarm (TPR), the median **lead** (chunks between the alarm and the episode's last chunk =
    how early we'd abort), and the alarm/onset position as a fraction of episode length.

    ``k_sigma`` default is **0.25** (was 0.5): a smaller slack makes the cumulative sum react to
    milder *sustained* drift, which — since ``h`` is re-calibrated per k, keeping the success-FPR
    matched — is a strict TPR gain in the FD main-exp sweep (mean accel TPR 0.527→0.551 across the 6
    cells, every cell ≥ its k=0.5 value; k=0.1 gains a bit more but starts riding long-episode random
    walk on a few embedding detectors). See the k_sigma sweep tables ``fd_table_cusum_k*.md``.

    **Calibration set.** ``calib_streams`` = a HELD-OUT set of success episodes (the deployment-
    realistic protocol, and Sentinel's: ``M≈10–50`` successes, split-conformal). It is disjoint from
    ``succ_streams``, so ``succ_fire_rate`` becomes a genuine OUT-OF-SAMPLE specificity measurement
    rather than a number that equals ``fpr`` by construction. Failures never enter calibration under
    either path, so TPR was already leak-free.

    ``calib_streams=None`` keeps the legacy in-sample path (calibrate on every scored success);
    there ``succ_fire_rate ≈ fpr`` is a *target*, not a measurement — read ``calibration`` in the
    returned dict before quoting it. ``conformal`` defaults to True with a held-out set (small ``M``
    needs the order statistic) and False in-sample (large ``M``; keeps historical tables comparable).
    """
    succ = [np.asarray(s, float) for s in succ_streams if np.asarray(s, float).size]
    fail = [np.asarray(s, float) for s in fail_streams if np.asarray(s, float).size]
    held_out = calib_streams is not None
    if conformal is None:
        conformal = held_out
    calib = ([np.asarray(s, float) for s in calib_streams if np.asarray(s, float).size]
             if held_out else succ)
    if not calib:
        succ, fail = [], []   # nothing to calibrate on -> emit the empty result below
    if not succ or not fail:
        return {"fpr": float(fpr), "tpr": float("nan"), "n_fail": len(fail), "n_succ": len(succ),
                "median_lead": float("nan"), "median_alarm_frac": float("nan"),
                "median_onset_frac": float("nan"), "mu0": float("nan"), "k": float("nan"),
                "h": float("nan"), "succ_fire_rate": float("nan"),
                "n_calib": len(calib), "calibration": "held-out" if held_out else "in-sample",
                "conformal": bool(conformal)}
    # μ0 / σ / h all come from the CALIBRATION successes only (== the scored successes on the
    # legacy in-sample path, a disjoint held-out set otherwise).
    pooled = np.concatenate(calib)
    mu0 = float(pooled.mean())
    sigma = float(pooled.std())
    k = float(k_sigma * sigma)
    calib_peaks = np.array([cusum_peak(s, mu0, k) for s in calib])
    h = (conformal_height(calib_peaks, fpr) if conformal
         else float(np.quantile(calib_peaks, 1.0 - float(fpr))))

    # Measured on the SCORED successes — out-of-sample whenever `calib` is held out.
    s_fire = sum(cusum_alarm(s, h, mu0, k)[0] is not None for s in succ)
    leads, alarm_frac, onset_frac, n_fire = [], [], [], 0
    for f in fail:
        al, on = cusum_alarm(f, h, mu0, k)
        if al is None:
            continue
        n_fire += 1
        n = len(f)
        leads.append(n - 1 - al)
        alarm_frac.append(al / n)
        onset_frac.append((on / n) if on is not None else float("nan"))
    return {
        "fpr": float(fpr),
        "tpr": n_fire / len(fail),
        "n_fail": len(fail),
        "n_succ": len(succ),
        "n_fail_fire": int(n_fire),
        "succ_fire_rate": s_fire / len(succ),
        "median_lead": float(np.median(leads)) if leads else float("nan"),
        "median_alarm_frac": float(np.median(alarm_frac)) if alarm_frac else float("nan"),
        "median_onset_frac": float(np.nanmedian(onset_frac)) if onset_frac else float("nan"),
        "mu0": mu0, "k": k, "h": h,
        "n_calib": len(calib),
        # 'held-out'  -> succ_fire_rate is a MEASURED out-of-sample specificity
        # 'in-sample' -> succ_fire_rate == fpr by construction; quote it as the target only
        "calibration": "held-out" if held_out else "in-sample",
        "conformal": bool(conformal),
    }


def _agg(stream: np.ndarray, how: str) -> float:
    a = np.asarray(stream, float)
    if a.size == 0:
        return float("nan")
    if how == "mean":
        return float(a.mean())
    if how == "max":
        return float(a.max())
    if how == "last":
        return float(a[-1])
    raise ValueError(f"unknown aggregation {how!r}; one of mean/max/last")


def _first_k_mean(stream: np.ndarray, k: int) -> float:
    a = np.asarray(stream, float)
    return float(a[:k].mean()) if a.size >= 1 else float("nan")


# ----------------------------------------------------------------------------- loader
def load_episodes(
    data_roots: Sequence[str | Path],
    *,
    splits: Sequence[str],
    score_keys: Sequence[str],
) -> list[dict]:
    """Load closed-loop episodes carrying per-chunk score stream(s).

    Walks ``<root>/<split>/*.json`` for every root × split, descends into
    ``tasks.<name>.episodes``, and keeps an episode iff it has ``success``/``n_chunks``
    and at least one of ``score_keys``. Each returned dict has ``task, split, success,
    n_chunks``, ``uid`` and ``streams = {key: np.ndarray}`` for whichever keys were present.

    ``uid`` is a stable ``split/task/rollout_id`` identity (falling back to the within-task
    position when a legacy json has no ``rollout_id``). It exists so the CUSUM calibration split
    can name the SAME held-out episodes for every detector, keeping the cross-detector comparison
    matched — a per-detector draw would give each one its own threshold sample.
    """
    eps: list[dict] = []
    n_files = 0
    for root in data_roots:
        for split in splits:
            for jf in sorted(glob.glob(os.path.join(str(root), split, "*.json"))):
                try:
                    d = json.load(open(jf))
                except (json.JSONDecodeError, OSError):
                    logger.warning("skipping unreadable json %s", jf)
                    continue
                n_files += 1
                for tname, t in d.get("tasks", {}).items():
                    for pos, e in enumerate(t.get("episodes", [])):
                        if "success" not in e or "n_chunks" not in e:
                            continue
                        streams = {}
                        for key in score_keys:
                            v = e.get(key)
                            if v:
                                streams[key] = np.asarray(v, float)
                        if not streams:
                            continue
                        rid = e.get("rollout_id", f"pos{pos}")   # legacy json: fall back to position
                        eps.append(
                            {
                                "task": tname,
                                "split": split,
                                "uid": f"{split}/{tname}/{rid}",
                                "success": bool(e["success"]),
                                "n_chunks": int(e["n_chunks"]),
                                "streams": streams,
                            }
                        )
    logger.info("loaded %d episodes from %d json files (roots=%s, splits=%s)",
                len(eps), n_files, [str(r) for r in data_roots], list(splits))
    return eps


# --------------------------------------------------------------------------- analysis
def analyze_key(
    eps: list[dict],
    score_key: str,
    *,
    agg: str = "mean",
    early_ks: Sequence[int] = (1, 2, 3, 5, 8, 10, 15),
    fpr_targets: Sequence[float] = (0.1,),
    min_survivors: int = 20,
    cusum_calib_uid_sets: Sequence[Collection[str]] | None = None,
) -> dict:
    """All metrics for ONE score key over the episodes that carry it.

    ``cusum_calib_uid_sets`` is a list of draws, each naming SUCCESS episodes reserved for CUSUM
    threshold calibration. Per draw, those episodes are excluded from the CUSUM evaluation
    population (so ``succ_fire_rate`` is a measured out-of-sample specificity) and the results are
    averaged across draws; ``*_std`` fields carry the across-draw scatter. The hindsight
    AUROC/Cohen-d/early-window blocks below involve no calibration and keep the full population, so
    those numbers stay comparable to earlier tables.
    """
    sub = [e for e in eps if score_key in e["streams"]]
    succ = [e for e in sub if e["success"]]
    fail = [e for e in sub if not e["success"]]

    # whole-episode aggregate per episode
    def whole(e: dict) -> float:
        return _agg(e["streams"][score_key], agg)

    ms = np.array([whole(e) for e in succ]) if succ else np.zeros(0)
    mf = np.array([whole(e) for e in fail]) if fail else np.zeros(0)
    a_auroc, _ = auroc(mf, ms)
    a_d = cohen_d(mf, ms)

    # [B] length confound
    ns = np.array([e["n_chunks"] for e in succ]) if succ else np.zeros(0)
    nf = np.array([e["n_chunks"] for e in fail]) if fail else np.zeros(0)

    # [C] within-(task,split) pair-weighted AUROC
    by_cell: dict[tuple, list[dict]] = {}
    for e in sub:
        by_cell.setdefault((e["task"], e["split"]), []).append(e)
    cell_aucs, cell_w = [], []
    for group in by_cell.values():
        gs = [whole(e) for e in group if e["success"]]
        gf = [whole(e) for e in group if not e["success"]]
        au, w = auroc(gf, gs)
        if not np.isnan(au) and w > 0:
            cell_aucs.append(au)
            cell_w.append(w)
    cell_aucs_a, cell_w_a = np.array(cell_aucs), np.array(cell_w)
    wauc = float((cell_aucs_a * cell_w_a).sum() / cell_w_a.sum()) if cell_w_a.sum() else float("nan")
    mean_cell_auc = float(cell_aucs_a.mean()) if cell_aucs_a.size else float("nan")
    n_pointing = int((cell_aucs_a > 0.5).sum())

    # [D] early-window AUROC, conditioned on survival to >= k chunks
    early = []
    for k in early_ks:
        surv = [e for e in sub if e["n_chunks"] >= k]
        if len(surv) < min_survivors:
            continue
        s = [_first_k_mean(e["streams"][score_key], k) for e in surv if e["success"]]
        f = [_first_k_mean(e["streams"][score_key], k) for e in surv if not e["success"]]
        au, _ = auroc(f, s)
        early.append(
            {
                "k": int(k),
                "n_surv": len(surv),
                "sr_surv": (len(s) / len(surv)) if surv else float("nan"),
                "succ_mean": float(np.mean(s)) if s else float("nan"),
                "fail_mean": float(np.mean(f)) if f else float("nan"),
                "auroc": au,
                "cohen_d": cohen_d(np.asarray(f), np.asarray(s)),
            }
        )

    tpr = {f"{fp:g}": tpr_at_fpr(mf, ms, fp) for fp in fpr_targets}

    # [E] causal online CUSUM — the honest streaming detector: calibrate the alarm height on a
    # HELD-OUT set of SUCCESS episodes and report failure TPR + lead time (chunks before episode
    # end) plus the realized out-of-sample success fire-rate. Detector-agnostic scale (μ0/k from
    # the calibration population). The held-out episodes are the same for every detector (chosen
    # once per cell by uid), so the comparison stays matched.
    draws = [set(u) for u in (cusum_calib_uid_sets or [])] or [set()]
    fail_streams = [e["streams"][score_key] for e in fail]

    def _cusum_over_draws(fp: float) -> dict:
        per_draw = []
        for uids in draws:
            calib_eps = [e for e in succ if e["uid"] in uids]
            eval_succ = [e for e in succ if e["uid"] not in uids] if uids else succ
            if uids and len(calib_eps) < len(uids):
                logger.warning("%s: only %d/%d CUSUM calibration episodes carry this score "
                               "(its threshold rests on fewer successes than its peers)",
                               score_key, len(calib_eps), len(uids))
            per_draw.append(cusum_online(
                [e["streams"][score_key] for e in eval_succ], fail_streams, fpr=fp,
                calib_streams=([e["streams"][score_key] for e in calib_eps] if uids else None)))
        if len(per_draw) == 1:
            return {**per_draw[0], "n_calib_draws": 1}
        # Average the CUSUM layer over calibration draws; keep the scatter alongside so a noisy
        # threshold can never masquerade as a precise operating point.
        out = dict(per_draw[0])
        out["n_calib_draws"] = len(per_draw)
        for f in ("tpr", "succ_fire_rate", "median_lead", "median_alarm_frac",
                  "median_onset_frac", "h", "mu0", "k"):
            v = np.array([d[f] for d in per_draw], float)
            out[f] = float(np.nanmean(v))
            out[f + "_std"] = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
        out["n_fail_fire"] = float(np.mean([d["n_fail_fire"] for d in per_draw]))
        return out

    cusum = {f"{fp:g}": _cusum_over_draws(fp) for fp in fpr_targets}

    return {
        "score_key": score_key,
        "n_episodes": len(sub),
        "n_success": len(succ),
        "n_fail": len(fail),
        "success_rate": (len(succ) / len(sub)) if sub else float("nan"),
        "whole_ep_auroc": a_auroc,
        "whole_ep_cohen_d": a_d,
        "succ_mean": float(ms.mean()) if ms.size else float("nan"),
        "fail_mean": float(mf.mean()) if mf.size else float("nan"),
        "succ_median_chunks": float(np.median(ns)) if ns.size else float("nan"),
        "fail_median_chunks": float(np.median(nf)) if nf.size else float("nan"),
        "within_task_auroc_pairweighted": wauc,
        "within_task_auroc_mean": mean_cell_auc,
        "n_cells": int(cell_aucs_a.size),
        "n_cells_pointing": n_pointing,
        "early_window": early,
        "tpr_at_fpr": tpr,
        "cusum_online": cusum,
        "agg": agg,
    }


def _plot_compare(results: dict[str, dict], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axe, axc) = plt.subplots(1, 2, figsize=(13, 5))
    # (1) early-window AUROC vs k, one line per detector — the online claim
    for name, r in results.items():
        ew = r.get("early_window") or []
        if not ew:
            continue
        ks = [row["k"] for row in ew]
        aus = [row["auroc"] for row in ew]
        axe.plot(ks, aus, "-o", ms=4, label=name)
    axe.axhline(0.5, ls="--", color="grey", lw=1)
    axe.set_xlabel("early window k (first k chunks)")
    axe.set_ylabel("AUROC(fail > succ)")
    axe.set_ylim(0.45, 1.0)
    axe.set_title("Early-window detection (survival-conditioned)")
    axe.grid(alpha=0.3)
    axe.legend(fontsize=8)
    # (2) whole-ep vs within-task AUROC bars
    names = list(results.keys())
    x = np.arange(len(names))
    we = [results[n]["whole_ep_auroc"] for n in names]
    wt = [results[n]["within_task_auroc_pairweighted"] for n in names]
    axc.bar(x - 0.2, we, 0.4, label="whole-ep (hindsight)", color="#9ecae1")
    axc.bar(x + 0.2, wt, 0.4, label="within-task", color="#3182bd")
    axc.axhline(0.5, ls="--", color="grey", lw=1)
    axc.set_xticks(x)
    axc.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axc.set_ylabel("AUROC")
    axc.set_ylim(0.4, 1.0)
    axc.set_title("Whole-episode vs within-task AUROC")
    axc.grid(alpha=0.3, axis="y")
    axc.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _draw_cusum_calib(eps: list[dict], score_keys: Sequence[str], *, n: int, seed: int,
                      n_draws: int = 1) -> list[list[str]]:
    """``n_draws`` independent draws of ``n`` SUCCESS episode uids for CUSUM calibration.

    Drawn from successes carrying EVERY ``score_key`` so each detector calibrates on the identical
    episodes (a per-detector draw would confound the comparison with threshold-sample noise), and
    sorted-then-seeded so the choice is reproducible from ``seed`` alone.

    Several draws exist because an ``M=10`` threshold is *intrinsically* noisy — measured on
    π₀.₅/LIBERO, one draw gives TPR ±0.09 and a realized FPR anywhere in ``[0.00, 0.31]``, which is
    not a matched operating point to compare detectors at. The draw is independent of the (expensive)
    scoring, so averaging the CUSUM layer over many draws costs nothing and removes calibration-draw
    noise from the comparison while keeping every threshold honestly estimated from 10 held-out
    successes.

    Returns ``[]`` — the legacy in-sample path — when ``n <= 0`` or the pool is too small to also
    leave successes to evaluate on.
    """
    if n <= 0:
        return []
    keys = list(score_keys)
    pool = sorted(e["uid"] for e in eps
                  if e["success"] and all(k in e["streams"] for k in keys))
    if len(pool) < n + 2:   # need a calibration set AND successes left to measure FPR on
        logger.warning("CUSUM calibration disabled: only %d success episodes carry all %d score "
                       "keys, need >= %d — falling back to IN-SAMPLE calibration.",
                       len(pool), len(keys), n + 2)
        return []
    arr = np.array(pool, dtype=object)
    out = []
    for d in range(max(1, int(n_draws))):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed) & 0xFFFFFFFF, len(pool), d]))
        out.append(sorted(rng.choice(arr, size=int(n), replace=False).tolist()))
    return out


def run_failure_detection_eval(
    data_roots: Sequence[str | Path],
    *,
    score_keys: Sequence[str] = ("chunk_accels",),
    splits: Sequence[str] = ("target", "pretrain"),
    agg: str = "mean",
    early_ks: Sequence[int] = (1, 2, 3, 5, 8, 10, 15),
    fpr_targets: Sequence[float] = (0.1,),
    out_dir: str | Path | None = None,
    label: str | None = None,
    plot: bool = True,
    cusum_calib_n: int = 10,
    cusum_calib_seed: int = 0,
    cusum_calib_draws: int = 20,
) -> dict:
    """Score every ``score_key`` over the same closed-loop episodes and (optionally) write
    ``meta.json`` + ``compare_detectors.png`` to ``out_dir``. Returns the full summary dict.

    ``cusum_calib_n`` SUCCESS episodes (default **10**, Sentinel's ``M≈10–50`` deployment protocol)
    are reserved for CUSUM threshold calibration and excluded from the CUSUM evaluation population —
    so the reported success fire-rate is measured out-of-sample instead of being pinned to the
    target by construction. Set ``cusum_calib_n=0`` for the legacy in-sample behaviour. Draws are
    seeded by ``cusum_calib_seed`` and restricted to episodes carrying EVERY ``score_key``, so all
    detectors calibrate on exactly the same episodes.

    ``cusum_calib_draws`` (default **20**) independent M-episode draws are averaged. A single M=10
    threshold is intrinsically high-variance (measured: TPR ±0.09, realized FPR spanning
    ``[0.00, 0.31]``), which would leave detectors compared at *different* operating points. The
    draw is independent of scoring, so averaging costs nothing; each threshold is still estimated
    from exactly ``cusum_calib_n`` held-out successes. Set to 1 for a single draw.
    """
    data_roots = [Path(r) for r in data_roots]
    eps = load_episodes(data_roots, splits=splits, score_keys=score_keys)
    if not eps:
        raise ValueError(
            f"no episodes with any of {list(score_keys)} under "
            f"{[str(r) for r in data_roots]} (splits={list(splits)})"
        )

    calib_sets = _draw_cusum_calib(eps, score_keys, n=cusum_calib_n, seed=cusum_calib_seed,
                                   n_draws=cusum_calib_draws)
    detectors = {
        key: analyze_key(eps, key, agg=agg, early_ks=early_ks, fpr_targets=fpr_targets,
                         cusum_calib_uid_sets=calib_sets)
        for key in score_keys
    }
    detectors = {k: v for k, v in detectors.items() if v["n_episodes"] > 0}

    n_succ = sum(e["success"] for e in eps)
    summary = {
        "stage": "failure_detection_eval",
        "label": label,
        "data_roots": [str(r) for r in data_roots],
        "splits": list(splits),
        "agg": agg,
        "n_episodes": len(eps),
        "n_success": int(n_succ),
        "n_fail": int(len(eps) - n_succ),
        "score_keys": list(detectors.keys()),
        "cusum_calibration": {
            "mode": "held-out" if calib_sets else "in-sample",
            "n_requested": int(cusum_calib_n),
            "n_per_draw": len(calib_sets[0]) if calib_sets else 0,
            "n_draws": len(calib_sets),
            "seed": int(cusum_calib_seed),
            "estimator": "split-conformal ceil((M+1)(1-fpr)) order statistic" if calib_sets
                         else "np.quantile(peaks, 1-fpr)",
            "uid_sets": [list(u) for u in calib_sets],
            "note": ("threshold episodes excluded from the CUSUM eval population; succ_fire_rate "
                     "is measured out-of-sample, averaged over the draws (see *_std for scatter)"
                     if calib_sets else
                     "legacy: calibrated on every scored success; succ_fire_rate == target fpr"),
        },
        "detectors": detectors,
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        from fmaccel.core.io import write_json

        write_json(out / "meta.json", summary)
        if plot and detectors:
            try:
                _plot_compare(detectors, out / "compare_detectors.png",
                              title=label or "failure detectors")
                summary["plot"] = str(out / "compare_detectors.png")
            except Exception as exc:  # plotting must never break the metrics
                logger.warning("compare plot failed: %s", exc)
        summary["meta"] = str(out / "meta.json")

    return summary
