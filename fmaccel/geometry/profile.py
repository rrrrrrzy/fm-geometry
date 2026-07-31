"""accel_profile stage: WHERE along the denoise schedule a chunk's ``accel`` lives.

``accel = Σ_t c_t`` is already a sum over denoise steps (``chunk_geometry._accel``), so
the phase decomposition is **free**: keep the per-step contribution
``c_t = ‖Δv_{t+1}−Δv_t‖ / ⟨‖Δv‖⟩`` (``chunk_geometry.whole_chunk_curvature_profile``)
instead of summing. This stage pools every chunk of a recording, normalizes *within*
chunk (``ĉ_t = c_t / accel`` — the **shape**, not the magnitude, else high-accel chunks
just dominate by amplitude and re-measure accel), groups chunks by total accel
(low / mid / high + the top decile), and reports:

  * ``profile.png`` — mean ± 95% CI of ``ĉ`` vs **FM time**, per accel group, at both the
    chunk and the action-position granularity (mirrors ``chunk_geometry``'s two levels).
  * ``cm_vs_accel.png`` — curvature center-of-mass ``τ_cm = Σ_t c_t·time_t / Σ_t c_t`` vs
    total accel (+ Spearman ρ): does a higher-accel chunk push its bend to a specific
    denoise phase? (early-fraction = share of accel on the noise side is reported too.)
  * ``accel_profile.npz`` + ``meta.json`` — pooled ``c``/``accel``/``time_mid``/``tau_cm``/
    group ids, per-group mean profiles, ρ, and the ``max|c.sum−accel|`` invariant check.

Needs only the ``fm/`` recording — no ``chunk_divergence``, no model load, no GPU — so
it runs identically on any recorded dataset. The ``time`` axis is the π₀.₅ FM
schedule (descends 1.0→0.1; high = near noise, low = near the clean action), shared by
every recording, so several runs overlay on one axis (see ``cli/profile.py
--compare``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

import numpy as np

from fmaccel.core import runs
from fmaccel.recording.loader import FMRecording
from fmaccel.geometry.accel import (
    EPS, _spearman, action_level_accel, action_level_accel_profile,
    whole_chunk_curvature, whole_chunk_curvature_profile,
)

logger = logging.getLogger(__name__)

NPZ_NAME = "accel_profile.npz"
META_NAME = "meta.json"
# accel-quantile group cut points (low / mid / high) + the highlighted top decile.
GROUP_QS = (33.0, 67.0)
TOP_PCT = 90.0


def _valid_n_chunks_env(ro: Any, env_idx: int) -> int:
    """Chunks of ``env_idx``'s FIRST episode (up to & incl. the first env-step done).

    Mirrors ``chunk_geometry._valid_n_chunks`` but per batch slot, and treats a
    feedback-free recording (teacher-forced / server, ``n_env_steps==0``) as one
    episode of every chunk. Cutting at the first reset keeps the per-episode z-score
    (the accel normalization) over a single coherent observation stream."""
    if ro.n_env_steps == 0:
        return int(ro.n_chunks)
    done = np.asarray(ro.terminated[:, env_idx] | ro.truncated[:, env_idx], dtype=bool)
    hits = np.flatnonzero(done)
    if not len(hits):
        return int(ro.n_chunks)
    d_step = int(hits[0])
    starts = np.asarray(ro.env_step_at_chunk_start)
    return int(max(1, min(int(np.searchsorted(starts, d_step, side="right")), ro.n_chunks)))


def _time_mid(time_row: np.ndarray) -> np.ndarray:
    """FM-time coordinate for each of the ``T-1`` jerk terms: the midpoint of the two
    adjacent denoise timesteps the jerk compares. ``time_row`` is ``(T,)``, descending
    1.0→~0.1; returns ``(T-1,)`` (descending), high = near noise, low = near action."""
    t = np.asarray(time_row, np.float64)
    return (0.5 * (t[:-1] + t[1:])).astype(np.float32)


def _group_ids(accel: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Tercile group id per chunk (0=low,1=mid,2=high) + a top-decile boolean mask."""
    qs = np.percentile(accel, GROUP_QS)
    gid = np.digitize(accel, qs)                                    # 0/1/2 by accel tercile
    top = accel >= np.percentile(accel, TOP_PCT)
    labels = [f"low (≤p{GROUP_QS[0]:g})", f"mid", f"high (≥p{GROUP_QS[1]:g})"]
    return gid.astype(np.int64), top.astype(bool), labels


def _profile_stats(chat: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean and 95%-CI half-width of normalized profile ``chat (M, T-1)`` over rows in
    ``mask``. CI = 1.96·std/√n (mean of the per-step fraction across the group)."""
    sub = chat[mask]
    n = int(sub.shape[0])
    if n == 0:
        w = chat.shape[1]
        return np.full(w, np.nan, np.float32), np.zeros(w, np.float32), 0
    mean = sub.mean(axis=0)
    ci = 1.96 * sub.std(axis=0) / np.sqrt(max(n, 1))
    return mean.astype(np.float32), ci.astype(np.float32), n


def _collect(
    recording: FMRecording, idxs: Sequence[int], action_dim: int, fa: int,
) -> dict[str, np.ndarray]:
    """Pool per-step curvature contributions over every (rollout, env, chunk).

    Returns chunk-level ``c (M, T-1)`` + ``accel (M,)`` and action-level
    ``c_a (Ma, T-1)`` + ``accel_a (Ma,)`` (each action position its own unit), plus the
    ``max|c.sum−accel|`` invariant residual. Each (rollout, env)'s first episode is
    z-scored on its own (matching ``chunk_geometry``), so the per-step contributions
    sum to that episode's published accel."""
    c_chunk, a_chunk, c_act, a_act = [], [], [], []
    max_err = 0.0
    for ri in idxs:
        ro = recording.rollouts[ri]
        for env_idx in range(ro.batch_size):
            nv = _valid_n_chunks_env(ro, env_idx)
            if nv < 1:
                continue
            x_t = np.asarray(ro.x_t[:nv, :, env_idx, :fa, :], dtype=np.float32)   # (nv, T+1, fa, maxA)
            c = whole_chunk_curvature_profile(x_t, action_dim)                    # (nv, T-1)
            acc = whole_chunk_curvature(x_t, action_dim)["accel"]                 # (nv,)
            max_err = max(max_err, float(np.abs(c.sum(1) - acc).max()))
            c_chunk.append(c); a_chunk.append(acc)
            ca = action_level_accel_profile(x_t, action_dim).reshape(-1, c.shape[1])  # (nv*fa, T-1)
            aa = action_level_accel(x_t, action_dim).reshape(-1)                  # (nv*fa,)
            c_act.append(ca); a_act.append(aa)
    return {
        "c_chunk": np.concatenate(c_chunk) if c_chunk else np.zeros((0, 0), np.float32),
        "accel_chunk": np.concatenate(a_chunk) if a_chunk else np.zeros(0, np.float32),
        "c_act": np.concatenate(c_act) if c_act else np.zeros((0, 0), np.float32),
        "accel_act": np.concatenate(a_act) if a_act else np.zeros(0, np.float32),
        "max_sum_err": np.float32(max_err),
    }


def _phase_summary(c: np.ndarray, accel: np.ndarray, time_mid: np.ndarray) -> dict[str, Any]:
    """Per-unit τ_cm (curvature center-of-mass in FM time) + early-fraction (share of
    accel on the noise side), the group-mean normalized profiles, and ρ(τ_cm, accel)."""
    M, W = c.shape
    pos = accel > EPS
    chat = np.zeros_like(c)
    chat[pos] = c[pos] / accel[pos, None]                          # within-unit fraction (Σ=1)
    tau_cm = chat @ time_mid.astype(np.float32)                    # (M,) FM-time of the bend
    early = c.shape[1] - (c.shape[1] // 2)                         # noise-side half (ceil)
    early_frac = chat[:, :early].sum(axis=1)                       # high-time (noise) share
    gid, top, labels = _group_ids(accel)
    groups = []
    for g, lab in enumerate(labels):
        mean, ci, n = _profile_stats(chat, gid == g)
        groups.append({"label": lab, "n": n, "mean": mean, "ci": ci,
                       "tau_cm_mean": float(np.nanmean(tau_cm[gid == g])) if n else float("nan"),
                       "early_frac_mean": float(np.nanmean(early_frac[gid == g])) if n else float("nan")})
    top_mean, top_ci, top_n = _profile_stats(chat, top)
    groups.append({"label": f"top {100 - TOP_PCT:g}% (≥p{TOP_PCT:g})", "n": top_n,
                   "mean": top_mean, "ci": top_ci,
                   "tau_cm_mean": float(np.nanmean(tau_cm[top])) if top_n else float("nan"),
                   "early_frac_mean": float(np.nanmean(early_frac[top])) if top_n else float("nan")})
    return {"chat": chat, "tau_cm": tau_cm, "early_frac": early_frac, "gid": gid,
            "top": top, "groups": groups, "rho_cm_accel": _spearman(tau_cm, accel)}


# ------------------------------------------------------------------------ plotting
def _plot_profiles(time_mid, chunk_groups, action_groups, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, groups, lvl in ((axes[0], chunk_groups, "chunk-level"),
                            (axes[1], action_groups, "action-level")):
        for g in groups:
            if not g["n"]:
                continue
            dashed = g["label"].startswith("top")
            ax.plot(time_mid, g["mean"], "-" if not dashed else "--", lw=2 if not dashed else 1.6,
                    label=f"{g['label']} (n={g['n']})")
            if not dashed:
                ax.fill_between(time_mid, g["mean"] - g["ci"], g["mean"] + g["ci"], alpha=0.18)
        ax.invert_xaxis()                                          # noise (t=1) left → action (t≈0) right
        ax.set_xlabel("FM time  (1.0 = noise  →  0.1 = action)")
        ax.set_title(lvl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("normalized curvature  ĉ_t = c_t / accel  (fraction of bend at step t)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_cm(accel_c, tau_c, rho_c, accel_a, tau_a, rho_a, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, acc, tau, rho, lvl in ((axes[0], accel_c, tau_c, rho_c, "chunk-level"),
                                   (axes[1], accel_a, tau_a, rho_a, "action-level")):
        if len(acc):
            ax.scatter(acc, tau, s=6, alpha=0.25, edgecolors="none")
        ax.set_xlabel("total accel = Σ‖Δv‖/⟨‖v‖⟩")
        ax.set_ylabel("τ_cm  (FM time of curvature center-of-mass)")
        ax.set_title(f"{lvl}   ρ(τ_cm, accel) = {rho:+.3f}")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- driver
def run_accel_profile(
    run: Any,
    *,
    rollouts: Sequence[int] | None = None,
    max_rollouts: int | None = None,
    window: str = "full",
    progress: bool = True,
) -> tuple[runs.RunDir, dict]:
    """Decompose ``accel`` over the denoise schedule for one recording.

    Reads only ``<run>/fm/``. ``window`` selects the action sub-chunk the path is read
    over: ``"full"`` (whole chunk_size) or ``"exec"`` (first n_action_steps, the
    executed window). Writes ``<run>/accel_profile/`` and returns ``(run, summary)``."""
    rd = run if isinstance(run, runs.RunDir) else runs.resolve_run(run)
    recording = FMRecording.load(rd.fm_dir)
    if not recording.rollouts:
        raise ValueError(f"run {rd.run_id!r} has no recorded rollouts under {rd.fm_dir}")
    dims = recording.dims
    action_dim = int(dims["action_dim"])
    chunk_size = int(dims["chunk_size"])
    n_action_steps = int(dims["n_action_steps"])
    fa = chunk_size if window == "full" else max(1, min(n_action_steps, chunk_size))

    if rollouts is None:
        idxs = list(range(len(recording.rollouts)))
        if max_rollouts is not None:
            idxs = idxs[: int(max_rollouts)]
    else:
        idxs = [int(i) for i in rollouts]

    time_mid = _time_mid(recording.rollouts[idxs[0]].time[0])      # shared FM grid
    logger.info("accel_profile on run %s: %d rollouts, window=%s (fa=%d), T-1=%d steps",
                rd.run_id, len(idxs), window, fa, len(time_mid))

    pool = _collect(recording, idxs, action_dim, fa)
    c_chunk, accel_chunk = pool["c_chunk"], pool["accel_chunk"]
    c_act, accel_act = pool["c_act"], pool["accel_act"]
    if not len(accel_chunk):
        raise ValueError(f"run {rd.run_id!r}: no valid chunks collected")
    logger.info("pooled %d chunks / %d action-paths | max|Σc_t − accel| = %.2e",
                len(accel_chunk), len(accel_act), float(pool["max_sum_err"]))

    chunk = _phase_summary(c_chunk, accel_chunk, time_mid)
    action = _phase_summary(c_act, accel_act, time_mid)

    out_dir = rd.accel_profile_dir
    _plot_profiles(time_mid, chunk["groups"], action["groups"],
                   out_dir / "profile.png",
                   f"{rd.run_id}: where the denoise bend lives  (ĉ_t vs FM time, by accel group)")
    _plot_cm(accel_chunk, chunk["tau_cm"], chunk["rho_cm_accel"],
             accel_act, action["tau_cm"], action["rho_cm_accel"],
             out_dir / "cm_vs_accel.png",
             f"{rd.run_id}: curvature center-of-mass vs total accel")

    np.savez_compressed(
        out_dir / NPZ_NAME,
        time_mid=time_mid,
        c_chunk=c_chunk, accel_chunk=accel_chunk,
        chat_chunk=chunk["chat"], tau_cm_chunk=chunk["tau_cm"],
        early_frac_chunk=chunk["early_frac"], gid_chunk=chunk["gid"], top_chunk=chunk["top"],
        tau_cm_action=action["tau_cm"], accel_action=accel_act,
        early_frac_action=action["early_frac"],
        group_means_chunk=np.stack([g["mean"] for g in chunk["groups"]]),
        group_means_action=np.stack([g["mean"] for g in action["groups"]]),
        group_labels=np.array([g["label"] for g in chunk["groups"]], dtype=object),
    )

    def _groups_meta(gr):
        return [{"label": g["label"], "n": g["n"], "tau_cm_mean": g["tau_cm_mean"],
                 "early_frac_mean": g["early_frac_mean"]} for g in gr]

    summary = {
        "run_id": rd.run_id, "window": window, "fa": fa,
        "n_chunks": int(len(accel_chunk)), "n_actions": int(len(accel_act)),
        "n_denoise_jerk_terms": int(len(time_mid)),
        "time_mid": [float(x) for x in time_mid],
        "max_sum_err": float(pool["max_sum_err"]),
        "rho_cm_accel_chunk": float(chunk["rho_cm_accel"]),
        "rho_cm_accel_action": float(action["rho_cm_accel"]),
        "groups_chunk": _groups_meta(chunk["groups"]),
        "groups_action": _groups_meta(action["groups"]),
    }
    with open(out_dir / META_NAME, "w") as f:
        json.dump(summary, f, indent=2)
    rd.record_stage("accel_profile", {"window": window, "n_rollouts": len(idxs)})
    return rd, summary
