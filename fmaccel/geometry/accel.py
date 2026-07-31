"""``accel`` — the paper's denoising acceleration — and the stage that validates it.

**The claim this file implements.** For conditional flow matching with a linear interpolant,
``v(x,s) = (E[x1|x_s=x] - x)/(1-s)``. If the endpoint posterior collapses, the Jacobian is
``J = -I/(1-s)`` — an affine isotropic contraction whose integral curves are straight lines at
constant velocity, so acceleration is exactly zero. Departure from that template is not a
heuristic correlate of uncertainty; via a second-order Tweedie identity it *equals* the posterior
covariance up to a schedule factor. ``accel`` measures that departure as the normalized total
variation of the denoising velocity (Algorithm 1 in the paper):

    accel_p = p * sum_{t=1..p-1} ||v_t - v_{t-1}||  /  sum_{t=0..p-1} ||v_t||

It is **free**: the ``v_t`` already exist in the policy's normal forward pass, so this is numpy
over an array the eval already wrote — no extra model evaluations, no training, no resampling.
See ``docs/method.md`` §1.

This module also carries the stage that reads it off a recorded episode alongside *Straightness*
(chord/arc of the same path) and the *resample divergence* ground truth, and reports the rank
correlation between the free proxies and that GT — both per-episode and pooled over the run.

The accel↔divergence link is reported at **two granularities**:
  * **chunk level** — the whole chunk flattened into one ``chunk_size*act`` point; one
    accel and one divergence per chunk.
  * **action level** — **each action position inside the chunk is its own unit**: its
    own ``(T+1, act)`` denoise path (→ accel) and its own ``k``-candidate spread at
    that position (→ divergence), so a chunk yields ``chunk_size`` accel/divergence
    pairs. A pooled ``accel↔divergence`` scatter is drawn for both granularities.

Why ``accel`` and not Straightness: the chord/arc ratio saturates near 1.0 for plain CFM (~97% of
chunks land in [0.99, 1.0]), so although it correlates with the expensive divergence it can barely
*resolve* chunks. ``accel`` reads the same bend without the saturating denominator, keeping the
correlation with usable dynamic range. Both are reported — the contrast is the point. The stage
records rank correlations two ways, so the claim is re-verified on every run (see ``meta.json``):

  * ``mean_within_episode_rho_*`` — the per-episode ρ averaged across episodes.
  * ``run_rho_*`` — **one ρ pooled over every chunk of every episode** (the headline
    run-level number; NOT a mean of per-episode ρ).

It also reports a **prefix-accel ρ(t) curve** (chunk level only — action level
intentionally off): ``accel`` of the growing *noise→t denoise prefix* (the first ``j+2``
of ``T`` flow steps, each prefix self-normalized; the last cutoff equals the full
``accel``) is pooled and rank-correlated against the same divergence GT *per denoise
depth*. The curve answers **how early along the denoise the free proxy already ranks chunks like
the resample GT**, i.e. how few flow steps suffice. Its PEAK is the paper's ``rho_best`` at ``p*``:
``accel`` sums velocity differences, so the final Euler steps contribute the most discretization
noise while carrying the least signal, which depresses the full-path number and puts the
best-aligned readout mid-schedule. That is a property of the estimator, not a tuned choice — the
whole curve is reported, not just its argmax. Written to ``prefix_accel_rho.{png,npz}`` +
``run_rho_prefix_accel_vs_div`` in ``meta.json``.

This is a pure **analysis stage on an existing run** (numpy + matplotlib only — no
torch / lerobot / gym, so it imports without lerobot installed and needs no
GPU). It reuses two artifacts the upstream stages already wrote:

  * **straightness / accel** ← the FM recording (``<run>/fm/``). For each chunk the
    recorded denoising state ``x_t`` is ``(T+1, chunk_size, act)``; we **flatten the
    chunk** into one ``chunk_size*act`` vector, giving a single ``T+1``-point path per
    chunk. Straightness = ``‖x_T - x_0‖ / Σ_t ‖Δx_t‖`` of that whole-chunk path (1.0 =
    a perfectly straight noise→action shot; <1.0 = a curved/kinked denoise). Contrast
    ``metrics.straightness``/``posterior``, which score each chunk *position*
    separately — here the unit is the entire chunk.

  * **divergence** ← the ``chunk_divergence`` stage's npz (``<run>/chunk_divergence/
    chunk_divergence[_roN].npz``), which already stored ``chunks`` ``(n_step, k,
    chunk_size, act)`` (the ``k`` candidate plans resampled at every env step). The
    **resample unit here is the action chunk, not the action**: that npz resampled
    densely (every env step), so we keep only the chunk-start rows — the ``k``
    candidates drawn at each chunk's own re-plan observation — giving one k-candidate
    set per chunk. We then **flatten each candidate** to a ``chunk_size*act`` vector
    (each action dim **per-dim-standardized** by its run-pooled std, so dims weigh
    comparably — matching the per-dim z-score the accel side uses) and measure the
    pairwise spread of the ``k`` whole-chunk points (max + mean over the
    ``k*(k-1)/2`` pairs). Contrast ``chunk_divergence.pairwise_chunk_spread``,
    which averages a per-position L2 at *every* step — here the chunk is one point in
    flattened space and is sampled once per re-plan, so the spread is a true
    whole-plan distance.

Output (into ``<run>/chunk_geometry/``):
  * ``chunk_geometry[_roN].npz`` (per episode) — straightness, accel/turn_total/arc_chord,
    divergence max/mean, the action-level ``accel_action``/``divergence_action_max``
    ``(N, chunk_size)`` arrays, and per-chunk env-step bookkeeping.
  * ``geometry[_roN].png`` (per episode) — whole-chunk straightness + accel + per-chunk
    divergence along the episode.
  * ``accel_divergence_scatter.png`` (run level) — pooled accel↔divergence scatter,
    chunk-level panel vs action-level panel, each annotated with its Spearman ρ.
  * top-level ``meta.json`` (per-episode ρ + the pooled run-level chunk & action ρ).

Any run with an fm recording + a ``chunk_divergence`` npz works (that stage is the
producer of the npz); divergence is skipped (with a warning) for episodes missing it.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from fmaccel.core import runs
from fmaccel.core.io import write_json
from fmaccel.recording.loader import FMRecording

logger = logging.getLogger(__name__)

EPS = 1e-12


# --------------------------------------------------------------------------- stats
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation of two 1-D arrays (nan if <3 points or a constant input)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(a) != len(b):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); ra -= ra.mean()
    rb = np.argsort(np.argsort(b)).astype(float); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > EPS else float("nan")


# --------------------------------------------------------------------------- math
def _path_straightness(traj: np.ndarray) -> np.ndarray:
    """Chord/arc ratio of each ``(..., T+1, dim)`` path along axis -2. -> ``(...,)``.

    1.0 = the points are collinear and monotone (a straight shot); smaller = the
    path doubles back / curves. Sum of step lengths in the denominator, endpoint
    distance in the numerator.
    """
    diffs = np.diff(traj, axis=-2)                              # (..., T, dim)
    path = np.linalg.norm(diffs, axis=-1).sum(axis=-1)         # (...,)
    chord = np.linalg.norm(traj[..., -1, :] - traj[..., 0, :], axis=-1)  # (...,)
    return chord / (path + EPS)


def _accel_per_step(xz: np.ndarray) -> np.ndarray:
    """Per-denoise-step curvature contributions of z-scored paths ``xz``.

    Operates on the last two axes ``(..., T+1, dim)`` and returns ``(..., T-1)``: the
    term ``c_t = ‖Δv_{t+1}−Δv_t‖ / ⟨‖Δv_t‖⟩`` (jerk magnitude at denoise step ``t``,
    normalized by the path's *mean* step). ``_accel_per_step(xz).sum(-1) == _accel(xz)``
    by construction — so this is the **phase decomposition** of ``accel``: it says
    *where along the denoise schedule* the bend lives, summing back to the headline
    curvature. The ``mean_step`` denominator is one scalar per path (shared across all
    ``T-1`` terms), so the contributions are directly comparable across steps.
    """
    steps = np.diff(xz, axis=-2)                               # (..., T, dim)  ∝ v_t
    steplen = np.linalg.norm(steps, axis=-1)                   # (..., T)
    mean_step = steplen.mean(axis=-1)                          # (...,)
    jerk = np.linalg.norm(np.diff(steps, axis=-2), axis=-1)   # (..., T-1)  ‖Δv_{t+1}−Δv_t‖
    return jerk / (mean_step[..., None] + EPS)                # (..., T-1)


def _resolve_std(flat: np.ndarray, fixed_std: np.ndarray | None) -> np.ndarray:
    """Per-dim normalizer for the flattened path ``flat`` ``(N, T+1, KA)``.

    Default (``fixed_std is None``) = the in-batch std ``flat.std(axis=(0,1))`` (z-score
    against the chunks passed in: the whole episode offline, or the single chunk for the
    k=1 online gate). Passing ``fixed_std`` ``(KA,)`` substitutes an EXTERNAL reference —
    e.g. a demo-distribution std so the online k=1 score lands on the offline label scale
    (``score_chunks`` then yields demo-comparable accels regardless of how few chunks are
    in the batch). Its length must equal the flattened dim ``KA = window * action_dim``."""
    if fixed_std is None:
        return flat.std(axis=(0, 1))
    sd = np.asarray(fixed_std, dtype=np.float32)
    if sd.ndim != 1 or sd.shape[0] != flat.shape[-1]:
        raise ValueError(f"fixed_std shape {sd.shape} != flattened dim ({flat.shape[-1]},) — "
                         f"action_dim / exec-window must match how the std was computed")
    return sd


def _accel(xz: np.ndarray) -> np.ndarray:
    """``accel = Σ_t‖Δv_{t+1}−Δv_t‖ / ⟨‖Δv_t‖⟩`` of z-scored paths ``xz``.

    Operates on the last two axes ``(..., T+1, dim)`` and returns ``(...,)`` — the
    free, non-saturating denoise-path curvature used at both chunk and action level.
    The per-step decomposition is :func:`_accel_per_step` (``.sum(-1)`` recovers this).
    """
    return _accel_per_step(xz).sum(axis=-1)


def _prefix_accel(xz: np.ndarray) -> np.ndarray:
    """Cumulative ``accel`` over growing **noise→t prefixes** of z-scored paths ``xz``.

    Operates on the last two axes ``(..., T+1, dim)`` and returns ``(..., T-1)``: column
    ``j`` is :func:`_accel` evaluated on the *sub-path* ``xz[..., : j+3, :]`` — the first
    ``j+2`` denoise steps (noise → the ``(j+2)``-th denoise point), i.e. the same
    ``Σ‖Δv_{t+1}−Δv_t‖ / ⟨‖Δv‖⟩`` curvature truncated at a growing denoise depth. Each
    prefix is read as a *self-contained* trajectory — its jerk sum normalized by *its
    own* mean step — so column ``j`` answers "if the denoiser had stopped here, how
    curved is the path so far?". The last column (``j = T-2``, the full path) **equals**
    ``_accel(xz)`` exactly, so prefix-accel is a strict refinement of the headline accel.
    """
    steps = np.diff(xz, axis=-2)                               # (..., T, dim)  ∝ v_t
    steplen = np.linalg.norm(steps, axis=-1)                   # (..., T)
    jerk = np.linalg.norm(np.diff(steps, axis=-2), axis=-1)   # (..., T-1)  ‖Δv_{t+1}−Δv_t‖
    csum_jerk = np.cumsum(jerk, axis=-1)                       # (..., T-1)  Σ over the first (col+1) jerks
    cs_step = np.cumsum(steplen, axis=-1)                      # (..., T)    Σ over the first steps
    counts = np.arange(2, steplen.shape[-1] + 1)              # (T-1,)  prefix length in steps: 2..T
    mean_step = cs_step[..., 1:] / counts                     # (..., T-1)  mean step over the first (col+2) steps
    return csum_jerk / (mean_step + EPS)                       # (..., T-1)


def whole_chunk_straightness(x_t: np.ndarray, action_dim: int) -> np.ndarray:
    """Per-chunk whole-chunk straightness of the recorded denoising path.

    ``x_t`` is ``(N, T+1, chunk_size, max_action_dim)`` for one episode's ``N`` chunks.
    We truncate to ``action_dim``, flatten the chunk into a single ``chunk_size*act``
    vector (so each chunk is one ``T+1``-point path), z-score per flattened dim across
    the whole episode (dims comparable), and return ``straight`` ``(N,)`` — the
    chord/arc straightness in the full flattened space.
    """
    N, Tp1 = x_t.shape[0], x_t.shape[1]
    flat = x_t[..., :action_dim].reshape(N, Tp1, -1).astype(np.float32)   # (N, T+1, KA)
    sd = flat.std(axis=(0, 1)) + EPS                                       # (KA,)
    flatz = flat / sd
    return _path_straightness(flatz)                                      # (N,)


def whole_chunk_curvature(x_t: np.ndarray, action_dim: int,
                          fixed_std: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Free, *non-saturating* curvature readouts of the recorded denoising path.

    Straightness (chord/arc) saturates near 1.0 for plain CFM, so 97% of chunks pile
    into [0.99, 1.0] and the metric can barely resolve them — even though it correlates
    with the (expensive) per-chunk divergence at ρ≈−0.7. These readouts capture the
    *same* bend without the saturating denominator, so they keep that correlation while
    spreading over a usable range. All come from the z-scored flattened path
    ``xz`` (``(N, T+1, chunk_size*act)``); its step ``Δxz_t`` is proportional to the
    recorded velocity ``v_t`` (Euler step ``Δx = v·dt``, ``dt`` constant), so nothing
    beyond the recording is needed — exactly as free as straightness.

      * ``accel`` ``(N,)`` — ``Σ_t‖Δxz_{t+1}−Δxz_t‖ / ⟨‖Δxz_t‖⟩``: total change of the
        path tangent over the denoise, normalized by mean step (scale-free). **Headline**
        — best divergence correlation (ρ≈+0.72) and least saturated of the family.
      * ``turn_total`` ``(N,)`` — ``Σ_t (1 − cos(Δxz_t, Δxz_{t+1}))``: pure direction
        turning (ignores speed changes).
      * ``arc_chord`` ``(N,)`` — ``path − chord`` in z-scored units: straightness'
        numerator with no division, so it never saturates.
    """
    N, Tp1 = x_t.shape[0], x_t.shape[1]
    flat = x_t[..., :action_dim].reshape(N, Tp1, -1).astype(np.float32)
    xz = flat / (_resolve_std(flat, fixed_std) + EPS)                      # (N, T+1, KA)
    steps = np.diff(xz, axis=1)                                            # (N, T, KA)  ∝ v_t
    steplen = np.linalg.norm(steps, axis=-1)                              # (N, T)
    path = steplen.sum(axis=1)                                            # (N,)
    chord = np.linalg.norm(xz[:, -1] - xz[:, 0], axis=-1)                 # (N,)
    accel = _accel(xz)                                                    # (N,)
    sh = steps / (steplen[..., None] + EPS)                               # unit tangents
    cos_adj = (sh[:, :-1] * sh[:, 1:]).sum(axis=-1)                       # (N, T-1)
    turn_total = (1.0 - cos_adj).sum(axis=1)                              # (N,)
    return {"accel": accel.astype(np.float32),
            "turn_total": turn_total.astype(np.float32),
            "arc_chord": (path - chord).astype(np.float32)}


def prefix_fm_time(time_row: np.ndarray) -> np.ndarray:
    """FM time *reached* by each noise→t prefix (the last denoise step folded in).

    ``time_row`` is ``(T,)`` descending 1.0→~0.1 (the FM times ``v`` was evaluated at).
    Prefix column ``j`` integrates the first ``j+2`` denoise steps, whose last step ran
    at ``time_row[j+1]``; returns ``time_row[1:]`` ``(T-1,)`` (descending — high = the
    prefix stopped near the noise end, low = it denoised down near the clean action)."""
    return np.asarray(time_row, np.float32)[1:]


def whole_chunk_prefix_accel(x_t: np.ndarray, action_dim: int,
                             fixed_std: np.ndarray | None = None) -> np.ndarray:
    """Per-chunk **prefix accel**: ``accel`` of the growing noise→t denoise prefix.

    Same z-scored flattened whole-chunk path as :func:`whole_chunk_curvature` (per-dim
    std over the *whole episode*, so the prefixes live in the headline accel's space),
    but the curvature is accumulated over only the first ``j+2`` denoise steps for each
    column ``j`` (:func:`_prefix_accel`). Returns ``prefix`` ``(N, T-1)`` with
    ``prefix[:, -1] == whole_chunk_curvature(x_t, action_dim)["accel"]`` (float eps):
    column ``j`` is the accel a denoiser would have measured had it stopped at that
    depth. Pair it with :func:`prefix_fm_time` to ask *how early along the denoise* the
    accel signal that tracks divergence is already present — i.e. how few of the ``T``
    flow steps suffice for the free proxy to rank chunks like the resample GT does.
    """
    N, Tp1 = x_t.shape[0], x_t.shape[1]
    flat = x_t[..., :action_dim].reshape(N, Tp1, -1).astype(np.float32)
    xz = flat / (_resolve_std(flat, fixed_std) + EPS)                      # (N, T+1, KA)
    return _prefix_accel(xz).astype(np.float32)                            # (N, T-1)


def score_chunks(x_t: np.ndarray, action_dim: int, *, n_exec: int | None = None,
                 mode: str = "accel_prefix:7", fixed_std: np.ndarray | None = None) -> np.ndarray:
    """Score recorded denoise paths by one named ``accel`` variant — the single entry
    point the ``accel`` detector uses (:mod:`fmaccel.detectors.accel`).

    ``x_t`` is ``(k, T+1, chunk, max_action_dim)`` — every Euler iterate of each of the ``k``
    paths (noise → action). Truncated to ``action_dim``, then scored by ``mode`` (lower =
    straighter denoise = lower posterior spread). ``n_exec`` is the executed action window
    ``min(n_action_steps, chunk_size)``; modes that say *exec* restrict the flattened path to
    the first ``n_exec`` chunk positions, which is what the paper reports ("both scores are
    computed only over the executed action window rather than the full action horizon").

    Modes:

    * ``"accel"``               — whole-chunk accel over **all** ``chunk`` positions, full denoise.
    * ``"accel_exec"``          — accel over the first ``n_exec`` positions, full denoise.
    * ``"accel_prefix:<j>"``    — accel over the executed window at **denoise depth ``j+2``**
      (so ``accel_prefix[:, -1] == accel_exec``). A *denoise-axis* prefix, NOT "first ``j``
      chunk steps": the paper's prefix ``accel_p`` with ``p = j+2``. The deployed default is
      the second-to-last prefix. **Default ``j=7``.**
    * ``"accel_prefix_full:<j>"`` — same denoise prefix, over all ``chunk`` positions.
    * ``"window:<n>"``          — accel over the first ``n`` chunk positions, full denoise
      (a *position* prefix, not a denoise-depth one).

    Returns ``accel`` ``(k,)`` float32.
    """
    arr = np.asarray(x_t, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"x_t must be (k, T+1, chunk, act); got {arr.shape}")
    base, sep, col = mode.partition(":")
    exec_slice = slice(0, int(n_exec)) if n_exec is not None else slice(None)

    if base == "accel":
        return whole_chunk_curvature(arr, action_dim, fixed_std)["accel"]
    if base == "accel_exec":
        return whole_chunk_curvature(arr[:, :, exec_slice, :], action_dim, fixed_std)["accel"]
    if base in ("accel_prefix", "accel_prefix_full"):
        if not sep:
            raise ValueError(f"{base!r} needs a depth, e.g. {base}:7")
        j = int(col)
        positions = exec_slice if base == "accel_prefix" else slice(None)
        prefix = whole_chunk_prefix_accel(arr[:, :, positions, :], action_dim, fixed_std)  # (k, T-1)
        if not -prefix.shape[1] <= j < prefix.shape[1]:
            raise IndexError(f"prefix depth {j} out of range for {prefix.shape[1]} denoise "
                             f"depths (T={arr.shape[1] - 1}); use a smaller j")
        return prefix[:, j].astype(np.float32)
    if base == "window":
        if not sep:
            raise ValueError("window mode needs a length, e.g. window:8")
        n = int(col)
        return whole_chunk_curvature(arr[:, :, :n, :], action_dim, fixed_std)["accel"]
    raise ValueError(f"unknown accel mode {mode!r}")


def whole_chunk_divergence(chunks: np.ndarray, scale_act: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit whole-chunk divergence of the ``k`` candidate plans.

    ``chunks`` is ``(M, k, chunk_size, act)`` — one row per resample unit (the caller
    passes one row *per action chunk*, i.e. the chunk-start resamples). Each candidate
    is flattened to a ``chunk_size*act`` point; per row we take the max and mean of the
    ``k*(k-1)/2`` pairwise Euclidean distances in the full flattened space. Returns
    ``(max_full, mean_full)`` each ``(M,)``.

    ``scale_act`` ``(act,)`` **per-dim-standardizes** the distance: each action dim is
    divided by its run-pooled std (tiled over the ``chunk_size`` flattened positions)
    so dims weigh comparably — matching the per-dim z-score the accel side already uses.
    ``None`` keeps the raw flattened distance.
    """
    ns, k = chunks.shape[0], chunks.shape[1]
    K = chunks.shape[2]
    flat = chunks.reshape(ns, k, -1).astype(np.float32)                   # (n_step, k, KA)
    if scale_act is not None:
        flat = flat / np.tile(np.asarray(scale_act, np.float32), K)       # per-dim standardize (act tiled over positions)
    iu = np.triu_indices(k, 1)

    def _spread(pts: np.ndarray) -> tuple[float, float]:
        if pts.shape[0] < 2:
            return 0.0, 0.0
        diff = pts[:, None] - pts[None, :]                                # (k, k, dim)
        dist = np.linalg.norm(diff, axis=-1)[iu]                          # (k*(k-1)/2,)
        return float(dist.max()), float(dist.mean())

    max_full = np.zeros(ns, np.float32); mean_full = np.zeros(ns, np.float32)
    for t in range(ns):
        max_full[t], mean_full[t] = _spread(flat[t])
    return max_full, mean_full


# --------------------------------------------------------------------- action level
def action_level_accel(x_t: np.ndarray, action_dim: int) -> np.ndarray:
    """Per-(chunk, action-position) accel of the recorded denoise path.

    ``x_t`` is ``(N, T+1, chunk_size, max_action_dim)``. Unlike the chunk-level readout
    (which flattens all positions into one ``chunk_size*act`` path), here **each action
    position is its own ``(T+1, act)`` denoise path** — so a chunk yields ``chunk_size``
    accel values. We z-score per action dim across every action-path of the episode
    (the dims comparable, mirroring the chunk-level normalization) and return ``accel``
    ``(N, chunk_size)``.
    """
    N, Tp1, K = x_t.shape[0], x_t.shape[1], x_t.shape[2]
    paths = np.transpose(x_t[..., :action_dim], (0, 2, 1, 3)).astype(np.float32)  # (N, K, T+1, act)
    flat = paths.reshape(N * K, Tp1, action_dim)                          # (N*K, T+1, act)
    xz = flat / (flat.std(axis=(0, 1)) + EPS)                            # per act-dim z-score
    return _accel(xz).reshape(N, K).astype(np.float32)                    # (N, K)


# ---------------------------------------------------- denoise-phase decomposition
def whole_chunk_curvature_profile(x_t: np.ndarray, action_dim: int) -> np.ndarray:
    """Per-(chunk, denoise-step) curvature contribution of the recorded path.

    Phase decomposition of :func:`whole_chunk_curvature`'s ``accel``: same z-scored
    flattened path (``(N, T+1, chunk_size*act)``, per-dim std over the episode), but
    the per-step jerk terms are *kept* rather than summed. Returns ``c`` ``(N, T-1)``
    with ``c.sum(1) == whole_chunk_curvature(x_t, action_dim)["accel"]`` (float eps).
    Use it to ask *where along the denoise schedule* a chunk's bend concentrates.
    """
    N, Tp1 = x_t.shape[0], x_t.shape[1]
    flat = x_t[..., :action_dim].reshape(N, Tp1, -1).astype(np.float32)
    xz = flat / (flat.std(axis=(0, 1)) + EPS)                             # (N, T+1, KA)
    return _accel_per_step(xz).astype(np.float32)                         # (N, T-1)


def action_level_accel_profile(x_t: np.ndarray, action_dim: int) -> np.ndarray:
    """Per-(chunk, action-position, denoise-step) curvature contribution.

    Phase decomposition of :func:`action_level_accel`: each action position is its own
    ``(T+1, act)`` path (z-scored per act-dim over every action-path of the episode),
    with the per-step jerk terms kept. Returns ``c`` ``(N, chunk_size, T-1)`` with
    ``c.sum(-1) == action_level_accel(x_t, action_dim)``.
    """
    N, Tp1, K = x_t.shape[0], x_t.shape[1], x_t.shape[2]
    paths = np.transpose(x_t[..., :action_dim], (0, 2, 1, 3)).astype(np.float32)  # (N, K, T+1, act)
    flat = paths.reshape(N * K, Tp1, action_dim)                          # (N*K, T+1, act)
    xz = flat / (flat.std(axis=(0, 1)) + EPS)                            # per act-dim z-score
    return _accel_per_step(xz).reshape(N, K, -1).astype(np.float32)       # (N, K, T-1)


def action_level_divergence(chunks: np.ndarray, scale_act: np.ndarray | None = None) -> np.ndarray:
    """Per-(chunk, action-position) divergence of the ``k`` candidate plans.

    ``chunks`` is ``(M, k, chunk_size, act)``. For each chunk row and each action
    position we take the ``k`` candidates' action at that position ``(k, act)`` and
    measure the max of the ``k*(k-1)/2`` pairwise L2 distances — so a re-plan yields
    ``chunk_size`` divergence values. Returns ``max`` ``(M, chunk_size)``.

    ``scale_act`` ``(act,)`` **per-dim-standardizes** the distance (each action dim
    divided by its run-pooled std before the L2); ``None`` keeps the raw distance.
    """
    M, k, K, act = chunks.shape
    iu = np.triu_indices(k, 1)
    out = np.zeros((M, K), np.float32)
    if k < 2:
        return out
    sc = None if scale_act is None else np.asarray(scale_act, np.float32)
    for m in range(M):
        c = chunks[m].astype(np.float32)                                  # (k, K, act)
        diff = c[:, None] - c[None, :]                                    # (k, k, K, act)
        if sc is not None:
            diff = diff / sc                                             # (act,) broadcast — per-dim standardize
        dist = np.linalg.norm(diff, axis=-1)                             # (k, k, K)
        out[m] = dist[iu].max(axis=0)                                     # (K,)
    return out


# ------------------------------------------------------------------------ plotting
def _plot_summary(div_x, straight_full, accel, chunk_start_step,
                  max_full, mean_full, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    div_x = np.asarray(div_x)                                       # chunk-start env step per resampled chunk
    cs = np.asarray(chunk_start_step)
    fig, (ax0, axa, ax1) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

    # --- straightness (per chunk; saturated near 1.0 — kept for reference) ---
    ax0.plot(cs, straight_full, "-o", ms=3, lw=1.6, color="C0", label="straightness")
    if len(straight_full):
        worst = int(np.argmin(straight_full))
        ax0.annotate(f"min {straight_full[worst]:.2f}", xy=(cs[worst], straight_full[worst]),
                     xytext=(4, 6), textcoords="offset points", fontsize=9, color="C0")
    ax0.set_ylabel("straightness\n(saturated)")
    lo = float(np.min(straight_full)) if len(straight_full) else 0.0
    ax0.set_ylim((max(0.0, lo - 0.03), 1.005) if lo > 0.8 else (0.0, 1.02))  # zoom in when saturated near 1
    ax0.grid(alpha=0.25)
    ax0.legend(loc="lower right", fontsize=8)
    ax0.set_title(title)

    # --- accel: the free, non-saturating curvature readout (tracks divergence, ρ≈+0.72) ---
    axa.plot(cs, accel, "-o", ms=3, lw=1.6, color="C2", label="accel = Σ‖Δv‖/⟨‖v‖⟩")
    if len(accel):
        hi = int(np.argmax(accel))
        axa.annotate(f"max {accel[hi]:.2f}", xy=(cs[hi], accel[hi]),
                     xytext=(4, -10), textcoords="offset points", fontsize=9, color="C2")
    axa.set_ylabel("denoise accel\n(discriminative)")
    axa.grid(alpha=0.25)
    axa.legend(loc="upper right", fontsize=8)

    # --- divergence (per action chunk: one resample at each re-plan observation) ---
    ax1.plot(div_x, max_full, "-o", ms=3, lw=1.7, color="C3", label="max")
    ax1.plot(div_x, mean_full, "-o", ms=2, lw=1.0, color="C0", alpha=0.7, label="mean")
    if len(max_full):
        peak = int(np.argmax(max_full))
        xp = int(div_x[peak])
        ax1.axvline(xp, color="0.6", ls=":", lw=1.0)
        ax1.annotate(f"peak {max_full[peak]:.1f}@chunk{peak} (step{xp})", xy=(xp, max_full[peak]),
                     xytext=(5, -2), textcoords="offset points", fontsize=9, color="C3")
    ax1.set_xlabel("env step at each chunk's re-plan (one resample per action chunk)")
    ax1.set_ylabel("whole-chunk\ndivergence (L2)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_accel_div_scatter(chunk_accel, chunk_div, action_accel, action_div,
                            rho_chunk, rho_action, out_path, title, clip_pct=97.0):
    """Pooled-over-the-run accel↔divergence scatter, chunk level vs action level.

    accel and divergence are both heavy-tailed (a few catastrophic forks dwarf the
    bulk), so the view is clipped to the per-axis ``clip_pct`` percentile to keep the
    dense region readable. The Spearman ρ in each title is rank-based over **all**
    points — clipping only changes what is shown, not the reported correlation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axc, axa) = plt.subplots(1, 2, figsize=(11, 5))
    panels = [
        (axc, chunk_accel, chunk_div, rho_chunk, "chunk", "C0", 0.45,
         "whole-chunk divergence (max pairwise L2)"),
        (axa, action_accel, action_div, rho_action, "action", "C3", 0.12,
         "per-action divergence (max pairwise L2)"),
    ]
    for ax, xa, ya, rho, lab, col, alpha, ylab in panels:
        xa = np.asarray(xa); ya = np.asarray(ya)
        ax.scatter(xa, ya, s=7, alpha=alpha, color=col, edgecolors="none")
        off = 0
        if xa.size:
            xmax = float(np.percentile(xa, clip_pct)) or 1.0
            ymax = float(np.percentile(ya, clip_pct)) or 1.0
            off = int(((xa > xmax) | (ya > ymax)).sum())
            ax.set_xlim(-0.02 * xmax, xmax * 1.03)
            ax.set_ylim(-0.02 * ymax, ymax * 1.03)
        ax.set_xlabel("accel = Σ‖Δv‖/⟨‖v‖⟩  (free denoise-path curvature)")
        ax.set_ylabel(ylab)
        rtxt = f"{rho:+.3f}" if rho == rho else "n/a"
        ax.set_title(f"{lab}-level   Spearman ρ = {rtxt}   "
                     f"(n={len(xa)}; view≤p{clip_pct:g}, {off} off-panel)")
        ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_prefix_rho(prefix_time, n_steps, rho_prefix, full_rho, n_pooled, out_path, title):
    """ρ(prefix-accel, divergence) as a function of how deep the noise→t prefix denoised.

    x-axis is FM time *reached* by the prefix (inverted: noise on the left → clean
    action on the right); the dashed line is the full-path accel ρ (the last point, by
    construction). Reads off how many of the ``T`` flow steps the free proxy needs
    before it ranks chunks like the resample GT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pt = np.asarray(prefix_time, float)
    rp = np.asarray(rho_prefix, float)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(pt, rp, "-o", ms=5, lw=1.8, color="C2", label="ρ(prefix accel, divergence)")
    if full_rho == full_rho:
        ax.axhline(full_rho, color="0.5", ls="--", lw=1.2,
                   label=f"full-path accel ρ = {full_rho:+.3f}")
    for t, r, ns in zip(pt, rp, n_steps):
        if r == r:
            ax.annotate(f"{int(ns)}", xy=(t, r), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color="0.35")
    ax.invert_xaxis()                                              # noise (t=1) left → action (t≈0) right
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.set_xlabel("FM time reached by the noise→t prefix  (1.0 = noise → ~0.1 = action; "
                  "labels = # denoise steps folded in)")
    ax.set_ylabel("Spearman ρ  vs whole-chunk divergence (GT)")
    ax.set_title(f"{title}\nhow early the free accel proxy tracks divergence (n={n_pooled} chunks)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- driver
def _pooled_dim_scale(samples: np.ndarray, eps: float = 1e-6, clip_pct: float = 1.0) -> np.ndarray:
    """Run-pooled per-action-dim **robust** scale for standardizing the divergence distance.

    ``samples`` is ``(..., act)`` pooled over **every** rollout, so the scale is one
    fixed vector for the whole run — divergence is then comparable across episodes
    (which the pooled run-level ρ relies on), not re-scaled per episode. Per dim the unit
    is a **winsorized std**: each dim is clipped to its ``[clip_pct, 100-clip_pct]``
    percentiles before ``std``, so a single catastrophically-diverged rollout can't hijack
    the scale while clean data keeps ~its plain std (IQR would also reweight heavy-tailed /
    bimodal clean dims — see the ``chunk_divergence._pooled_dim_scale`` docstring). Only
    the legacy fallback path uses this; the preferred path reads the scale the
    ``chunk_divergence`` stage already wrote. Near-constant dims (scale < ``eps``) stay
    un-rescaled. Returns ``(act,)`` float32."""
    arr = np.asarray(samples, np.float32)
    flat = arr.reshape(-1, arr.shape[-1])
    lo, hi = np.percentile(flat, [clip_pct, 100.0 - clip_pct], axis=0)
    sd = np.clip(flat, lo, hi).std(axis=0)                                # winsorized (clipped-tail) std
    return np.where(sd < eps, 1.0, sd).astype(np.float32)


def _load_div_chunks(cd_dir: Any, suffix: str, chunk_size: int, *, full_window: bool = False):
    """Load one rollout's chunk-start resamples from the ``chunk_divergence`` npz.

    Returns ``(chunks (n_div, k, fa, act) | None, div_step (n_div,), fa, have_div,
    dim_scale (act,) | None)``. Centralizes the slice-to-chunk-starts +
    ``first_actions``-window logic so the pooled-scale pre-pass and the per-rollout
    main loop read the exact same chunks, and surfaces the per-dim standardization
    scale that the ``chunk_divergence`` stage wrote (the single source of truth, so
    this stage's divergence is standardized by the *same* scale as that stage's
    headline ``max_pairwise_dist``). ``chunks``/``dim_scale`` are ``None`` when the
    npz is missing / has no saved ``chunks`` / predates the scale (legacy npz).

    ``full_window=True`` skips the ``[:, :, :fa, :]`` slice and returns chunks over the FULL
    ``chunk_size`` horizon; ``fa`` is still the real ``first_actions`` so the caller can take the
    executed window itself with one npz read. Needed by cross-time detectors (STAC), whose overlap
    ``cur[:, :chunk-n_exec]`` vs ``prev[:, n_exec:]`` lives entirely PAST the executed window and is
    destroyed by the default truncation. Every geometry caller keeps ``full_window=False``, so the
    accel / divergence numerics are untouched."""
    cd_npz = cd_dir / f"chunk_divergence{suffix}.npz"
    if not cd_npz.exists() and suffix:                                     # single-rollout fallback
        cd_npz = cd_dir / "chunk_divergence.npz"
    if not cd_npz.exists():
        return None, np.zeros(0, np.int64), chunk_size, False, None
    cd_env = None
    dim_scale = None
    with np.load(cd_npz, allow_pickle=True) as cd:
        if "chunks" not in cd.files:
            return None, np.zeros(0, np.int64), chunk_size, False, None
        chunks_dense = np.asarray(cd["chunks"], np.float32)               # (n_row, k, K, act)
        cd_start = np.asarray(cd["chunk_start"], bool)                    # (n_row,)
        fa = max(1, min(int(cd["first_actions"]), chunk_size)) if "first_actions" in cd.files else chunk_size
        if "env_step" in cd.files:                                        # context producer: env step / row
            cd_env = np.asarray(cd["env_step"], np.int64)
        if "dim_scale" in cd.files:                                       # run-pooled per-dim std from chunk_divergence
            dim_scale = np.asarray(cd["dim_scale"], np.float32)
    div_row = np.flatnonzero(cd_start)                                    # rows holding a per-chunk resample
    chunks = chunks_dense[div_row]                                        # (n_div, k, K, act)
    if not full_window:
        chunks = chunks[:, :, :fa, :]                                     # (n_div, k, fa, act)
    div_step = (cd_env[div_row] if cd_env is not None else div_row).astype(np.int64)
    return chunks, div_step, fa, True, dim_scale


def _valid_n_chunks(ro: Any, env_idx: int = 0) -> int:
    """Chunks up to (and including) the one holding the first env-step ``done``."""
    done = np.asarray(ro.terminated[:, env_idx] | ro.truncated[:, env_idx], dtype=bool)
    hits = np.flatnonzero(done)
    if not len(hits):
        return int(ro.n_chunks)
    d_step = int(hits[0])
    starts = np.asarray(ro.env_step_at_chunk_start)
    return int(max(1, min(int(np.searchsorted(starts, d_step, side="right")), ro.n_chunks)))


def run_chunk_geometry(
    run: Any,                       # run-id or RunDir of the producing run
    *,
    rollouts: Sequence[int] | None = None,
    max_rollouts: int | None = 3,
    progress: bool = True,
) -> tuple[runs.RunDir, dict]:
    """Whole-chunk straightness + accel (from ``fm/``) vs divergence (from
    ``chunk_divergence/``) along each recorded episode, with the per-episode summary
    plot and both per-episode and pooled run-level rank correlations.
    Pure numpy/matplotlib — no model load. Returns ``(run, summary)``."""
    from fmaccel.recording import format as fmt

    rd = run if isinstance(run, runs.RunDir) else runs.resolve_run(run)
    # Memory-bounded load: pick the rollout_ids to analyze from the manifest FIRST and
    # hand only those to the loader (each context sidecar is tens of MB — loading all
    # 2000 of a full-run recording would need ~100GB RAM and needless I/O). Mirrors
    # chunk_divergence's filtered load; when `rollouts` is None we take the first
    # `max_rollouts` manifest ids, so the default selection is byte-identical to before
    # (the loader preserves rollout_id on each record, so output naming is unchanged).
    manifest_ids = [int(r["rollout_id"]) for r in fmt.read_manifest(rd.fm_dir)["rollouts"]]
    if rollouts is None:                                # cap only the "all rollouts" default
        want_ids = manifest_ids if max_rollouts is None else manifest_ids[: int(max_rollouts)]
    else:
        want_ids = [int(i) for i in rollouts]          # explicit list is honored in full
        # Fail loud on a mistyped / stale id rather than silently analyzing fewer rollouts:
        # the loader just skips ids absent from the manifest, so an unfiltered raise here is
        # the only place a bad --rollouts entry surfaces (rollout_ids are the manifest ids).
        missing = [i for i in want_ids if i not in set(manifest_ids)]
        if missing:
            raise ValueError(f"run {rd.run_id!r}: requested rollout ids {missing} are not in the "
                             f"recording manifest ({len(manifest_ids)} rollouts, ids "
                             f"{manifest_ids[0]}..{manifest_ids[-1]})")
    recording = FMRecording.load(rd.fm_dir, rollout_ids=want_ids, load_context=False)
    if not recording.rollouts:
        raise ValueError(f"run {rd.run_id!r} has no recorded rollouts under {rd.fm_dir}")
    dims = recording.dims
    action_dim = int(dims["action_dim"])
    n_action_steps = int(dims["n_action_steps"])
    chunk_size = int(dims["chunk_size"])
    cd_dir = rd.root / "chunk_divergence"

    # `recording` now holds exactly the requested rollouts; iterate its positions (the
    # sub-loop and pooled-scale pre-pass both use `idxs` as indices INTO recording.rollouts,
    # and name outputs by ro.rollout_id, which the loader preserved).
    idxs = list(range(len(recording.rollouts)))
    units = [(ri, env_idx) for ri in idxs
             for env_idx in range(recording.rollouts[ri].batch_size)]

    def unit_suffix(ro: Any, env_idx: int) -> str:
        if len(units) <= 1:
            return ""
        base = f"_ro{int(ro.rollout_id)}"
        return f"{base}_e{env_idx}" if ro.batch_size > 1 else base

    out_dir = rd.chunk_geometry_dir
    # Suffix on the RUN's total rollout count (not the filtered subset): a multi-rollout
    # run's chunk_divergence stage wrote `_ro<id>`-suffixed npz, so we must read them by
    # that same suffix even when analyzing only a subset here.
    suffix_all = len(manifest_ids) > 1
    episodes: list[dict] = []
    # pooled-over-the-whole-run buffers: every chunk of every episode that has a
    # divergence GT, so the run-level ρ is one rank-correlation, not a mean of per-episode ρ.
    pool_accel: list[np.ndarray] = []
    pool_accel_full: list[np.ndarray] = []     # whole-chunk accel (all chunk_size positions),
                                               # the other window vs the executed-window pool_accel
    pool_straight: list[np.ndarray] = []
    pool_div: list[np.ndarray] = []
    # prefix-accel pool (chunk level only): accel of the growing noise→t denoise prefix,
    # one (M, T-1) block per episode, aligned row-for-row with pool_div, so the run-level
    # ρ(prefix-accel at cutoff t, divergence) is one pooled rank-correlation per cutoff.
    pool_prefix: list[np.ndarray] = []
    # action-level pools: each (chunk, action-position) is its own unit.
    pool_accel_a: list[np.ndarray] = []
    pool_div_a: list[np.ndarray] = []
    # FM time reached by each noise→t prefix (shared grid; first rollout sets it).
    prefix_time = prefix_fm_time(recording.rollouts[idxs[0]].time[0])      # (T-1,)
    logger.info("chunk_geometry on run %s: %d rollouts", rd.run_id, len(idxs))

    fa = chunk_size                                     # executed-action window (set from the npz below)

    # --- run-pooled per-dim divergence scale (the standardization): each action dim
    #     divides the divergence L2 by one fixed std so dims weigh comparably and
    #     divergence is comparable across episodes for the pooled ρ. PREFER the scale the
    #     chunk_divergence stage already wrote into each npz (computed from the recorded
    #     executed plan — robust, and the SAME scale as that stage's headline
    #     max_pairwise_dist). Only legacy npz (pre-standardization) fall back to recomputing
    #     it here from per-replan candidate means (means, not the raw k spread, so it never
    #     flattens the very forking it measures; in the chunks' own space so it works for
    #     both the normalized π₀.₅ producer and a raw un-normalized producer). ---
    sd_div: np.ndarray | None = None
    scale_src = "none"
    fallback_rows: list[np.ndarray] = []
    for ri, env_idx in units:
        ro = recording.rollouts[ri]
        sfx = unit_suffix(ro, env_idx)
        ch, _, _fa, have, ds = _load_div_chunks(cd_dir, sfx, chunk_size)
        if not have:
            continue
        if ds is not None:
            sd_div = ds                                                  # stored scale (all npz of a run share it)
            scale_src = "chunk_divergence npz (recorded-plan std)"
            break
        fallback_rows.append(ch.mean(axis=1).reshape(-1, action_dim))    # (M*fa, act) per-replan means
    if sd_div is None and fallback_rows:
        sd_div = _pooled_dim_scale(np.concatenate(fallback_rows, axis=0))
        scale_src = "recomputed from candidate means (legacy npz had no dim_scale)"
    if sd_div is not None:
        logger.info("per-dim divergence scale (run-pooled, %s): %s",
                    scale_src, np.array2string(sd_div, precision=3))

    for ri, env_idx in units:
        ro = recording.rollouts[ri]
        rid = int(ro.rollout_id)
        suffix = unit_suffix(ro, env_idx)
        nv = _valid_n_chunks(ro, env_idx)

        # --- locate the divergence npz FIRST: it carries the executed-action window
        #     (``first_actions``), so straightness/accel are read on the *same*
        #     sub-chunk the divergence is measured over (apples-to-apples). The
        #     chunk_divergence stage stored one row per chunk's re-plan observation
        #     (chunk_start all True for the context producer; the chunk-start subset
        #     for the dense gym producer), making the chunk the resample unit.
        #     ``_load_div_chunks`` is the same loader the pooled-scale pre-pass used, so
        #     both read identical chunks. ``div_step`` = the ENV STEP of each resampled
        #     chunk (context producer stores env_step per row; the dense gym producer
        #     marks chunk-starts on env-step rows, so the row index IS the env step). ---
        chunks, div_step, fa, have_div, _ = _load_div_chunks(cd_dir, suffix, chunk_size)
        if not have_div:
            logger.warning("ro%d: no usable chunk_divergence npz (missing, or no saved 'chunks' — "
                           "run chunk_divergence with save_chunks=True); straightness-only.", rid)
            chunks = np.zeros((0, 0, 0, 0), np.float32)

        # --- straightness + free curvature on the first ``fa`` (executed) actions of
        #     the recorded FM denoising path — same window as the divergence unit. ---
        x_t = np.asarray(ro.x_t[:nv, :, env_idx, :fa, :], dtype=np.float32)
        straight_full = whole_chunk_straightness(x_t, action_dim)
        curv = whole_chunk_curvature(x_t, action_dim)                     # accel / turn_total / arc_chord
        accel = curv["accel"]                                             # executed-window accel (first fa)
        # whole-chunk accel (all chunk_size positions) — the OTHER accel window, so we can
        # report which one best tracks the executed-window divergence GT (== accel when fa==chunk).
        accel_full = accel if fa >= chunk_size else \
            whole_chunk_curvature(
                np.asarray(ro.x_t[:nv, :, env_idx, :, :], np.float32), action_dim
            )["accel"]
        prefix_accel = whole_chunk_prefix_accel(x_t, action_dim)          # (nv, T-1) cumulative noise→t accel
        accel_action = action_level_accel(x_t, action_dim)                # (nv, fa)
        chunk_start_step = np.asarray(ro.env_step_at_chunk_start[:nv], dtype=np.int64)

        if have_div:
            max_full, mean_full = whole_chunk_divergence(chunks, scale_act=sd_div)
            div_action = action_level_divergence(chunks, scale_act=sd_div)  # (n_div_chunks, fa)
        else:
            max_full = mean_full = np.zeros(0, np.float32)
            div_action = np.zeros((0, accel_action.shape[1]), np.float32)

        # --- save arrays ---
        np.savez_compressed(
            out_dir / f"chunk_geometry{suffix}.npz",
            straightness_full=straight_full,
            accel=accel, accel_full=accel_full,
            turn_total=curv["turn_total"], arc_chord=curv["arc_chord"],
            prefix_accel=prefix_accel, prefix_fm_time=prefix_time,
            chunk_start_step=chunk_start_step,
            divergence_max_full=max_full, divergence_mean_full=mean_full,
            divergence_chunk_step=div_step,
            accel_action=accel_action, divergence_action_max=div_action,
            divergence_dim_scale=(sd_div if sd_div is not None else np.ones(action_dim, np.float32)),
        )

        # --- plot ---
        succ_title = f"{rd.run_id} ro{rid}" + (f" env{env_idx}" if ro.batch_size > 1 else "")
        _plot_summary(
            div_step, straight_full, accel, chunk_start_step,
            max_full, mean_full,
            out_dir / f"geometry{suffix}.png",
            title=f"{succ_title}: straightness vs free curvature (accel) vs divergence",
        )

        _peak = int(np.argmax(max_full)) if len(max_full) else -1
        # per-episode verification: how well does each FREE readout predict divergence?
        M = min(len(accel), len(max_full))
        rho_acc = _spearman(accel[:M], max_full[:M]) if M else float("nan")
        rho_acc_full = _spearman(accel_full[:M], max_full[:M]) if M else float("nan")
        rho_str = _spearman(straight_full[:M], max_full[:M]) if M else float("nan")
        if have_div and M:                                                # feed the run-level pool
            pool_accel.append(accel[:M]); pool_accel_full.append(accel_full[:M])
            pool_straight.append(straight_full[:M]); pool_div.append(max_full[:M])
            pool_prefix.append(prefix_accel[:M])                           # (M, T-1) aligned with pool_div
        # action level: align the first Ma chunks, then flatten every (chunk, position) pair.
        Ma = min(accel_action.shape[0], div_action.shape[0])
        if have_div and Ma:
            aa = accel_action[:Ma].ravel(); da = div_action[:Ma].ravel()
            rho_acc_action = _spearman(aa, da)
            pool_accel_a.append(aa); pool_div_a.append(da)
        else:
            rho_acc_action = float("nan")
        ep = {
            "rollout_id": rid, "env_idx": int(env_idx),
            "n_chunks": int(nv), "n_div_chunks": int(len(max_full)),
            "straightness_full_mean": float(straight_full.mean()) if len(straight_full) else None,
            "straightness_full_min": float(straight_full.min()) if len(straight_full) else None,
            "accel_mean": float(accel.mean()) if len(accel) else None,
            "accel_max": float(accel.max()) if len(accel) else None,
            "rho_accel_vs_div": rho_acc, "rho_accel_full_vs_div": rho_acc_full,
            "rho_straightness_vs_div": rho_str,
            "rho_accel_vs_div_action": rho_acc_action,
            "n_action_units": int(Ma * accel_action.shape[1]) if have_div else 0,
            "divergence_peak": float(max_full.max()) if len(max_full) else None,
            "divergence_peak_chunk": _peak if _peak >= 0 else None,
            "divergence_peak_env_step": int(div_step[_peak]) if _peak >= 0 else None,
            "divergence_mean": float(max_full.mean()) if len(max_full) else None,
            "have_divergence": bool(have_div),
        }
        episodes.append(ep)
        logger.info("ro%d: chunks=%d straight min=%.3f | accel mean=%.3f | div peak=%s | "
                    "ρ(accel,div) exec=%.2f full=%.2f action=%.2f | ρ(straight,div)=%.2f",
                    rid, nv, ep["straightness_full_min"] or 0.0, ep["accel_mean"] or 0.0,
                    f"{ep['divergence_peak']:.1f}" if ep["divergence_peak"] else "n/a",
                    rho_acc, rho_acc_full, rho_acc_action, rho_str)

    _rho_a = [e["rho_accel_vs_div"] for e in episodes if e["rho_accel_vs_div"] == e["rho_accel_vs_div"]]
    _rho_s = [e["rho_straightness_vs_div"] for e in episodes if e["rho_straightness_vs_div"] == e["rho_straightness_vs_div"]]
    # run-level ρ: pool every chunk across every episode and rank-correlate once.
    if pool_div:
        pa = np.concatenate(pool_accel); ps = np.concatenate(pool_straight); pd_ = np.concatenate(pool_div)
        paf = np.concatenate(pool_accel_full)
        run_rho_accel = _spearman(pa, pd_)
        run_rho_accel_full = _spearman(paf, pd_)
        run_rho_straight = _spearman(ps, pd_)
        n_pooled = int(len(pd_))
    else:
        pa = ps = pd_ = paf = np.zeros(0, np.float32)
        run_rho_accel = run_rho_accel_full = run_rho_straight = float("nan")
        n_pooled = 0
    # prefix-accel run ρ curve: pool the (M, T-1) prefix blocks and rank-correlate each
    # cutoff column against the SAME pooled divergence — one ρ per denoise depth, so the
    # last entry equals run_rho_accel (full path) by construction.
    if pool_prefix and n_pooled:
        pp = np.concatenate(pool_prefix)                                  # (n_pooled, T-1)
        run_rho_prefix = np.array([_spearman(pp[:, j], pd_) for j in range(pp.shape[1])], float)
    else:
        pp = np.zeros((0, len(prefix_time)), np.float32)
        run_rho_prefix = np.full(len(prefix_time), np.nan)
    # action-level run ρ: pool every (chunk, action-position) across every episode.
    if pool_div_a:
        pa_a = np.concatenate(pool_accel_a); pd_a = np.concatenate(pool_div_a)
        run_rho_accel_action = _spearman(pa_a, pd_a)
        n_pooled_a = int(len(pd_a))
    else:
        pa_a = pd_a = np.zeros(0, np.float32)
        run_rho_accel_action = float("nan")
        n_pooled_a = 0

    # pooled accel↔divergence scatter, chunk level vs action level.
    if n_pooled or n_pooled_a:
        _plot_accel_div_scatter(
            pa, pd_, pa_a, pd_a, run_rho_accel, run_rho_accel_action,
            out_dir / "accel_divergence_scatter.png",
            title=(f"{rd.run_id}: accel vs whole-unit divergence (pooled over the run) — "
                   f"chunk n={n_pooled} vs action n={n_pooled_a}"),
        )

    # prefix-accel ρ(t) curve: how early along the denoise the free proxy tracks the GT.
    n_prefix_steps = [int(j + 2) for j in range(len(prefix_time))]         # denoise steps folded into each prefix
    if n_pooled:
        _plot_prefix_rho(
            prefix_time, n_prefix_steps, run_rho_prefix, run_rho_accel, n_pooled,
            out_dir / "prefix_accel_rho.png", title=f"{rd.run_id}: prefix-accel vs divergence",
        )
        np.savez_compressed(
            out_dir / "prefix_accel_rho.npz",
            prefix_fm_time=prefix_time, prefix_n_steps=np.asarray(n_prefix_steps, np.int64),
            run_rho_prefix=run_rho_prefix.astype(np.float32),
            prefix_accel=pp.astype(np.float32), divergence=pd_.astype(np.float32),
        )

    summary = {
        "source_run": rd.run_id, "stage": "chunk_geometry",
        "unit": f"first-{fa} executed sub-chunk ({fa}*action_dim flattened)"
                if fa < chunk_size else "whole action chunk (chunk_size*action_dim flattened)",
        "action_dim": action_dim, "n_action_steps": n_action_steps,
        "chunk_size": chunk_size, "first_actions": int(fa),
        "straightness": f"from fm/ x_t denoising path: chord/arc of the first-{fa}-action flattened path (1.0=straight); SATURATES near 1",
        "accel": ("free, non-saturating curvature: sum of denoise-tangent change / mean step "
                  "(Σ‖Δv‖/⟨‖v‖⟩); more discriminative surrogate for divergence than straightness"),
        "divergence": (f"from chunk_divergence/*.npz 'chunks', subsampled to chunk-start rows and "
                       f"truncated to the first {fa} actions, so the resample unit is the executed "
                       f"sub-chunk (one k-candidate set per re-plan obs): max/mean pairwise L2 of the "
                       f"k flattened plans, each action dim PER-DIM-STANDARDIZED by its run-pooled std"),
        "action_level": (f"each (chunk, action-position) over the first {fa} positions is its own unit: "
                         f"accel of that position's (T+1, act) denoise path vs max pairwise L2 (per-dim "
                         f"standardized) of the k candidates at that position"),
        "prefix_accel": ("chunk-level only (action-level intentionally off): accel of the growing noise→t "
                         "denoise prefix over the first-fa executed window (first j+2 of T flow steps, each "
                         "prefix self-normalized by its own mean step; last cutoff == accel_exec); answers "
                         "how few denoise steps the free proxy needs before it ranks chunks like the "
                         "resample-GT divergence"),
        "divergence_standardization": ("each action dim divided by its run-pooled winsorized std (1%-clipped) before "
                                       "the pairwise L2, so dims weigh comparably and divergence is comparable "
                                       "across episodes (robust so a diverged rollout can't hijack the scale); "
                                       "accel/straightness keep their existing per-episode per-dim z-score "
                                       "(unchanged — per-dim denoise-path std is ~episode-invariant)"),
        "divergence_dim_scale": [float(x) for x in sd_div] if sd_div is not None else None,
        "divergence_dim_scale_source": scale_src,
        # run-level: ONE ρ pooled over all units across episodes (the headline numbers).
        "run_rho_accel_vs_div": run_rho_accel,                  # chunk level, executed window (first fa)
        "run_rho_accel_full_vs_div": run_rho_accel_full,        # chunk level, whole chunk (all positions)
        "run_rho_straightness_vs_div": run_rho_straight,        # chunk level
        "run_rho_accel_vs_div_action": run_rho_accel_action,    # action level
        "n_pooled_chunks": n_pooled,
        "n_pooled_actions": n_pooled_a,
        # one-stop comparison of every free accel readout vs the SAME pooled divergence GT,
        # so the label run can pick the window/depth that best tracks the resample spread.
        "rho_vs_divergence": {
            "accel_exec": None if run_rho_accel != run_rho_accel else float(run_rho_accel),
            "accel_full": None if run_rho_accel_full != run_rho_accel_full else float(run_rho_accel_full),
            "straightness": None if run_rho_straight != run_rho_straight else float(run_rho_straight),
            "accel_action": None if run_rho_accel_action != run_rho_accel_action else float(run_rho_accel_action),
            "prefix_accel_by_depth": [None if r != r else float(r) for r in run_rho_prefix],
        },
        # prefix-accel ρ(t): one pooled rank-correlation per denoise depth (noise→t),
        # so the LAST entry == run_rho_accel_vs_div (full path). FM time DESCENDS
        # (prefix_fm_time[0] near noise → last near the clean action).
        "prefix_fm_time": [float(x) for x in prefix_time],
        "prefix_n_steps": n_prefix_steps,
        "run_rho_prefix_accel_vs_div": [None if r != r else float(r) for r in run_rho_prefix],
        # per-episode ρ averaged across episodes (kept for reference; not the same as run_rho_*).
        "mean_within_episode_rho_accel_vs_div": float(np.mean(_rho_a)) if _rho_a else None,
        "mean_within_episode_rho_straightness_vs_div": float(np.mean(_rho_s)) if _rho_s else None,
        "n_rollouts": len(episodes), "episodes": episodes,
        "outputs": {
            "npz": "chunk_geometry[_roN].npz", "summary_plot": "geometry[_roN].png",
            "scatter": "accel_divergence_scatter.png",
            "prefix_rho": "prefix_accel_rho.{png,npz}",
        },
    }
    write_json(out_dir / "meta.json", summary)
    rd.record_stage("chunk_geometry", {
        "n_rollouts": len(episodes), "n_pooled_chunks": n_pooled, "n_pooled_actions": n_pooled_a,
        "run_rho_accel_vs_div": run_rho_accel, "run_rho_accel_full_vs_div": run_rho_accel_full,
        "run_rho_accel_vs_div_action": run_rho_accel_action,
    })
    logger.info("chunk_geometry done -> %s | run ρ(accel,div) exec=%.3f full=%.3f (%d) | action=%.3f (%d)",
                out_dir, run_rho_accel, run_rho_accel_full, n_pooled, run_rho_accel_action, n_pooled_a)
    return rd, summary
