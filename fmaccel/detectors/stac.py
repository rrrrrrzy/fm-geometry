"""``stac`` — Statistical Temporal Action Consistency (Sentinel; a score in FAIL-Detect).

STAC (Agia et al., Sentinel, CoRL 2024, arXiv:2410.04640; reused as a score in FAIL-Detect,
Xu et al. RSS 2025, arXiv:2503.08558) measures whether the policy's *action-chunk distribution*
is consistent across consecutive re-plan decisions. At decision $t$ it draws a batch of MC action
chunks and, comparing them to the previous decision's chunks over the **overlapping future
timesteps**, takes the MMD (RBF kernel, biased V-statistic, median-heuristic bandwidth) between
the two sample sets. A large MMD = the plan distribution shifted = inconsistent = **more likely
failure** (native polarity, no flip). Sentinel then accumulates the per-decision MMDs into a
cumulative statistic $\\eta_t=\\sum_{i\\le t}\\mathrm{MMD}_i$ and thresholds it with a split-conformal
band calibrated on SUCCESS rollouts — that cumulative-sum + threshold is the deployment layer and
lives in the shared harness (accumulation + CUSUM/conformal), so this detector returns the **raw
per-decision MMD**.

Action-posterior-only (never a hidden state / embedding) and model-agnostic, so it maps cleanly to
a flow-matching VLA: our per-decision MC-resample set (``rec.resample``) IS Sentinel's "batch of B
sampled action sequences". Distinct from accel (within-chunk denoise bend) and from the resample
oracle (within-chunk spread): STAC is the *cross-time* drift of the resample posterior.

FAITHFULNESS (audited against ``agiachris/sentinel`` ``error_utils.py`` + ``CXU-TRI/FAIL-Detect``
``eval_load_baseline.py``):
* **Overlap window (Sentinel ``compute_temporal_error``):** with exec horizon $k$ = ``n_exec`` and
  chunk length $h$ = ``chunk``, the previous chunk's TAIL ``prev[:, k:h]`` (its continuation past the
  executed prefix) is compared to the current chunk's HEAD ``curr[:, :h-k]`` — the SAME absolute
  future timesteps. Getting the direction backwards compares non-overlapping times.
* **Biased V-statistic** ``mean(Kxx)+mean(Kyy)−2·mean(Kxy)`` with the diagonal *included* (what both
  Sentinel ``compute_mmd_rbf`` and FAIL-Detect ``STAC_UQ`` return; the unbiased U-statistic would not
  reproduce them).
* **Median-heuristic bandwidth** ``γ = 1/(2·median(pooled SQUARED pairwise distances, zeros
  excluded))`` over the pooled ``[X;Y]`` set → kernel ``exp(−‖·‖²/(2·median²))``. sklearn's
  ``rbf_kernel`` is ``exp(−γ‖·‖²)`` with **no** ``/2``, so the ``/2`` lives in ``γ`` — do not divide
  again. ⚠️ Sentinel uses median-of-**squared** distances; FAIL-Detect's ``median_trick_bandwidth``
  uses ``2·(median of RAW distances)²`` — numerically different (``median(d²) ≠ (median d)²``). We
  follow **Sentinel** (median of squared) for the paper claim.
* **Binary gripper dropped (Sentinel ``filter_gripper_action``):** the discrete gripper command
  biases the MMD with jumps, so Sentinel omits it before the distance (rotation is KEPT — the
  starred ``STAC MMD*`` default). Pass ``drop_dims`` = the gripper column(s) for this policy
  (π₀.₅/LIBERO gripper is the LAST action dim). Default ``drop_dims=None`` keeps every dim.

DEVIATIONS / gotchas:
* 🐛 **FIXED 2026-07-27 — the overlap was being truncated away upstream.** ``chunk_divergence``
  writes ``first_actions = n_action_steps`` and ``chunk_geometry._load_div_chunks`` sliced
  ``chunks[:, :, :first_actions]``, which is right for accel/divergence (they want the *executed*
  plan's spread) but left this detector with ``chunk == n_exec`` on EVERY model — so ``overlap``
  was always 0 and the no-overlap fallback below always fired, comparing two DISJOINT absolute
  time spans (``t…t+k`` vs ``t−k…t``). That measures "the robot moved", not "the plan changed".
  Fix: require ``resample_full``/``prev_resample_full`` (the untruncated horizon) instead of the
  executed-window pair. Effect at k=8, online CUSUM TPR@FPR=0.1: on every configuration with
  ``chunk > n_exec`` (i.e. where a real plan overlap exists) the detector goes from near-chance to
  strongly discriminative — π₀.₅/LIBERO 0.20→0.93 is representative. Restricting the MMD to the
  genuinely overlapping future window is what makes STAC work at all in that regime. Configurations
  whose NATIVE ``chunk == n_exec`` (no overlap to recover) were never affected and are unchanged.
* When ``chunk == n_exec`` (no plan overlap) Sentinel asserts $h-k>0$ and has
  no in-paper fallback (its own remedy is to reduce $k$). We fall back to the MMD between the two
  FULL chunk sample sets — a pure cross-time distribution-drift signal. This is what FAIL-Detect's
  ``STAC_UQ`` does (full-chunk flatten, no overlap slice), so it is **FAIL-Detect-style, not
  Sentinel-faithful** — flagged, not silent.
* $N$ here is the MC-resample count (typ. 8–64) vs Sentinel's $B=256$; the biased V-stat has
  $O(1/N)$ positive bias, so keep $N$ FIXED across calibration and test (the median heuristic
  self-normalizes and the bias cancels in the conformal quantile). Degenerate ($<2$ samples per set)
  → ``0.0`` (the most-confident end); first decision (no ``prev_resample``) → ``0.0`` (matches
  FAIL-Detect's zero at $t=0$).
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise SQUARED Euclidean distances between rows of ``A`` (n,d) and ``B`` (m,d)."""
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)


def mmd2_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    """Squared MMD between sample sets ``X`` (n,d), ``Y`` (m,d), RBF kernel — Sentinel-faithful.

    * **BIASED V-statistic** ``mean(Kxx)+mean(Kyy)−2·mean(Kxy)`` (diagonal *included*), matching
      Sentinel ``error_utils.compute_mmd_rbf`` and FAIL-Detect ``STAC_UQ``.
    * **Median heuristic** ``γ = 1/(2·median(pooled SQUARED pairwise distances > 0))`` so the kernel
      is ``exp(−‖·‖²/(2·median²))`` (Sentinel's ``gamma='median'``: median of SQUARED distances, not
      squared-of-median). The ``/2`` lives in ``γ`` because sklearn's ``rbf_kernel`` has none.

    Returns ``0.0`` for degenerate inputs (``<2`` samples per set)."""
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return 0.0
    Kxx, Kyy, Kxy = _sq_dists(X, X), _sq_dists(Y, Y), _sq_dists(X, Y)
    pooled = np.concatenate([Kxx[np.triu_indices(n, 1)], Kyy[np.triu_indices(m, 1)], Kxy.ravel()])
    pos = pooled[pooled > 0]
    med = float(np.median(pos)) if pos.size else 1.0  # median of SQUARED pooled distances (Sentinel)
    gamma = 1.0 / (2.0 * med + 1e-12)                 # exp(-||.||^2 / (2*median^2)) — median heuristic
    kxx, kyy, kxy = np.exp(-gamma * Kxx), np.exp(-gamma * Kyy), np.exp(-gamma * Kxy)
    mmd2 = kxx.mean() + kyy.mean() - 2.0 * kxy.mean()  # BIASED V-statistic (diagonal included), as in source
    return float(max(mmd2, 0.0))


class StacDetector(Detector):
    """Temporal MMD between consecutive decisions' resample sets (higher = inconsistent = failure)."""

    name = "stac"
    # FULL-horizon sets, not the executed-window `resample` pair: the overlap Sentinel compares
    # (`cur[:, :chunk-n_exec]` vs `prev[:, n_exec:]`) lies entirely PAST the executed window, so
    # the executed-window sets would leave overlap=0 and silently force the no-overlap fallback
    # onto EVERY model — comparing two disjoint absolute time spans. See ChunkRecord's docstring.
    requires = frozenset({"resample_full", "prev_resample_full"})
    online = False

    def __init__(self, *, n_exec: int | None = None, drop_dims: tuple[int, ...] | None = None) -> None:
        self.n_exec = n_exec
        # Action columns to omit before the distance (Sentinel drops the binary gripper). None keeps all.
        self.drop_dims = None if drop_dims is None else tuple(int(d) for d in drop_dims)

    def _keep_dims(self, act: int) -> np.ndarray:
        idx = np.arange(act)
        if self.drop_dims:
            idx = np.array([d for d in idx if d not in set(self.drop_dims)], dtype=np.int64)
        return idx

    def score(self, rec: ChunkRecord) -> float:
        self.check(rec)
        Rt = np.asarray(rec.resample_full, np.float32)[..., : rec.action_dim]       # (N, chunk, act)
        Rp = np.asarray(rec.prev_resample_full, np.float32)[..., : rec.action_dim]  # (M, chunk, act)
        if Rt.ndim != 3 or Rp.ndim != 3 or Rt.shape[0] < 2 or Rp.shape[0] < 2:
            return 0.0
        keep = self._keep_dims(Rt.shape[-1])          # drop gripper column(s) before the distance
        if keep.size == 0:
            return 0.0
        Rt, Rp = Rt[:, :, keep], Rp[:, :, keep]
        chunk = Rt.shape[1]
        exec_w = int(rec.n_exec if rec.n_exec is not None else (self.n_exec if self.n_exec is not None else chunk))
        overlap = chunk - exec_w
        if overlap >= 1:                       # current HEAD vs previous TAIL (same absolute future steps)
            cur, prev = Rt[:, :overlap, :], Rp[:, exec_w:, :]
        else:                                  # chunk==exec: no plan overlap -> full-chunk distribution drift
            cur, prev = Rt, Rp                 # (FAIL-Detect-style fallback; flagged non-Sentinel-faithful)
        X = cur.reshape(cur.shape[0], -1)
        Y = prev.reshape(prev.shape[0], -1)
        return mmd2_rbf(X, Y)                   # biased V-stat MMD² — higher = inconsistent = failure
