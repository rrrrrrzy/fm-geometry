"""``ace`` — FIPER's Action-Chunk Entropy leg (Römer et al., NeurIPS 2025, arXiv:2510.09459).

At each decision FIPER draws a batch of action chunks and measures the Shannon entropy of the
end-effector-position occupancy histogram, averaged over the prediction horizon — high entropy =
spread / multimodal intended behavior = uncertain. We compute it from the recorded MC-resample
endpoints (the $N$ resampled chunks at this decision): for each chunk position, bin the $N$
samples' position-subspace coordinates on a fixed-cell grid and take the entropy of the occupancy,
then average over the (executed) horizon. Higher = more spread = **more likely failure** (native
convention, no flip).

This is FIPER's training-free action leg and a natural **sibling of accel** (sample-spread entropy
vs velocity temporal bend). ⚠️ Keep $N$ and the grid distinct from the oracle's spread definition
so ACE does not silently reconstruct the resample GT (the ``oracle_resample_spread`` detector).

Grid (matched to FIPER ``entropy_eval.py``): **fixed cell size, adaptive limits**. ``fit`` calibrates
the cell as ``cellsize_factor · R_d`` per dim, where ``R_d`` = the per-dim working-area range over the
success set (FIPER eq.11, ``cellsize_factor`` default 0.03); each per-decision histogram is then anchored
at that decision's per-dim min (adaptive limits). If ``fit`` is not called, a single absolute ``cell``
scalar is used as a fallback. Shannon entropy is biased at small $N$, so prefer a larger resample $K$.

⚠️ ``pos_dims=(0,1,2)`` assumes the first 3 action dims are 3D Cartesian end-effector POSITION (FIPER's
``required_actions:[position]``). For this repo's policy (π₀.₅ on LIBERO) the action space is
joint/delta-EE, so ACE here measures occupancy spread in the first-3-action-dim subspace, NOT a working-
area EE grid — a faithful-in-form but semantically different quantity. Set ``pos_dims`` to the model's
actual position columns when known.
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector


def _occupancy_entropy(points: np.ndarray, cell) -> float:
    """Shannon entropy (base 2) of the occupancy histogram of ``points`` (N, D) on a grid with per-dim
    cell size ``cell`` (scalar or ``(D,)``), anchored at the per-decision per-dim min (FIPER's adaptive
    limits, fixed cell). Non-finite rows are dropped; 0 for <2 valid points or a single occupied cell."""
    pts = points[np.isfinite(points).all(axis=1)]  # drop NaN/inf resample rows (else float->int sentinel)
    if pts.shape[0] < 2:
        return 0.0
    cell = np.where(np.asarray(cell, np.float64) > 0, cell, 1.0)  # guard zero/degenerate cell
    idx = np.floor((pts - pts.min(axis=0)) / cell).astype(np.int64)  # adaptive origin = per-decision min
    _, counts = np.unique(idx, axis=0, return_counts=True)
    if counts.size < 2:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


class AceDetector(Detector):
    """Action-chunk entropy of the resample endpoints (higher = more spread = more likely failure)."""

    name = "ace"
    requires = frozenset({"resample"})
    online = False

    def __init__(self, *, pos_dims: tuple[int, ...] = (0, 1, 2), cell: float = 0.05,
                 cellsize_factor: float = 0.03, n_exec: int | None = None) -> None:
        self.pos_dims = tuple(int(d) for d in pos_dims)
        self.cell = float(cell)
        self.cellsize_factor = float(cellsize_factor)
        self.n_exec = n_exec
        self._cell_vec: np.ndarray | None = None  # per-dim calibrated cell (set by fit)

    def _sel_dims(self, act: int) -> list[int]:
        return [d for d in self.pos_dims if d < act] or list(range(min(3, act)))

    def fit(self, success_rollouts=None) -> None:
        """Calibrate the per-dim cell = ``cellsize_factor · R_d`` from the success rollouts' resample
        position coords (FIPER eq.11: ``R_d`` = per-dim working-area range). No-op (scalar ``cell``
        fallback) if no resample-bearing success rollouts are supplied."""
        if success_rollouts is None:
            return None
        rows: list[np.ndarray] = []
        for r in success_rollouts:
            R = r.resample if isinstance(r, ChunkRecord) else (r.get("resample") if isinstance(r, dict) else None)
            if R is None:
                continue
            R = np.asarray(R, np.float32)
            if R.ndim != 3:
                continue
            dims = self._sel_dims(R.shape[-1])
            rows.append(R[:, :, dims].reshape(-1, len(dims)))
        if not rows:
            return None
        P = np.concatenate(rows, axis=0)
        rng = P.max(axis=0) - P.min(axis=0)  # per-dim working-area range R_d
        fallback = float(rng.max()) if rng.max() > 0 else 1.0
        rng = np.where(rng > 0, rng, fallback)  # FIPER: zero ranges replaced by max range
        self._cell_vec = (self.cellsize_factor * rng).astype(np.float32)
        return None

    def score(self, rec: ChunkRecord) -> float:
        self.check(rec)
        R = np.asarray(rec.resample, np.float32)  # (N, chunk, act)
        if R.ndim != 3 or R.shape[0] < 2:
            return 0.0
        dims = self._sel_dims(R.shape[-1])
        pos = R[:, :, dims]  # (N, chunk, D)
        n_exec = rec.n_exec if rec.n_exec is not None else self.n_exec
        H = pos.shape[1] if n_exec is None else min(int(n_exec), pos.shape[1])
        cell = self._cell_vec if (self._cell_vec is not None and len(self._cell_vec) == len(dims)) else self.cell
        ent = [_occupancy_entropy(pos[:, h, :], cell) for h in range(H)]
        return float(np.mean(ent)) if ent else 0.0  # mean horizon entropy
