"""``oracle_resample_spread`` — the GROUND-TRUTH uncertainty, as the ORACLE upper bound.

The Monte-Carlo resample posterior spread (pairwise distance of the $N$ resampled action
chunks at a decision) **is the project's uncertainty ground truth** — accel is *validated
against* it ($\\rho\\approx0.86$). It is therefore **NOT a baseline**: scoring it as a competing
detector double-counts the GT and wins by construction. This class exists only so the shared
harness can draw it as the **oracle / upper-bound reference line** in `compare_detectors.png`;
the registry name is deliberately verbose and it must never be presented as a competing method.

Reuses :func:`fmaccel.geometry.accel.whole_chunk_divergence` (the exact spread the
chunk-geometry leg validates accel against). To make the oracle **equal** that GT, the per-dim
distance scale must be the SAME run-pooled ``dim_scale`` chunk-geometry uses for every chunk (so the
spread *magnitude* is comparable across decisions). Supply it as ``scale_act`` or via
``rec.extra['dim_scale']`` (the post-hoc :mod:`fmaccel.detection.score` surfaces the
run-pooled scale into ``rec.extra``). ⚠️ **Do NOT** per-chunk self-normalize: dividing by *this*
decision's own resample std makes the score scale-invariant and erases the across-chunk spread
magnitude that IS the uncertainty — so the ``None`` fallback here is the RAW flattened distance, never
a per-chunk self-std.
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector


class OracleResampleSpread(Detector):
    """Resample posterior spread = uncertainty GT. ORACLE / upper bound, **not a baseline**."""

    name = "oracle_resample_spread"
    requires = frozenset({"resample"})
    online = False

    def __init__(self, *, stat: str = "max", scale_act: np.ndarray | None = None,
                 n_exec: int | None = None) -> None:
        if stat not in ("max", "mean"):
            raise ValueError(f"stat must be max/mean, got {stat!r}")
        self.stat = stat
        self.scale_act = None if scale_act is None else np.asarray(scale_act, np.float32)
        self.n_exec = n_exec

    def score(self, rec: ChunkRecord) -> float:
        from fmaccel.geometry.accel import whole_chunk_divergence

        self.check(rec)
        R = np.asarray(rec.resample, np.float32)[..., : rec.action_dim]  # (N, chunk, act)
        if R.ndim != 3 or R.shape[0] < 2:
            return 0.0
        n_exec = rec.n_exec if rec.n_exec is not None else self.n_exec
        if n_exec is not None:
            R = R[:, : int(n_exec), :]
        # Use the SAME run-pooled per-dim scale the GT uses (constructor arg, else rec.extra['dim_scale']);
        # fall back to RAW distance (scale_act=None) — NEVER a per-chunk self-std, which would divide out
        # the across-chunk spread magnitude that is the uncertainty ground truth.
        scale = self.scale_act
        if scale is None:
            dim_scale = (rec.extra or {}).get("dim_scale")
            scale = None if dim_scale is None else np.asarray(dim_scale, np.float32)[: rec.action_dim]
        mx, mn = whole_chunk_divergence(R[None], scale_act=scale)  # each (1,)
        return float(mx[0] if self.stat == "max" else mn[0])  # higher spread = more uncertain
