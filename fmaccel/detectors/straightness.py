"""``straightness`` — chord/arc ratio of the denoise path, a free online baseline.

A geometric **sibling** of accel: both read the recorded denoise trajectory and quantify
how far it bends from a straight noise→action shot. ``straightness`` = $\\lVert x_T-x_0\\rVert
/ \\sum_t\\lVert\\Delta x_t\\rVert$ of the per-dim z-scored flattened chunk path (1.0 = perfectly
straight = confident); we return ``1 − straightness`` so the score follows the accel
convention (**higher = curvier = more likely failure**). Reuses
:func:`fmaccel.geometry.accel.whole_chunk_straightness` — numpy, free, online.

NOTE for the paper: under plain CFM, straightness **saturates** (~97% of chunks in
[0.99, 1.0]); it still correlates with the resample-GT divergence ($\\rho\\approx-0.7$) but can
barely *resolve* chunks, so this baseline is **expected to underperform accel** — which is
precisely the documented reason accel (the non-saturating curvature) exists. Report it as the
saturating geometric comparator, explicitly near-equivalent to (not novel over) accel.
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector


class StraightnessDetector(Detector):
    """Chord/arc straightness of the denoise path as a per-chunk failure score."""

    name = "straightness"
    requires = frozenset({"x_t"})
    online = True

    def __init__(self, *, n_exec: int | None = None) -> None:
        self.n_exec = n_exec

    def score(self, rec: ChunkRecord) -> float:
        from fmaccel.geometry.accel import whole_chunk_straightness

        self.check(rec)
        x = np.asarray(rec.x_t, np.float32)[None]  # (1, T+1, chunk, act)
        n_exec = rec.n_exec if rec.n_exec is not None else self.n_exec
        if n_exec is not None:
            x = x[:, :, : int(n_exec), :]
        s = float(whole_chunk_straightness(x, rec.action_dim)[0])  # (0,1], 1=straight=confident
        return 1.0 - s  # accel convention: higher = curvier = more likely failure
