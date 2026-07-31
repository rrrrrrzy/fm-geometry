"""``fiper`` — the assembled FIPER detector (Römer et al., NeurIPS 2025, arXiv:2510.09459).

The published detector is the **conjunction (AND)** of two failure-data-free legs, each conformally
calibrated on SUCCESS-only rollouts: the OOD leg ``rnd_oe`` (RND on the obs embedding) and the
action leg ``ace`` (action-chunk entropy of the resample posterior). We normalize each leg's score
by its success-calibrated threshold $\\tau$ ($\\alpha$-quantile of success scores) and take the
element-wise **min** of the two normalized scores — an alarm fires only when *both* legs exceed
their threshold, which suppresses false alarms on benign OOD. Higher = both legs high = **more likely
failure** (native); reporting AND (not a single leg) is the faithful FIPER.

For the shared harness's AUROC the per-chunk score is this normalized min (no $\\alpha$-tuned binary,
so the comparison stays threshold-free). Needs both legs' signals: ``obs_emb`` (the obs-embedding
hand-off, see ``rnd_oe``) and ``resample`` (the K candidates, see ``ace``). ``fit`` trains RND-OE,
calibrates the ACE per-dim cell from the success working-area ranges (FIPER eq.11), and sets both
per-leg success-$\\alpha$ thresholds (floored to a positive leg-scale) on the same success rollouts.
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector


class FiperDetector(Detector):
    """FIPER = AND(RND-OE, ACE), per-leg conformally normalized (higher = more likely failure)."""

    name = "fiper"
    requires = frozenset({"obs_emb", "resample"})
    online = False

    def __init__(self, *, alpha: float = 0.95, ace_cell: float = 0.05,
                 ace_pos_dims: tuple[int, ...] = (0, 1, 2), n_exec: int | None = None,
                 **rnd_kwargs) -> None:
        from fmaccel.detectors.ace import AceDetector
        from fmaccel.detectors.rnd import RndOeDetector

        self.alpha = float(alpha)
        self.rnd = RndOeDetector(**rnd_kwargs)
        self.ace = AceDetector(pos_dims=ace_pos_dims, cell=ace_cell, n_exec=n_exec)
        self._tau_rnd: float | None = None
        self._tau_ace: float | None = None

    def fit(self, success_rollouts) -> None:
        recs = list(success_rollouts)
        self.rnd.fit(recs)  # train RND-OE predictor on success obs embeddings (stack_signal pulls obs_emb)
        self.ace.fit(recs)  # calibrate ACE per-dim cell from success working-area ranges (FIPER eq.11)
        s_rnd = np.array([self.rnd.score(r) for r in recs], float)
        s_ace = np.array([self.ace.score(r) for r in recs], float)
        self._tau_rnd = self._pos_threshold(s_rnd)
        self._tau_ace = self._pos_threshold(s_ace)

    def _pos_threshold(self, s: np.ndarray) -> float:
        """alpha-quantile success threshold, floored to a POSITIVE scale tied to the leg's magnitude.
        The ACE leg is frequently exactly 0 on success chunks (single occupied cell), so a bare quantile
        can be 0; falling back to a fixed 1.0 would discontinuously rescale that leg, so use max(s)."""
        tau = float(np.quantile(s, self.alpha))
        if tau > 0:
            return tau
        mx = float(np.max(s)) if s.size else 0.0
        return mx if mx > 0 else 1.0

    def score(self, rec: ChunkRecord) -> float:
        self.check(rec)
        if self._tau_rnd is None:
            raise RuntimeError("fiper.score before fit(); call fit(success rollouts) first")
        n_rnd = self.rnd.score(rec) / (self._tau_rnd + 1e-12)
        n_ace = self.ace.score(rec) / (self._tau_ace + 1e-12)
        return float(min(n_rnd, n_ace))  # AND: both legs must exceed their threshold
