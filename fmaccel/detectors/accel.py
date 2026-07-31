"""``accel`` — the reference detector (the method under test).

This is a thin :class:`~fmaccel.detectors.base.Detector` wrapper over
:func:`fmaccel.geometry.accel.score_chunks`, so the failure-detection
signal is the per-chunk ``accel`` the closed-loop server records as ``chunk_accels`` — no
second implementation to drift. Free, online, training-free, pure numpy. (It is
**bit-for-bit** identical to the recorded ``chunk_accels`` only when ``mode`` / ``fixed_std``
/ ``n_exec`` match the server's recording config; with different settings the streams differ.)

``accel`` (``Σ_t‖Δv_{t+1}−Δv_t‖ / ⟨‖Δv_t‖⟩`` of the per-dim z-scored denoise path) is in
the score convention *natively*: a straighter denoise → smaller accel → lower posterior
spread → more confident, so ``score`` needs no sign flip. ``mode`` selects the same accel
variant family the labels were built with (default ``accel_prefix:7`` = executed-window accel
read at denoise depth 9, the variant used for the validated closed-loop detector).

⚠️ SCALE (load-bearing): with the **default ``fixed_std=None``** ``score`` self-normalizes
**each chunk** by its own denoise-path std (it feeds one chunk as a ``(1, T+1, …)`` batch), so
scores are NOT on the offline per-episode-/run-pooled label scale and will **not** reproduce
the validated AUROC (the repo's ``accel_normalization_scale`` note measured corr ≈ 0.30
between the two scales). Pass ``fixed_std`` (the demo-distribution per-dim std, ``action_dim`` /
window matched) to land on the label scale.
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector


class AccelDetector(Detector):
    """Free temporal-bend proxy ``accel`` as a per-chunk failure score (lower = confident)."""

    name = "accel"
    requires = frozenset({"x_t"})
    online = True

    def __init__(
        self,
        mode: str = "accel_prefix:7",
        *,
        fixed_std: np.ndarray | None = None,
        n_exec: int | None = None,
    ) -> None:
        self.mode = mode
        self.fixed_std = None if fixed_std is None else np.asarray(fixed_std, np.float32)
        self.n_exec = n_exec

    def fit(self, success_rollouts=None) -> None:
        """Training-free. The demo-scale ``fixed_std`` (if any) is supplied at construction."""
        return None

    def score(self, rec: ChunkRecord) -> float:
        # Imported lazily so the registry/base stay import-cheap; score_chunks itself is
        # pure numpy (no torch / lerobot), preserving lerobot-free portability.
        from fmaccel.geometry.accel import score_chunks

        self.check(rec)
        x = np.asarray(rec.x_t, np.float32)[None]  # (1, T+1, chunk, act)
        n_exec = rec.n_exec if rec.n_exec is not None else self.n_exec
        # exec/prefix modes restrict to the executed window; with n_exec unknown score_chunks would
        # silently fall back to the WHOLE chunk (a different, undocumented signal) — flag it instead.
        if n_exec is None and self.mode.startswith(("accel_exec", "accel_prefix")):
            raise ValueError(
                f"accel mode {self.mode!r} is an executed-window variant but n_exec is unknown "
                f"(rec.n_exec and detector n_exec both None); pass n_exec so the executed window is "
                f"well-defined instead of silently scoring the whole chunk."
            )
        accel = score_chunks(x, rec.action_dim, n_exec=n_exec, mode=self.mode, fixed_std=self.fixed_std)
        return float(accel[0])
