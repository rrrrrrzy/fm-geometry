"""The Monte-Carlo resample posterior — the uncertainty ground truth.

Fix the observation, resample ``K`` action chunks with different noise seeds, and measure how far
apart they land. That spread is what the paper treats as the ground-truth action-posterior
uncertainty, and what the free ``accel`` proxy is validated against (Spearman ρ, Table 1).

* :mod:`fmaccel.posterior.resample` — re-run the policy's FM head on a recorded decision, rebuilding
  the *exact* recorded conditioning from the ``--record-context`` sidecar. Gated on bitwise
  reproduction of the recorded chunk: if the replayed conditioning does not reproduce, the
  resampled posterior is not the one the policy actually faced.
* :mod:`fmaccel.posterior.divergence` — the per-decision divergence of ``K`` resampled chunks along
  a whole episode (the ``D_resample`` column of Table 1).
* :mod:`fmaccel.posterior.scatter` — the dense single-decision posterior sweep (multimodality and
  Gaussianity diagnostics).

``resample``/``divergence`` need the policy on a GPU; the consumers of what they write do not.
"""
