"""The free geometric readouts of the recorded denoising trajectory.

Everything here is **free**: it reads the Euler iterates the policy already produced in its normal
forward pass, so it costs no extra model evaluations, no training and no resampling.

* :mod:`fmaccel.geometry.accel` — ``accel``, the paper's denoising acceleration: the normalized
  total variation of the denoising velocity (Algorithm 1). Includes the prefix variant
  ``accel_p`` over the first ``p`` of ``T`` Euler steps, and :func:`~fmaccel.geometry.accel.score_chunks`,
  the named-mode entry point the detector uses.
* :mod:`fmaccel.geometry.straightness` — ``Straightness``, the chord-to-arc ratio of the denoising
  path; the saturating geometric sibling ``accel`` is compared against.
* :mod:`fmaccel.geometry.profile` — where along the denoise the bending happens (the per-step
  accel profile), used for the prefix-depth analysis.

This subpackage is numpy-only and imports no policy code.
"""
