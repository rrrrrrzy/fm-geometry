"""Online failure detection: scoring, calibration, and the GPU capture stages.

* :mod:`fmaccel.detection.score` — run any detector in :mod:`fmaccel.detectors` over a recorded run
  and emit one score stream per episode. Numpy-only: it reads the recording off disk, so a cell
  scores in an environment with no policy installed.
* :mod:`fmaccel.detection.cusum` — turn score streams into alarms and evaluate them: the one-sided
  CUSUM recursion, the split-conformal calibration of the alarm height on held-out successful
  episodes, and the TPR-at-target-FPR / detection-lead metrics of Table 2.
* :mod:`fmaccel.detection.captures` — the three GPU stages that materialize what some detectors
  need but the recorder does not store (observation embeddings, action-expert hidden features, and
  the Diff-DAgger flow-matching loss).
"""
