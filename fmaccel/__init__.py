"""fm-geometry: the geometry of flow-matching uncertainty, and ``accel``, its free proxy.

Reference implementation for *"The Geometric Nature and a Free Proxy for Flow-Matching
Uncertainty"*. The pipeline is: record the FM denoising trajectory during closed-loop eval
(:mod:`fmaccel.recording`) → read the free geometric scores off it (:mod:`fmaccel.geometry`) →
validate them against the Monte-Carlo resample posterior (:mod:`fmaccel.posterior`) → run them as
online failure detectors against the literature (:mod:`fmaccel.detectors`,
:mod:`fmaccel.detection`).

Top-level names are exposed LAZILY (PEP 562): importing ``fmaccel`` — or a light submodule like
``fmaccel.geometry.accel`` — does not pull in the lerobot-heavy policy path. That laziness is
load-bearing: every score in the paper is computed **post-hoc from the on-disk recording**, so the
whole analysis side stays importable (and runnable) in an environment with no policy installed at
all. Only ``get_model("pi05")`` imports lerobot.
"""

import importlib

# attribute name -> defining submodule (imported on first access only).
_LAZY = {
    # registry (extension point — add a model/dataset there, not here)
    "get_model": "fmaccel.registry",
    "get_dataset": "fmaccel.registry",
    "list_models": "fmaccel.registry",
    "list_datasets": "fmaccel.registry",
    # run-directory schema
    "RunDir": "fmaccel.core.runs",
    "create_run": "fmaccel.core.runs",
    "resolve_run": "fmaccel.core.runs",
    "list_runs": "fmaccel.core.runs",
    # recording (model-agnostic recorder + numpy loader)
    "FMRecorder": "fmaccel.recording.recorder",
    "FMRecording": "fmaccel.recording.loader",
    "RolloutRecord": "fmaccel.recording.loader",
    # the free geometric scores (Algorithm 1 / the discrete accel estimator)
    "score_chunks": "fmaccel.geometry.accel",
    "whole_chunk_curvature": "fmaccel.geometry.accel",
    "whole_chunk_prefix_accel": "fmaccel.geometry.accel",
    # resample posterior — the uncertainty ground truth accel is validated against
    "ChunkResampleSession": "fmaccel.posterior.resample",
    "build_resample_policy": "fmaccel.posterior.resample",
    "resample_to_npz": "fmaccel.posterior.resample",
    "chunk_posterior_metrics": "fmaccel.posterior.scatter",
    "standardization_stats": "fmaccel.posterior.scatter",
    "sweep_to_files": "fmaccel.posterior.scatter",
}


def __getattr__(name):  # PEP 562 module-level lazy attribute access
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = sorted(_LAZY)
