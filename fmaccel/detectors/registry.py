"""Lazy failure-detector registry — the extension point for baselines.

Mirrors :mod:`fmaccel.registry` (models / datasets): a detector is "write one
:class:`~fmaccel.detectors.base.Detector` subclass + add one line here". Names map to
``"module:Class"`` strings imported **only when resolved** (:func:`get_detector`), so
``import fmaccel.detectors`` and :func:`list_detectors` stay lerobot-free and cheap —
a baseline that pulls torch / sklearn only costs that import when you actually build it.
"""

from __future__ import annotations

import importlib
from typing import Any

# name -> "package.module:ClassName" (imported lazily on resolve).
# accel is the free reference detector; literature baselines land here as they are built
# (build order + design in docs/baselines.md):
_DETECTORS: dict[str, str] = {
    "accel": "fmaccel.detectors.accel:AccelDetector",  # free temporal-bend proxy — the method under test
    # --- Phase 1: free, online, training-free comparators (read the recorded denoise path) ---
    "straightness": "fmaccel.detectors.straightness:StraightnessDetector",  # chord/arc (saturating geometric sibling of accel)
    "sparc": "fmaccel.detectors.sparc:SparcDetector",                        # spectral arc length of executed-action jerk
    # --- Phase 2: resample-based (need the K-sample posterior, scored post-hoc from chunk_divergence) ---
    "ace": "fmaccel.detectors.ace:AceDetector",                              # FIPER action-chunk entropy leg
    "stac": "fmaccel.detectors.stac:StacDetector",                           # Sentinel/FAIL-Detect temporal action-chunk MMD
    "oracle_resample_spread": "fmaccel.detectors.oracle:OracleResampleSpread",  # GT upper bound — NOT a baseline
    # --- Phase 3: loss family (model-in-the-loop; velocity field injected via rec.extra) ---
    "fm_loss": "fmaccel.detectors.fm_loss:FmLossDetector",                   # FM re-noise loss = Diff-DAgger score
    # --- Phase 4: embedding-OOD density (need the obs-embedding capture hook; fit on success) ---
    "rnd_oe": "fmaccel.detectors.rnd:RndOeDetector",                         # FIPER OOD leg / FAIL-Detect RND (torch)
    "logpzo": "fmaccel.detectors.logpzo:LogpZoDetector",                     # FAIL-Detect best score (torch CFM density)
    "pca_kmeans": "fmaccel.detectors.density:PcaKmeansDetector",            # classical density (numpy)
    "knn": "fmaccel.detectors.density:KnnDetector",                          # kNN embedding OOD (numpy)
    "mahalanobis": "fmaccel.detectors.density:MahalanobisDetector",          # Mahalanobis OOD (numpy)
    # --- Phase 5: assembled ---
    "fiper": "fmaccel.detectors.fiper:FiperDetector",                        # AND(rnd_oe, ace) + per-leg conformal
    # --- Phase 6: supervised probe (category-C control; needs FAILURE labels + hidden hook) ---
    "safe": "fmaccel.detectors.safe:SafeDetector",                           # SAFE last-hidden-state failure probe (MLP/LSTM)
}


def _resolve(spec: str) -> Any:
    module_path, _, cls_name = spec.partition(":")
    if not cls_name:
        raise ValueError(f"registry spec must be 'module:Class', got {spec!r}")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def get_detector(name: str) -> Any:
    """Return the detector class registered under ``name`` (imports its module now)."""
    try:
        spec = _DETECTORS[name]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; available: {list_detectors()}") from None
    return _resolve(spec)


def list_detectors() -> list[str]:
    """Registered detector names (no imports — safe without lerobot installed)."""
    return sorted(_DETECTORS)


def register_detector(name: str, spec: str) -> None:
    """Programmatically register a detector (e.g. from a notebook/plugin)."""
    _DETECTORS[name] = spec
