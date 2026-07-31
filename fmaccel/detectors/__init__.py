"""Failure-detection detectors: one carrier + one protocol + a lazy registry.

The free ``accel`` reference and every literature baseline implement the same
:class:`~fmaccel.detectors.base.Detector` interface and are resolved by name, so they
all flow through one evaluation harness
(:mod:`fmaccel.detection.cusum`). See ``docs/baselines.md`` for
the design, the shortlist, and the build order.

Importing this package is lerobot-free (numpy-only base + string-table registry); a
detector's heavy deps (torch / sklearn) load only when that detector is resolved.
"""

from __future__ import annotations

from fmaccel.detectors.base import REQUIREABLE, ChunkRecord, Detector
from fmaccel.detectors.registry import (
    get_detector,
    list_detectors,
    register_detector,
)

__all__ = [
    "ChunkRecord",
    "Detector",
    "REQUIREABLE",
    "get_detector",
    "list_detectors",
    "register_detector",
]
