"""Lazy model / dataset registries.

The central extension point of the refactored repo: adding a model or dataset is
"write one adapter class + add one line here". Names map to ``"module:Class"``
strings and the target module is imported **only when the name is resolved**
(:func:`get_model` / :func:`get_dataset`), never at package import time.

This laziness is load-bearing: the π₀.₅ adapter imports lerobot, but the whole analysis path
(``geometry``, ``detectors``, ``detection``, ``recording``) must stay importable **without
lerobot** — every score in the paper is computed post-hoc from the on-disk recording, not from a
live policy. Because the tables below are plain string literals, ``import fmaccel.registry`` and
:func:`list_models` touch nothing heavy; only ``get_model("pi05")`` pulls lerobot.

This release ships the **π₀.₅ / LIBERO reference path** only. The paper additionally evaluates
SmolVLA, GR00T N1.7 and VLA-JEPA on LIBERO / RoboCasa / D3IL; those adapters are not included
here. Adding one is a single adapter class plus one line in the tables below — the recording
format, the ``accel`` estimator and every detector are model-agnostic.
"""

from __future__ import annotations

import importlib
from typing import Any

# name -> "package.module:ClassName" (imported lazily on resolve)
_MODELS: dict[str, str] = {
    "pi05": "fmaccel.models.pi05:Pi05Adapter",           # π₀.₅ (PaliGemma + Gemma action expert) — imports lerobot
}

_DATASETS: dict[str, str] = {
    "libero": "fmaccel.datasets.libero:LiberoDataset",   # closed-loop vec-env rollouts
}


def _resolve(spec: str) -> Any:
    module_path, _, cls_name = spec.partition(":")
    if not cls_name:
        raise ValueError(f"registry spec must be 'module:Class', got {spec!r}")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def get_model(name: str) -> Any:
    """Return the model-adapter class registered under ``name`` (imports it now)."""
    try:
        spec = _MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {list_models()}") from None
    return _resolve(spec)


def get_dataset(name: str) -> Any:
    """Return the dataset-adapter class registered under ``name`` (imports it now)."""
    try:
        spec = _DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {list_datasets()}") from None
    return _resolve(spec)


def list_models() -> list[str]:
    """Registered model names (no imports — safe without the policy installed)."""
    return sorted(_MODELS)


def list_datasets() -> list[str]:
    """Registered dataset names (no imports — safe without the policy installed)."""
    return sorted(_DATASETS)


def register_model(name: str, spec: str) -> None:
    """Programmatically register a model adapter (e.g. from a plugin/notebook)."""
    _MODELS[name] = spec


def register_dataset(name: str, spec: str) -> None:
    """Programmatically register a dataset adapter."""
    _DATASETS[name] = spec
