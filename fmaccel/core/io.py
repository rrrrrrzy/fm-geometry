"""Small JSON / npz IO helpers used across pipelines (lerobot-free).

Centralizes the read/write idioms (mkdir-parents, indent=2, compressed npz,
``allow_pickle`` for object arrays like task descriptions) so individual
pipelines stop re-implementing them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path | str, obj: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False, default=_json_default))
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def save_npz(path: Path | str, *, compressed: bool = True, **arrays: Any) -> Path:
    """Save named arrays to ``path`` (mkdir parents). Object arrays (e.g. task
    description strings) are stored as-is via numpy's pickle support."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if compressed else np.savez
    saver(path, **arrays)
    return path


def load_npz(path: Path | str) -> dict[str, np.ndarray]:
    """Load an npz eagerly into a plain dict (closes the file handle)."""
    with np.load(Path(path), allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
