"""``.env`` loading + repo-root path setup, shared by every ``cli/`` and ``experiments/`` entry.

A thin script's first line is ``import _bootstrap``, which prepends the repo root to
``sys.path`` (so ``import fmaccel`` works when running by path without an editable install) and
then calls :func:`bootstrap` here. Idempotent. With ``pip install -e .`` neither step is needed,
but both stay harmless.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"

_DONE = False


def ensure_repo_on_path() -> None:
    """Prepend the repo root to ``sys.path`` so ``import fmaccel`` works
    for scripts run by path without an editable install."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_env(path: Path = ENV_FILE) -> None:
    """Load ``<repo>/.env`` into ``os.environ`` without overriding existing
    values (so ``FOO=bar python scripts/...`` still wins). No ``${...}``
    expansion; surrounding quotes are stripped."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def bootstrap() -> None:
    """Idempotent: ensure repo on path + .env loaded."""
    global _DONE
    if _DONE:
        return
    ensure_repo_on_path()
    load_env()
    _DONE = True
