"""Path + .env bootstrap for thin scripts run as ``python scripts/X.py``.

Insert the repo root onto ``sys.path`` (so ``import fmaccel`` works without an
editable install), then load ``.env``. Import this first in every script::

    import _bootstrap  # noqa: F401
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmaccel.core.bootstrap import bootstrap  # noqa: E402

bootstrap()
