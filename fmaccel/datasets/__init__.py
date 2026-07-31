"""Dataset / environment adapters: the "add a dataset = one file" extension point.

Each adapter abstracts how a benchmark is run behind :class:`DatasetAdapter`,
hiding whether it is a lerobot vectorized env (LIBERO), an out-of-process sim
driven over a socket bridge, or a teacher-forced state stream. Resolve by registry
name via :func:`fmaccel.registry.get_dataset`; the lerobot-importing ``libero``
adapter is never imported here, so this package loads without lerobot too.
"""

from fmaccel.datasets.base import DatasetAdapter, TaskSpec

__all__ = ["DatasetAdapter", "TaskSpec"]
