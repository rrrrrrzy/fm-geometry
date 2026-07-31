"""FM-head trajectory recording: the model-agnostic recorder, its on-disk format,
and a numpy-only loader.

``recorder.FMRecorder`` is driven by a :class:`fmaccel.models.base.FMModelAdapter`
— it owns the generic policy/env hooks + serialization, and delegates the only
model-specific part (how to intercept the FM head's noise and per-step
``(t, x_t, v_t)``) to the adapter. ``loader`` reads recordings with numpy only,
so case-study notebooks can inspect them without lerobot installed.
"""

from fmaccel.recording.loader import FMRecording, RolloutRecord
from fmaccel.recording.recorder import FMRecorder

__all__ = ["FMRecorder", "FMRecording", "RolloutRecord"]
