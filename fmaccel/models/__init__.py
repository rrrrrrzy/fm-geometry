"""Model adapters: the "add a model = one file" extension point.

Each adapter wraps a flow-matching policy behind :class:`FMModelAdapter`, telling
the recorder/resampler exactly what they need (dims, how to intercept the FM
head). Resolve adapters by registry name via
:func:`fmaccel.registry.get_model` — never import ``pi05`` here, so this
package stays importable without lerobot installed.
"""

from fmaccel.models.base import FMConfig, FMModelAdapter

__all__ = ["FMConfig", "FMModelAdapter"]
