"""Shared core utilities for the CLI entry points and pipelines.

Everything here is lerobot-free (stdlib + numpy only), so the whole ``core``
package stays importable without lerobot installed. Heavy, model/dataset-specific
work lives in :mod:`fmaccel.models`, :mod:`fmaccel.datasets`, and
:mod:`fmaccel.pipelines`; this package only provides the plumbing they share:
argparse fragments, the run-directory schema, .env bootstrap, npz/json IO, and
multi-GPU shard math.
"""
