"""FM recording on-disk format v3 (the data contract).

Clean break from the legacy v2 layout (all rollouts of a task packed into one npz
with ``r<k>_``-prefixed keys). v3 is **one npz per rollout** with plain array
names, indexed by a typed ``manifest.json``::

    fm/
      manifest.json
      rollouts/
        <prefix>rollout_0000.npz           # plain keys (see ROLLOUT_KEYS)
        <prefix>rollout_0000_context.npz   # optional (see CONTEXT_KEYS)

Why: each rollout file is self-contained, so multi-GPU recording writes are
embarrassingly parallel (each worker emits its own ``<prefix>rollout_*.npz`` and
the merged manifest is just a concatenated rollout list — no rollout-id
renumbering hack), and a reader can mmap one rollout without touching the rest.

manifest.json schema (version 3)::

    {
      "version": 3,
      "dims": {n_action_steps, chunk_size, max_action_dim, action_dim,
               num_inference_steps, dt},
      "has_context": bool,
      "rollouts": [
        {"file": "rollouts/rollout_0000.npz",
         "context_file": "rollouts/rollout_0000_context.npz" | null,
         "rollout_id": int, "task_id": int, "task_group": str|null,
         "task_descs": [str], "n_chunks": int, "batch_size": int, "n_env_steps": int}
      ],
      ...run_metadata (model, dataset, source, ...)
    }

Per-rollout npz array shapes (T = num_inference_steps, B = batch, K = chunk_size,
D = max_action_dim, A = real action_dim):
  time                     (n_chunks, T)
  noise                    (n_chunks, B, K, D)
  x_t                      (n_chunks, T+1, B, K, D)
  v_t                      (n_chunks, T,   B, K, D)
  chunk_actions            (n_chunks, B, K, A)
  env_step_at_chunk_start  (n_chunks,) int32
  chunk_idx                (n_chunks,) int32
  task_descs               (n_chunks, B) object
  terminated / truncated   (n_env_steps, B) bool

Context npz array shapes:
  ctx_images     (n_chunks, ncam, B, 3, H, W) float32
  ctx_img_masks  (n_chunks, ncam, B)
  ctx_tokens     (n_chunks, B, L)
  ctx_masks      (n_chunks, B, L)

numpy-only — imports here stay light so the loader runs in the lerobot-free env.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 3
MANIFEST_NAME = "manifest.json"
ROLLOUTS_DIRNAME = "rollouts"

ROLLOUT_KEYS = (
    "time", "noise", "x_t", "v_t", "chunk_actions",
    "env_step_at_chunk_start", "chunk_idx", "task_descs", "terminated", "truncated",
)
CONTEXT_KEYS = ("ctx_images", "ctx_img_masks", "ctx_tokens", "ctx_masks")


@dataclass
class FMDims:
    n_action_steps: int
    chunk_size: int
    max_action_dim: int
    action_dim: int
    num_inference_steps: int
    dt: float


def rollout_filename(idx: int, prefix: str = "") -> str:
    return f"{prefix}rollout_{idx:04d}.npz"


def context_filename(rollout_file: str) -> str:
    p = Path(rollout_file)
    return str(p.with_name(p.stem + "_context.npz"))


def write_rollout_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write one rollout's arrays (uncompressed — these fp32 tensors are large and
    barely compress; matches the legacy recorder's ``np.savez``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def write_context_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def write_manifest(fm_dir: Path, manifest: dict[str, Any], name: str = MANIFEST_NAME) -> Path:
    fm_dir.mkdir(parents=True, exist_ok=True)
    p = fm_dir / name
    with open(p, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return p


def read_manifest(fm_dir: Path | str) -> dict[str, Any]:
    fm_dir = Path(fm_dir)
    p = fm_dir if fm_dir.name == MANIFEST_NAME else fm_dir / MANIFEST_NAME
    with open(p) as f:
        m = json.load(f)
    version = int(m.get("version", -1))
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{p}: FM format version {version} != {FORMAT_VERSION}. This is a clean-break "
            "refactor; legacy v2 recordings are not read — re-record."
        )
    return m
