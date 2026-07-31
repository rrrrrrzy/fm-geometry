"""Numpy-only loader for FM recordings (format v3).

Preserves the legacy ``RolloutRecord`` / ``FMRecording`` public API exactly (so
the analysis stages port mechanically), but reads the v3 one-npz-per-rollout
layout via :mod:`fmaccel.recording.format`. No torch / lerobot — a recording
can be inspected on a laptop, without lerobot installed.

``chunk_actions`` are model-space (post ``[:, :, :action_dim]`` truncation, BEFORE
the env postprocessor that scales/unnormalizes into env action space); replay
re-runs the same pipeline so this is transparent there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fmaccel.recording import format as fmt


@dataclass
class RolloutRecord:
    rollout_id: int
    rollout_idx_in_task: int
    file: str
    task_id: int
    task_group: str | None
    time: np.ndarray                       # (n_chunks, T)
    noise: np.ndarray                      # (n_chunks, B, chunk_size, max_action_dim)
    x_t: np.ndarray                        # (n_chunks, T+1, B, chunk_size, max_action_dim)
    v_t: np.ndarray                        # (n_chunks, T,   B, chunk_size, max_action_dim)
    chunk_actions: np.ndarray              # (n_chunks, B, chunk_size, action_dim)
    env_step_at_chunk_start: np.ndarray    # (n_chunks,) int32
    chunk_idx: np.ndarray                  # (n_chunks,) int32
    task_descs: np.ndarray                 # (n_chunks, B) object/str
    terminated: np.ndarray                 # (n_env_steps, B) bool
    truncated: np.ndarray                  # (n_env_steps, B) bool
    dt: float
    # Optional exact FM prefix inputs (present iff recorded with capture_context).
    # All ``ctx_*`` arrays from the sidecar npz, keyed verbatim. The default π₀.₅
    # (LeRobot) schema is {ctx_images, ctx_img_masks, ctx_tokens, ctx_masks};
    # adapter-defined schemas (e.g. an adapter storing ctx_prefix_* tensors) pass through untouched.
    ctx_arrays: dict[str, np.ndarray] | None = None

    @property
    def has_context(self) -> bool:
        return self.ctx_arrays is not None

    # -- π₀.₅ (LeRobot) context schema accessors (None for other schemas) -----
    @property
    def ctx_images(self) -> np.ndarray | None:        # (n_chunks, ncam, B, 3, H, W)
        return (self.ctx_arrays or {}).get("ctx_images")

    @property
    def ctx_img_masks(self) -> np.ndarray | None:     # (n_chunks, ncam, B)
        return (self.ctx_arrays or {}).get("ctx_img_masks")

    @property
    def ctx_tokens(self) -> np.ndarray | None:        # (n_chunks, B, L)
        return (self.ctx_arrays or {}).get("ctx_tokens")

    @property
    def ctx_masks(self) -> np.ndarray | None:         # (n_chunks, B, L)
        return (self.ctx_arrays or {}).get("ctx_masks")

    def context(self, env_idx: int, chunk_idx: int) -> dict[str, np.ndarray] | None:
        # π₀.₅ (LeRobot) schema only — degrade to None for adapter-defined schemas that
        # lack the image/token prefix keys (e.g. packed ctx_prefix_* tensors, or a separate
        # ctx_lang_tokens + ctx_state pair). Those adapters resample via their own
        # ``sample_with_context`` off ``ctx_arrays``, not this accessor; returning None
        # here makes a generic consumer raise a clean "no context" instead of a TypeError.
        if self.ctx_images is None or self.ctx_tokens is None or self.ctx_masks is None:
            return None
        return {
            "images": self.ctx_images[chunk_idx, :, env_idx],        # (ncam, 3, H, W)
            "img_masks": self.ctx_img_masks[chunk_idx, :, env_idx],   # (ncam,)
            "tokens": self.ctx_tokens[chunk_idx, env_idx],            # (L,)
            "masks": self.ctx_masks[chunk_idx, env_idx],              # (L,)
        }

    @property
    def n_chunks(self) -> int:
        return int(self.x_t.shape[0])

    @property
    def num_inference_steps(self) -> int:
        return int(self.v_t.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.x_t.shape[2])

    @property
    def n_env_steps(self) -> int:
        return int(self.terminated.shape[0])

    def chunk(self, env_idx: int, chunk_idx: int) -> dict[str, Any]:
        return {
            "time": self.time[chunk_idx],
            "noise": self.noise[chunk_idx, env_idx],
            "x_t": self.x_t[chunk_idx, :, env_idx],
            "v_t": self.v_t[chunk_idx, :, env_idx],
            "chunk_actions": self.chunk_actions[chunk_idx, env_idx],
            "env_step_at_chunk_start": int(self.env_step_at_chunk_start[chunk_idx]),
            "task_desc": str(self.task_descs[chunk_idx, env_idx]),
        }

    def episode(self, env_idx: int) -> dict[str, np.ndarray]:
        """All chunks for a single batch slot. After the first True in
        ``terminated[:, env_idx]`` the subsequent chunks belong to a *new* episode
        (LIBERO auto-resets on done); use :meth:`episode_boundaries` to slice."""
        return {
            "time": self.time,
            "noise": self.noise[:, env_idx],
            "x_t": self.x_t[:, :, env_idx],
            "v_t": self.v_t[:, :, env_idx],
            "chunk_actions": self.chunk_actions[:, env_idx],
            "env_step_at_chunk_start": self.env_step_at_chunk_start,
            "chunk_idx": self.chunk_idx,
            "task_descs": self.task_descs[:, env_idx],
            "terminated": self.terminated[:, env_idx],
            "truncated": self.truncated[:, env_idx],
        }

    def episode_boundaries(self, env_idx: int) -> np.ndarray:
        done = self.terminated[:, env_idx] | self.truncated[:, env_idx]
        return np.flatnonzero(done).astype(np.int64)


@dataclass
class FMRecording:
    root: Path                 # the fm/ directory containing manifest.json
    manifest: dict[str, Any]
    rollouts: list[RolloutRecord]

    @property
    def dims(self) -> dict[str, Any]:
        return self.manifest.get("dims", {})

    @classmethod
    def load(
        cls,
        path: Path | str,
        rollout_ids: "Sequence[int] | None" = None,
        *,
        load_context: bool = True,
    ) -> "FMRecording":
        """Load from an ``fm/`` directory (or a path to its manifest.json).

        ``rollout_ids`` (optional): load ONLY the manifest entries whose ``rollout_id`` is in
        this set, in the given order — the memory-bounded path for large recordings where each
        rollout's context sidecar is tens of MB (loading all 2000 would need ~100GB RAM). When
        None, every rollout is loaded (the original behaviour). Downstream stages that pass a
        ``--rollouts`` shard should hand the SAME ids here so a worker only holds its own slice;
        the returned ``rollouts`` list preserves ``rollout_id`` on each record, so output naming
        (``_ro<rollout_id>``) is unchanged regardless of how the list was filtered.

        ``load_context=False`` skips the large ``*_context.npz`` sidecars. Geometry and other
        consumers that only read the recorded FM trajectory should use this path; otherwise a
        full batched LIBERO run can waste hundreds of GB loading conditioning tensors it never
        touches.
        """
        p = Path(path)
        root = p.parent if p.name == fmt.MANIFEST_NAME else p
        manifest = fmt.read_manifest(root)
        dt = float(manifest["dims"]["dt"])

        want = None if rollout_ids is None else {int(i) for i in rollout_ids}
        rollouts: list[RolloutRecord] = []
        for ro in manifest["rollouts"]:
            if want is not None and int(ro["rollout_id"]) not in want:
                continue
            npz_path = root / ro["file"]
            with np.load(npz_path, allow_pickle=True) as data:
                rec = RolloutRecord(
                    rollout_id=int(ro["rollout_id"]),
                    rollout_idx_in_task=int(ro.get("rollout_idx_in_task", ro["rollout_id"])),
                    file=ro["file"],
                    task_id=int(ro["task_id"]),
                    task_group=ro.get("task_group"),
                    time=np.asarray(data["time"]),
                    noise=np.asarray(data["noise"]),
                    x_t=np.asarray(data["x_t"]),
                    v_t=np.asarray(data["v_t"]),
                    chunk_actions=np.asarray(data["chunk_actions"]),
                    env_step_at_chunk_start=np.asarray(data["env_step_at_chunk_start"]),
                    chunk_idx=np.asarray(data["chunk_idx"]),
                    task_descs=np.asarray(data["task_descs"]),
                    terminated=np.asarray(data["terminated"]),
                    truncated=np.asarray(data["truncated"]),
                    dt=dt,
                )
            ctx_file = ro.get("context_file")
            if load_context and ctx_file:
                ctx_path = root / ctx_file
                if ctx_path.exists():
                    with np.load(ctx_path, allow_pickle=True) as cdata:
                        rec.ctx_arrays = {k: np.asarray(cdata[k]) for k in cdata.files}
            rollouts.append(rec)
        return cls(root=root, manifest=manifest, rollouts=rollouts)
