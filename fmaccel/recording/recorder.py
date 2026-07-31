"""Model-agnostic FM-head recorder, driven by an :class:`FMModelAdapter`.

Splits cleanly into three hook layers:
  * **policy hooks** (generic) — ``reset`` / ``select_action`` /
    ``predict_action_chunk``; read only ``batch["task"]`` + the returned chunk,
    so they work for any policy exposing those method names.
  * **FM-head hooks** (model-specific) — installed by ``adapter.attach_fm_hooks``;
    the adapter intercepts its own ``sample_actions`` / ``denoise_step`` signatures
    (π₀.₅'s image/token prefix vs the toy's state vector) and calls back into
    :meth:`on_sample_actions` / :meth:`on_denoise_step` with the model-agnostic
    payload (noise, ``(t, x_t, v_t)``). This is what made the old ``FMRecorder`` +
    ``ToyFMRecorder`` two classes; now it is one recorder + N adapters.
  * **VectorEnv.step hooks** (generic) — capture ``(terminated, truncated)`` per
    env step to recover episode boundaries (LIBERO auto-resets on done). With
    ``envs=None`` (ZMQ server / toy / teacher-forced) call :meth:`set_task` once
    after ``attach()``.

Serialization is the v3 one-npz-per-rollout format (:mod:`fmaccel.recording.format`).

Usage::

    rec = FMRecorder(adapter=adapter, output_dir=run.fm_dir, envs=envs,
                     run_metadata={...}, capture_context=True)
    rec.attach()
    # ... run eval / drive the policy ...
    rec.detach()        # flushes + restores patched methods
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from fmaccel.recording import format as fmt

logger = logging.getLogger(__name__)


@dataclass
class _ChunkBuffer:
    chunk_idx: int
    env_step_at_chunk_start: int
    task_descs: list[str]
    noise: torch.Tensor | None = None
    times: list[float] = field(default_factory=list)
    x_t_in: list[torch.Tensor] = field(default_factory=list)
    v_t_out: list[torch.Tensor] = field(default_factory=list)
    x_final: torch.Tensor | None = None             # exact x(0) if the adapter reports it
    chunk_actions: torch.Tensor | None = None
    ctx: dict | None = None                         # only when capture_context


@dataclass
class _RolloutBuffer:
    rollout_id: int
    task_id: int | None = None
    task_group: str | None = None
    chunks: list[_ChunkBuffer] = field(default_factory=list)
    terminated: list[np.ndarray] = field(default_factory=list)
    truncated: list[np.ndarray] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.chunks and not self.terminated


class FMRecorder:
    def __init__(
        self,
        adapter,
        output_dir: Path | str,
        envs: Mapping[str, Mapping[int, Any]] | None = None,
        run_metadata: dict[str, Any] | None = None,
        file_prefix: str = "",
        manifest_name: str = fmt.MANIFEST_NAME,
        capture_context: bool = False,
    ) -> None:
        self.adapter = adapter
        self.policy = adapter.policy
        self.model = adapter.model
        self.output_dir = Path(output_dir)              # the fm/ directory
        self.run_metadata = dict(run_metadata or {})
        self.file_prefix = file_prefix
        self.manifest_name = manifest_name  # per-worker name for multi-GPU shards

        cfg = adapter.config
        self.num_inference_steps = int(cfg.num_inference_steps)
        self.chunk_size = int(cfg.chunk_size)
        self.max_action_dim = int(cfg.max_action_dim)
        self.n_action_steps = int(cfg.n_action_steps)
        self.action_dim = int(cfg.action_dim)

        self.capture_context = bool(capture_context) and bool(adapter.supports_context)
        if capture_context and not adapter.supports_context:
            logger.warning("capture_context requested but %s.supports_context is False; ignoring.",
                           type(adapter).__name__)

        self._vec_envs = self._collect_vec_envs(envs)

        self._attached = False
        self._env_step = 0
        self._chunk_idx_in_rollout = 0
        self._rollout_id_global = -1

        self._current_chunk: _ChunkBuffer | None = None
        self._current_rollout: _RolloutBuffer | None = None

        self._pending_task_id: int | None = None
        self._pending_task_group: str | None = None
        self._pending_rollouts: list[_RolloutBuffer] = []

        self._rollout_file_idx = 0
        self._manifest_rollouts: list[dict[str, Any]] = []
        self._any_context = False

        self._orig_policy_reset: Callable | None = None
        self._orig_policy_select_action: Callable | None = None
        self._orig_policy_predict_action_chunk: Callable | None = None
        self._orig_env_steps: list[tuple[Any, Callable]] = []

    @staticmethod
    def _collect_vec_envs(envs) -> list[tuple[str, int, Any]]:
        if envs is None:
            return []
        return [(str(s), int(t), e) for s, group in envs.items() for t, e in group.items()]

    # ----------------------------------------------------------------- attach
    def attach(self) -> None:
        if self._attached:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._attach_policy_hooks()
        self.adapter.attach_fm_hooks(self)          # model-specific FM-head hooks
        for suite_name, task_id, vec_env in self._vec_envs:
            self._patch_vec_env_step(vec_env, suite_name, task_id)
        self._attached = True
        logger.info(
            "FMRecorder attached: T=%d chunk=%d max_adim=%d adim=%d n_action_steps=%d vec_envs=%d",
            self.num_inference_steps, self.chunk_size, self.max_action_dim, self.action_dim,
            self.n_action_steps, len(self._vec_envs),
        )

    def _attach_policy_hooks(self) -> None:
        policy = self.policy
        rec = self

        self._orig_policy_reset = policy.reset

        def _patched_reset(*a, **kw):
            rec._close_current_rollout()
            rec._rollout_id_global += 1
            rec._env_step = 0
            rec._chunk_idx_in_rollout = 0
            rec._current_chunk = None
            rec._current_rollout = _RolloutBuffer(rollout_id=rec._rollout_id_global)
            return rec._orig_policy_reset(*a, **kw)

        policy.reset = _patched_reset

        self._orig_policy_select_action = policy.select_action

        def _patched_select_action(batch, *a, **kw):
            out = rec._orig_policy_select_action(batch, *a, **kw)
            rec._env_step += 1
            return out

        policy.select_action = _patched_select_action

        self._orig_policy_predict_action_chunk = policy.predict_action_chunk

        def _patched_predict_action_chunk(batch, *a, **kw):
            task_descs_raw = batch.get("task", None)
            if task_descs_raw is None:
                bsize_guess = 1
                for v in batch.values():
                    if hasattr(v, "shape") and len(v.shape) > 0:
                        bsize_guess = int(v.shape[0])
                        break
                task_descs = [""] * bsize_guess
            else:
                task_descs = [str(t) for t in task_descs_raw]

            rec._current_chunk = _ChunkBuffer(
                chunk_idx=rec._chunk_idx_in_rollout,
                env_step_at_chunk_start=rec._env_step,
                task_descs=task_descs,
            )
            actions = rec._orig_policy_predict_action_chunk(batch, *a, **kw)
            rec._current_chunk.chunk_actions = actions.detach().cpu().to(torch.float32)
            if rec._current_rollout is not None:
                rec._current_rollout.chunks.append(rec._current_chunk)
            rec._current_chunk = None
            rec._chunk_idx_in_rollout += 1
            return actions

        policy.predict_action_chunk = _patched_predict_action_chunk

    # --------------------------------------------- adapter -> recorder callbacks
    def on_sample_actions(self, noise: torch.Tensor, context: dict | None = None) -> None:
        """Called by the adapter's ``sample_actions`` hook with the FM noise (and,
        if ``capture_context``, the exact prefix inputs the adapter built)."""
        if self._current_chunk is None:
            return
        self._current_chunk.noise = noise.detach().cpu().to(torch.float32)
        if self.capture_context and context is not None:
            self._current_chunk.ctx = context

    def on_denoise_step(self, t_scalar: float, x_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """Called by the adapter's ``denoise_step`` hook per FM iteration. Records
        ``(t, x_t, v_t)`` for this step."""
        if self._current_chunk is None:
            return
        self._current_chunk.times.append(float(t_scalar))
        self._current_chunk.x_t_in.append(x_t.detach().cpu().to(torch.float32))
        self._current_chunk.v_t_out.append(v_t.detach().cpu().to(torch.float32))

    def on_sample_actions_end(self, x_final: torch.Tensor) -> None:
        """Optionally called by the adapter with ``sample_actions``'s actual return
        value, so ``x_t[T]`` is the model's exact endpoint instead of the recorder's
        ``x_t[T-1] + dt*v_t[T-1]`` reconstruction. This matters when the model's own
        integrator rounds differently — e.g. a head that computes ``dt*v_t`` in bf16 under
        autocast lands ~1e-3 off the float64 reconstruction, which would break the
        ``x_t[T] == chunk_actions`` bitwise invariant the resample gate checks."""
        if self._current_chunk is None:
            return
        self._current_chunk.x_final = x_final.detach().cpu().to(torch.float32)

    # -------------------------------------------------------------- env hooks
    def _patch_vec_env_step(self, vec_env, suite_name: str, task_id: int) -> None:
        orig_step = vec_env.step
        rec = self

        def _patched_step(action):
            obs, reward, terminated, truncated, info = orig_step(action)
            rec._maybe_switch_task(task_id, suite_name)
            if rec._current_rollout is not None:
                rec._current_rollout.terminated.append(np.asarray(terminated, dtype=bool).reshape(-1))
                rec._current_rollout.truncated.append(np.asarray(truncated, dtype=bool).reshape(-1))
            return obs, reward, terminated, truncated, info

        vec_env.step = _patched_step
        self._orig_env_steps.append((vec_env, orig_step))

    def _maybe_switch_task(self, task_id: int, task_group: str) -> None:
        if self._pending_task_id is None:
            self._pending_task_id = task_id
            self._pending_task_group = task_group
            return
        if self._pending_task_id != task_id:
            self._flush_task_buffer()
            self._pending_task_id = task_id
            self._pending_task_group = task_group

    def set_task(self, task_id: int, task_group: str = "") -> None:
        """Tag the recording with a task when there is no patched env (``envs=None``)."""
        self._maybe_switch_task(int(task_id), str(task_group))

    def set_env_step(self, step: int) -> None:
        """Externally set the env-step counter (it normally advances per
        ``select_action``). For drivers that call ``predict_action_chunk`` directly —
        e.g. dense per-state labeling — call this before each chunk so
        ``env_step_at_chunk_start`` records the dataset timestep."""
        self._env_step = int(step)

    # ----------------------------------------------------------- close/flush
    def _close_current_rollout(self) -> None:
        if self._current_rollout is None or self._current_rollout.is_empty():
            self._current_rollout = None
            return
        ro = self._current_rollout
        if ro.terminated:
            last_done = ro.terminated[-1] | ro.truncated[-1]
            if not last_done.all():
                new_last_trunc = ro.truncated[-1].copy()
                new_last_trunc[~last_done] = True
                ro.truncated[-1] = new_last_trunc
        self._pending_rollouts.append(ro)
        self._current_rollout = None

    def _flush_task_buffer(self) -> None:
        rollouts = [ro for ro in self._pending_rollouts if ro.chunks]
        self._pending_rollouts = []
        if not rollouts:
            return
        task_id = self._pending_task_id
        task_group = self._pending_task_group
        if task_id is None:
            logger.warning("FMRecorder: rollouts pending without a task_id; skipping flush.")
            return

        T = self.num_inference_steps
        dt = -1.0 / T
        for ro in rollouts:
            self._write_rollout(ro, task_id, task_group, dt)

    def _write_rollout(self, ro: _RolloutBuffer, task_id: int, task_group: str | None, dt: float) -> None:
        chunks = ro.chunks
        time_arr = np.stack([np.asarray(c.times, dtype=np.float32) for c in chunks])
        noise_arr = np.stack([c.noise.numpy() for c in chunks])

        x_t_full, v_t_per = [], []
        for c in chunks:
            x_t_in = np.stack([t.numpy() for t in c.x_t_in])
            v_t = np.stack([t.numpy() for t in c.v_t_out])
            x_final = c.x_final.numpy() if c.x_final is not None else x_t_in[-1] + dt * v_t[-1]
            x_t_full.append(np.concatenate([x_t_in, x_final[None]], axis=0))
            v_t_per.append(v_t)
        x_t_arr = np.stack(x_t_full)
        v_t_arr = np.stack(v_t_per)
        chunk_actions_arr = np.stack([c.chunk_actions.numpy() for c in chunks])
        env_step_arr = np.asarray([c.env_step_at_chunk_start for c in chunks], dtype=np.int32)
        chunk_idx_arr = np.asarray([c.chunk_idx for c in chunks], dtype=np.int32)
        task_descs_arr = np.asarray([c.task_descs for c in chunks], dtype=object)

        B = int(noise_arr.shape[1])
        if ro.terminated:
            terminated_arr = np.stack(ro.terminated)
            truncated_arr = np.stack(ro.truncated)
        else:
            terminated_arr = np.zeros((0, B), dtype=bool)
            truncated_arr = np.zeros((0, B), dtype=bool)

        arrays: dict[str, np.ndarray] = {
            "time": time_arr, "noise": noise_arr, "x_t": x_t_arr, "v_t": v_t_arr,
            "chunk_actions": chunk_actions_arr, "env_step_at_chunk_start": env_step_arr,
            "chunk_idx": chunk_idx_arr, "task_descs": task_descs_arr,
            "terminated": terminated_arr, "truncated": truncated_arr,
        }

        idx = self._rollout_file_idx
        self._rollout_file_idx += 1
        rel_file = f"{fmt.ROLLOUTS_DIRNAME}/{fmt.rollout_filename(idx, self.file_prefix)}"
        fmt.write_rollout_npz(self.output_dir / rel_file, arrays)

        rel_ctx = None
        if self.capture_context:
            rel_ctx = self._write_context(rel_file, chunks)

        self._manifest_rollouts.append({
            "file": rel_file,
            "context_file": rel_ctx,
            "rollout_id": int(ro.rollout_id),
            "task_id": int(task_id),
            "task_group": task_group,
            "task_descs": [str(s) for s in task_descs_arr[0]],
            "n_chunks": int(len(chunks)),
            "batch_size": B,
            "n_env_steps": int(terminated_arr.shape[0]),
        })
        logger.info("FMRecorder: wrote %s (chunks=%d B=%d)", rel_file, len(chunks), B)

    def _write_context(self, rollout_file: str, chunks: list[_ChunkBuffer]) -> str | None:
        if any(c.ctx is None for c in chunks):
            logger.warning("capture_context: %s has chunks without ctx; skipping context sidecar", rollout_file)
            return None
        pack = getattr(self.adapter, "pack_context", None)
        if pack is not None:
            # Adapter-defined context schema (e.g. an adapter that stores packed prefix tensors).
            arrays = pack([c.ctx for c in chunks])
        else:
            # Default π₀.₅ (LeRobot) schema: raw prefix inputs (images/tokens).
            arrays = {
                "ctx_images": np.stack(
                    [np.stack([im.numpy() for im in c.ctx["images"]]) for c in chunks]
                ).astype(np.float32),
                "ctx_img_masks": np.stack([np.stack([m.numpy() for m in c.ctx["img_masks"]]) for c in chunks]),
                "ctx_tokens": np.stack([c.ctx["tokens"].numpy() for c in chunks]),
                "ctx_masks": np.stack([c.ctx["masks"].numpy() for c in chunks]),
            }
        rel_ctx = fmt.context_filename(rollout_file)
        fmt.write_context_npz(self.output_dir / rel_ctx, arrays)
        self._any_context = True
        return rel_ctx

    def flush(self) -> None:
        self._close_current_rollout()
        self._flush_task_buffer()
        manifest = {
            "version": fmt.FORMAT_VERSION,
            "dims": {
                "n_action_steps": self.n_action_steps,
                "chunk_size": self.chunk_size,
                "max_action_dim": self.max_action_dim,
                "action_dim": self.action_dim,
                "num_inference_steps": self.num_inference_steps,
                "dt": -1.0 / self.num_inference_steps,
            },
            "has_context": bool(self._any_context),
            "rollouts": self._manifest_rollouts,
            **self.run_metadata,
        }
        fmt.write_manifest(self.output_dir, manifest, name=self.manifest_name)
        logger.info("FMRecorder: wrote %s with %d rollouts", self.manifest_name, len(self._manifest_rollouts))

    # ----------------------------------------------------------------- detach
    def detach(self) -> None:
        if not self._attached:
            return
        try:
            self.flush()
        except Exception:
            logger.exception("FMRecorder.detach: flush() raised; restoring methods anyway.")
        self.policy.reset = self._orig_policy_reset
        self.policy.select_action = self._orig_policy_select_action
        self.policy.predict_action_chunk = self._orig_policy_predict_action_chunk
        self.adapter.detach_fm_hooks()
        for vec_env, orig_step in self._orig_env_steps:
            vec_env.step = orig_step
        self._orig_env_steps = []
        self._attached = False
