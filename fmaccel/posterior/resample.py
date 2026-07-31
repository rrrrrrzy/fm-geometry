"""Re-run pi0.5's FM head on a recorded `(rollout, env_idx, chunk)`.

Pairs with `fmaccel.recording.recorder.FMRecorder` (which captured the chunk in the
first place) and `fmaccel.replay.LiberoReplayer` (which replays whole
rollouts into MuJoCo). This module is for a different kind of question: take
*one* recorded chunk, rebuild the exact model input it saw, and re-run the
FM-head denoiser with a different noise draw to see how chunk output varies.

Two public surfaces:

- `ChunkResampleSession.reproduce()`: feed the stored noise back into the FM
  denoise loop and verify the produced chunk matches the recorded
  `chunk_actions` (sanity check that we rebuilt the input correctly).
- `ChunkResampleSession.resample(n_samples=...)`: sample N new noise tensors
  and run them through the same model state, returning chunks + (optionally)
  the full per-step FM trajectory for each sample.

Two layers of cheap reuse make this fast:

1. **Prefix cache (in-memory).** `setup()` runs the PaliGemma vision/language
   prefix once at B=1 and stashes (`prefix_pad_masks`, `past_key_values`). Each
   resample micro-batch expands them via `.expand` views (allocation-free) and
   runs only the Gemma-expert FM denoise loop. Skipping the per-micro-batch
   prefix forward is what lets micro_batch=2048 fit on a single H100/B200 —
   the eager prefix attention would otherwise materialize a
   (B, heads, S, S) fp32 matrix.
2. **Batch cache (on-disk).** The first run for a `(rollout, env_idx,
   chunk_idx, obs_size)` stages a single-slot LIBERO env, replays stored
   actions to the chunk start, preprocesses the obs, and saves the resulting
   policy-input batch under `<recording>/_resample_cache/`. Subsequent runs
   load that file and skip the LIBERO env entirely (the env+MuJoCo startup is
   the slowest single step in the pipeline — ~90s).

Reproduce-fidelity depends on how the conditioning is obtained:

- **Captured context (`use_context=True`, recording made with capture_context):**
  the exact embed_prefix inputs are reused, so reproduce matches the recorded
  chunk to the eager-attention floor (~1e-3 mean / ~1e-2 max). Preferred.
- **Env replay (no captured context):** the observation is reconstructed by
  re-stepping MuJoCo, which drifts from the recorded observation; reproduce
  error can reach O(1) (audited: max 2.5–3.0 on 3 of 4 legacy chunks). The FM
  velocity field is sensitive enough to that mismatch to move the endpoint, so
  posterior / multimodality analysis on replay-conditioned resamples is unsafe.

`resample_to_npz` runs the reproduce check and sets `meta["reproduce_ok"]`
(against `reproduce_tol`, default 5e-2), warning loudly when it fails.

Scope: only chunks *within the first episode* of `(rollout, env_idx)` —
chunks after the first `terminated|truncated` belong to a LIBERO auto-reset
new episode whose init_state isn't trivially recoverable here.

Future: `rollout_remainder()` is a stubbed extension point — continue env
stepping using a chosen resampled chunk, then run the policy normally for the
remainder. See the docstring on that method for the intended design.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fmaccel.recording.loader import FMRecording, RolloutRecord

logger = logging.getLogger(__name__)


def _build_policy_cfg(
    policy_path: str,
    *,
    device: str,
    n_action_steps: int,
    num_inference_steps: int,
):
    """Load + tweak the pi0.5 PreTrainedConfig for resampling.

    compile_model is forced off so `denoise_step` stays patchable (torch.compile
    bakes the FM loop into a graph and bypasses Python-level hooks); n_action_steps
    / num_inference_steps are pinned to the recording's values.
    """
    from lerobot.configs import PreTrainedConfig

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = False
    if hasattr(policy_cfg, "n_action_steps"):
        policy_cfg.n_action_steps = n_action_steps
    if hasattr(policy_cfg, "num_inference_steps"):
        policy_cfg.num_inference_steps = num_inference_steps
    return policy_cfg


def build_resample_policy(
    policy_path: str,
    *,
    device: str,
    suite_name: str | None = None,
    task_id: int | None = None,
    obs_size: int | None = None,
    n_action_steps: int,
    num_inference_steps: int,
):
    """Build a context-mode resampling policy straight from the checkpoint.

    Factored out of `ChunkResampleSession.setup` so a multi-chunk sweep can load
    the weights ONCE and inject the result into many per-chunk sessions via
    `ChunkResampleSession(..., policy=...)` — `from_pretrained` is the only
    multi-second step and is identical across chunks of the same recording.

    Context mode never starts an env, and a pretrained pi0.5 checkpoint already
    carries the correct input/output features in its config, so we load the
    policy directly from `policy_path` (mirroring `pi05_policy_server._build`).
    The earlier `make_policy(env_cfg=...)` route derived features from a
    *synthesized LIBERO* env_cfg and unconditionally overwrote `output_features`
    (factory.py), which silently gave the wrong action dim for non-LIBERO
    recordings and the wrong observation keys. `suite_name`/`task_id`/`obs_size` are accepted for
    call-site compatibility but unused here (they only matter for the
    env-replay path, which builds its own policy inline).
    """
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(policy_path)
    policy.config.device = device
    if hasattr(policy.config, "compile_model"):
        policy.config.compile_model = False  # keep denoise_step patchable
    if hasattr(policy.config, "n_action_steps"):
        policy.config.n_action_steps = n_action_steps
    if hasattr(policy.config, "num_inference_steps"):
        policy.config.num_inference_steps = num_inference_steps
    policy.to(device)
    policy.eval()
    return policy


class ChunkResampleSession:
    """One env + policy + chunk-position, set up once and reusable for
    reproduce/resample calls.

    Use as a context manager so the env always closes:

        with ChunkResampleSession(rec, rollout_idx=0, env_idx=0, chunk_idx=3) as sess:
            check = sess.reproduce()
            samples = sess.resample(n_samples=64, seed=0, capture_trajectory=True)
    """

    def __init__(
        self,
        recording: FMRecording,
        rollout_idx: int,
        env_idx: int,
        chunk_idx: int,
        *,
        device: str = "cuda",
        obs_size: int = 360,
        num_inference_steps: int | None = None,
        batch_cache_dir: Path | str | None = None,
        use_batch_cache: bool = True,
        use_context: bool = True,
        policy: Any | None = None,
    ) -> None:
        if rollout_idx < 0 or rollout_idx >= len(recording.rollouts):
            raise IndexError(
                f"rollout_idx={rollout_idx} out of range [0, {len(recording.rollouts)})"
            )
        self.recording = recording
        self.rollout: RolloutRecord = recording.rollouts[rollout_idx]
        if env_idx < 0 or env_idx >= self.rollout.batch_size:
            raise IndexError(
                f"env_idx={env_idx} out of range [0, {self.rollout.batch_size})"
            )
        if chunk_idx < 0 or chunk_idx >= self.rollout.n_chunks:
            raise IndexError(
                f"chunk_idx={chunk_idx} out of range [0, {self.rollout.n_chunks})"
            )

        # When the recording carries exact captured context, resample rebuilds
        # the recorded conditioning directly (no env replay), so the
        # first-episode restriction — which exists only because replay can't
        # reconstruct post-auto-reset init states — does not apply.
        self._use_context = bool(use_context) and self.rollout.has_context
        if not self._use_context:
            # The no-context fallback reconstructs the observation by replaying
            # actions into a single-slot LIBERO env. That is meaningless for
            # recordings made by an out-of-process pi0.5 policy server (its env lives
            # in a separate process over ZMQ), so fail loudly with the fix instead of
            # spinning up a bogus LIBERO env.
            if recording.manifest.get("source") == "pi05_policy_server":
                raise ValueError(
                    "Recording was made by pi05_policy_server without captured "
                    "context (has_context=False); the no-context fallback replays "
                    "into a LIBERO env, which does not apply here. Re-record the "
                    "eval with --record-fm-context, then resample with "
                    "use_context=True (the default)."
                )
            boundaries = self.rollout.episode_boundaries(env_idx)
            if boundaries.size > 0:
                first_done = int(boundaries[0])
                chunk_start = int(self.rollout.env_step_at_chunk_start[chunk_idx])
                if chunk_start > first_done:
                    raise ValueError(
                        f"chunk_idx={chunk_idx} starts at env_step={chunk_start}, after the "
                        f"first done at env_step={first_done} for env_idx={env_idx}. Only "
                        "chunks within the first episode are supported without captured "
                        "context. Re-record with capture_context=True to lift this."
                    )

        self.rollout_idx = int(rollout_idx)
        self.env_idx = int(env_idx)
        self.chunk_idx = int(chunk_idx)
        self.device = str(device)
        self.obs_size = int(obs_size)
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else self.rollout.num_inference_steps
        )

        m = recording.manifest
        suite_name = self.rollout.task_group or m.get("task_group")
        if not suite_name:
            raise KeyError(
                "Recording has no task_group (rollout or manifest); needed to rebuild env."
            )
        self.suite_name: str = str(suite_name)
        self.task_id: int = int(self.rollout.task_id)
        # Server-side recordings (source="pi05_policy_server") store the model under
        # "checkpoint" and carry no "policy_path"; LIBERO eval recordings use
        # "policy_path".
        policy_path = m.get("policy_path") or m.get("checkpoint")
        if not policy_path:
            raise KeyError(
                "Recording manifest has neither 'policy_path' nor 'checkpoint'; "
                "cannot locate the pi0.5 weights to rebuild for resampling."
            )
        self.policy_path: str = str(policy_path)
        dims = m["dims"]  # FM format v3: dims block (was top-level in v2)
        self.n_action_steps: int = int(dims["n_action_steps"])
        self.chunk_size: int = int(dims["chunk_size"])
        self.max_action_dim: int = int(dims["max_action_dim"])
        self.batch_size_orig: int = int(self.rollout.batch_size)
        # "seed" only feeds the env-replay path (re-stepping MuJoCo to rebuild
        # the LIBERO observation); context-mode resample ignores it. Server
        # recordings omit it, so default to 0 — harmless when use_context=True.
        start_seed = int(m.get("seed", 0))
        self.seed: int = (
            start_seed
            + int(self.rollout.rollout_idx_in_task) * self.batch_size_orig
            + self.env_idx
        )

        # Batch cache: keyed by (rollout, env, chunk, obs_size). Lives next to
        # the recording by default so it travels with it; clearable by deleting
        # the cache dir. On a hit we skip LIBERO env startup entirely (~90s).
        self.use_batch_cache = bool(use_batch_cache)
        if batch_cache_dir is not None:
            self.batch_cache_dir = Path(batch_cache_dir)
        else:
            self.batch_cache_dir = Path(self.recording.root) / "_resample_cache"

        self._setup_done = False
        self._envs: Any = None
        self._env: Any = None
        # An externally-built policy can be injected so a multi-chunk sweep
        # loads the weights once and reuses them across many per-chunk sessions;
        # close() won't free an injected policy (the owner still holds it).
        self._injected_policy = policy is not None
        self._policy: Any = policy
        self._batch: dict[str, Any] | None = None
        self._action_dim: int | None = None
        # Cached prefix forward outputs (B=1); reused across all resample
        # micro-batches via .expand views — see _expanded_prefix().
        self._prefix_pad_masks_1: torch.Tensor | None = None  # (1, S_prefix)
        self._past_key_values_1: Any = None                   # transformers DynamicCache
        # When using recorded context: the exact embed_prefix inputs for this
        # (chunk_idx, env_idx) slot, as device tensors. _compute_prefix_cache
        # uses these instead of replaying/re-preprocessing the observation.
        self._ctx_slot: tuple | None = None

    # ------------------------------------------------------------------ setup
    def __enter__(self) -> "ChunkResampleSession":
        self.setup()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def setup(self) -> "ChunkResampleSession":
        if self._setup_done:
            return self
        from lerobot.policies import make_policy
        from lerobot.utils.constants import ACTION
        from lerobot.utils.random_utils import set_seed

        set_seed(int(self.recording.manifest.get("seed", 0)))

        # Try the batch cache first — if hit, we skip the entire LIBERO env
        # startup chain (make_env + MuJoCo init + replay-to-chunk). Captured
        # context supersedes the batch cache entirely (it's exact, replay isn't).
        cached_batch = (
            self._load_batch_cache()
            if (self.use_batch_cache and not self._use_context)
            else None
        )

        if self._use_context:
            # Exact recorded conditioning is available — no env, no replay, no
            # re-preprocessing. Reuse an injected policy if one was given (a
            # multi-chunk sweep loads the weights once and shares them across
            # per-chunk sessions); otherwise build one here. Then load the
            # context slot; the prefix is rebuilt from the stored embed_prefix
            # inputs.
            if self._policy is None:
                self._policy = build_resample_policy(
                    self.policy_path,
                    device=self.device,
                    suite_name=self.suite_name,
                    task_id=self.task_id,
                    obs_size=self.obs_size,
                    n_action_steps=self.n_action_steps,
                    num_inference_steps=self.num_inference_steps,
                )
            self._policy.eval()
            self._action_dim = int(self._policy.config.output_features[ACTION].shape[0])
            self._load_context_slot()
            logger.info(
                "using recorded context for (r%d, e%d, c%d) — skipping LIBERO env + replay",
                self.rollout_idx, self.env_idx, self.chunk_idx,
            )
        elif cached_batch is not None:
            # No env needed — build the policy without ever importing the LIBERO
            # env adapter. make_policy still requires an env_cfg for feature
            # shape inference, so synthesize a minimal one.
            from lerobot.envs.configs import EnvConfig
            policy_cfg = _build_policy_cfg(
                self.policy_path,
                device=self.device,
                n_action_steps=self.n_action_steps,
                num_inference_steps=self.num_inference_steps,
            )
            libero_cls = EnvConfig.get_choice_class("libero")
            env_cfg = libero_cls(
                task=self.suite_name,
                task_ids=[self.task_id],
                observation_height=self.obs_size,
                observation_width=self.obs_size,
            )
            policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
            policy.eval()
            self._policy = policy
            self._action_dim = int(policy.config.output_features[ACTION].shape[0])
            self._batch = cached_batch
            logger.info(
                "batch cache hit at %s — skipping LIBERO env startup",
                self._batch_cache_path(),
            )
        else:
            from lerobot.envs import make_env, make_env_pre_post_processors
            from lerobot.envs.configs import EnvConfig
            from lerobot.envs.utils import preprocess_observation
            from lerobot.policies import make_pre_post_processors

            policy_cfg = _build_policy_cfg(
                self.policy_path,
                device=self.device,
                n_action_steps=self.n_action_steps,
                num_inference_steps=self.num_inference_steps,
            )
            libero_cls = EnvConfig.get_choice_class("libero")
            env_cfg = libero_cls(
                task=self.suite_name,
                task_ids=[self.task_id],
                observation_height=self.obs_size,
                observation_width=self.obs_size,
            )
            envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
            env = envs[self.suite_name][self.task_id]

            policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
            policy.eval()

            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=self.policy_path,
                preprocessor_overrides={
                    "device_processor": {"device": self.device},
                    "rename_observations_processor": {"rename_map": {}},
                },
            )
            env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

            self._envs = envs
            self._env = env
            self._policy = policy
            self._preprocessor = preprocessor
            self._postprocessor = postprocessor
            self._env_pre = env_pre
            self._env_post = env_post
            self._preprocess_observation = preprocess_observation
            self._ACTION = ACTION
            self._action_dim = int(policy.config.output_features[ACTION].shape[0])

            # Step the env to the chunk start using stored actions for this slot.
            self._batch = self._advance_to_chunk()
            if self.use_batch_cache:
                self._save_batch_cache()
            # Close env now — once the batch is built we don't need it again.
            # (rollout_remainder, if/when implemented, would need to reopen.)
            from lerobot.envs import close_envs
            close_envs(self._envs)
            self._envs = None
            self._env = None

        # Run paligemma prefix forward once at B=1 and cache. Every micro-batch
        # in resample() reuses this via .expand views, skipping the costly
        # (B, heads, S, S) fp32 prefix attention that the eager path otherwise
        # materializes per micro-batch (the source of the original OOM).
        self._compute_prefix_cache()

        self._setup_done = True
        logger.info(
            "ChunkResampleSession ready: suite=%s task=%d env_idx=%d chunk_idx=%d "
            "env_step_at_chunk_start=%d seed=%d action_dim=%d",
            self.suite_name,
            self.task_id,
            self.env_idx,
            self.chunk_idx,
            int(self.rollout.env_step_at_chunk_start[self.chunk_idx]),
            self.seed,
            self._action_dim,
        )
        return self

    # ----------------------------------------------------------- batch cache
    def _batch_cache_path(self) -> Path:
        fname = (
            f"batch_r{self.rollout_idx}_e{self.env_idx}_c{self.chunk_idx}"
            f"_o{self.obs_size}.pt"
        )
        return self.batch_cache_dir / fname

    def _save_batch_cache(self) -> None:
        path = self._batch_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        assert self._batch is not None
        for k, v in self._batch.items():
            if isinstance(v, torch.Tensor):
                payload[k] = v.detach().to("cpu")
            else:
                payload[k] = v
        blob = {
            "batch": payload,
            "policy_path": self.policy_path,
            "obs_size": self.obs_size,
            "task_group": self.suite_name,
            "task_id": self.task_id,
            "env_step_at_chunk_start": int(
                self.rollout.env_step_at_chunk_start[self.chunk_idx]
            ),
            "seed": self.seed,
            "version": 1,
        }
        torch.save(blob, path)
        logger.info("cached policy-input batch → %s", path)

    def _load_batch_cache(self) -> dict[str, Any] | None:
        path = self._batch_cache_path()
        if not path.exists():
            return None
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("batch cache %s failed to load (%s); ignoring", path, e)
            return None
        # Conservative invalidation: policy checkpoint changed, or obs size /
        # env_step_at_chunk_start drifted from the recording. These should
        # never silently mismatch — the batch depends on all of them.
        if blob.get("policy_path") != self.policy_path:
            logger.warning("batch cache policy_path mismatch; ignoring %s", path)
            return None
        if int(blob.get("obs_size", -1)) != self.obs_size:
            logger.warning("batch cache obs_size mismatch; ignoring %s", path)
            return None
        target = int(self.rollout.env_step_at_chunk_start[self.chunk_idx])
        if int(blob.get("env_step_at_chunk_start", -1)) != target:
            logger.warning(
                "batch cache env_step_at_chunk_start mismatch (cached=%s, want=%s); ignoring %s",
                blob.get("env_step_at_chunk_start"), target, path,
            )
            return None
        batch: dict[str, Any] = {}
        for k, v in blob["batch"].items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
            else:
                batch[k] = v
        return batch

    def _advance_to_chunk(self) -> dict[str, torch.Tensor]:
        """Replay env up to the chunk's env_step, return the policy-ready batch."""
        env = self._env
        target = int(self.rollout.env_step_at_chunk_start[self.chunk_idx])
        chunk_actions = self.rollout.chunk_actions  # (n_chunks, B_orig, chunk_size, A_model)
        K = self.n_action_steps

        # B=1: feed only this slot's actions.
        obs, _info = env.reset(seed=[self.seed])
        for t in range(target):
            k, s = divmod(t, K)
            a_model = chunk_actions[k, self.env_idx : self.env_idx + 1, s, :]  # (1, A_model)
            a_tensor = torch.from_numpy(np.ascontiguousarray(a_model)).to(self.device)
            a_tensor = self._postprocessor(a_tensor)
            transition = self._env_post({self._ACTION: a_tensor})
            a_env = transition[self._ACTION]
            a_env_np = a_env.to("cpu").numpy() if hasattr(a_env, "cpu") else np.asarray(a_env)
            obs, _r, term, trunc, _info = env.step(a_env_np)
            done = bool(np.asarray(term).any() or np.asarray(trunc).any())
            if done and t + 1 < target:
                # Sanity: episode ended early during replay, before reaching the
                # target chunk. This shouldn't happen given the first-episode
                # guard in __init__, but if MuJoCo / postprocess drift produces a
                # different early-done than the recording, surface it loudly.
                raise RuntimeError(
                    f"Env auto-reset during replay at env_step={t + 1} before reaching "
                    f"chunk start env_step={target}. Recording-vs-replay divergence: "
                    "the (rollout, env_idx, chunk_idx) likely isn't actually within the "
                    "first episode for this seed."
                )

        # Build the batch the same way lerobot_eval.rollout does.
        observation = self._preprocess_observation(obs)
        try:
            observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            try:
                observation["task"] = list(env.call("task"))
            except (AttributeError, NotImplementedError):
                observation["task"] = [""]
        observation = self._env_pre(observation)
        observation = self._preprocessor(observation)
        return observation

    # ----------------------------------------------------------- stored slice
    @property
    def stored_chunk_actions(self) -> torch.Tensor:
        """(chunk_size, action_dim) tensor — the recorded chunk for this slot/chunk."""
        return torch.from_numpy(
            np.asarray(self.rollout.chunk_actions[self.chunk_idx, self.env_idx])
        )

    @property
    def stored_noise(self) -> torch.Tensor:
        """(chunk_size, max_action_dim) — the noise fed to the FM head in the recording."""
        return torch.from_numpy(
            np.asarray(self.rollout.noise[self.chunk_idx, self.env_idx])
        )

    @property
    def stored_x_t(self) -> torch.Tensor:
        """(T+1, chunk_size, max_action_dim) — the recorded denoise trajectory."""
        return torch.from_numpy(
            np.asarray(self.rollout.x_t[self.chunk_idx, :, self.env_idx])
        )

    @property
    def stored_v_t(self) -> torch.Tensor:
        """(T, chunk_size, max_action_dim) — the recorded velocities per denoise step."""
        return torch.from_numpy(
            np.asarray(self.rollout.v_t[self.chunk_idx, :, self.env_idx])
        )

    @property
    def stored_times(self) -> torch.Tensor:
        """(T,) — denoise timesteps (1 = pure noise → 0 = action)."""
        return torch.from_numpy(np.asarray(self.rollout.time[self.chunk_idx]))

    # ---------------------------------------------------------- prefix cache
    def _load_context_slot(self) -> None:
        """Materialize the recorded embed_prefix inputs for (chunk_idx, env_idx)
        as B=1 device tensors, mirroring `_images_tokens_masks(1)`. Sets
        `self._ctx_slot`, consumed by `_compute_prefix_cache`."""
        ctx = self.rollout.context(self.env_idx, self.chunk_idx)
        if ctx is None:
            raise RuntimeError("_load_context_slot called but rollout has no context.")
        dev = self.device
        imgs = ctx["images"]      # (ncam, 3, H, W)
        imgm = ctx["img_masks"]   # (ncam,)
        images = [
            torch.from_numpy(np.ascontiguousarray(imgs[c])).to(dev, torch.float32).unsqueeze(0)
            for c in range(imgs.shape[0])
        ]
        img_masks = [
            torch.from_numpy(np.atleast_1d(np.ascontiguousarray(imgm[c]))).to(dev)
            for c in range(imgm.shape[0])
        ]
        tokens = torch.from_numpy(np.ascontiguousarray(ctx["tokens"])).to(dev).unsqueeze(0)
        masks = torch.from_numpy(np.ascontiguousarray(ctx["masks"])).to(dev).unsqueeze(0)
        self._ctx_slot = (images, img_masks, tokens, masks)

    @torch.no_grad()
    def _compute_prefix_cache(self) -> None:
        """Run paligemma_with_expert.forward at B=1 once; stash the (B=1)
        prefix_pad_masks and past_key_values.

        Mirrors the prefix path of `PI05Pytorch.sample_actions`. Every later
        call to `_run_micro_batch` / `reproduce` expands these to the
        micro-batch via `.expand()` (allocation-free) and runs only the FM
        denoise loop.
        """
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        model = self._policy.model
        if self._ctx_slot is not None:
            images, img_masks, tokens, masks = self._ctx_slot
        else:
            images, img_masks, tokens, masks = self._images_tokens_masks(batch_size=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
            "eager"
        )
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        self._prefix_pad_masks_1 = prefix_pad_masks
        self._past_key_values_1 = past_key_values
        n_layers = len(past_key_values.layers)
        s_prefix = int(prefix_pad_masks.shape[1])
        logger.info(
            "prefix cache built: B=1 prefix_len=%d num_layers=%d",
            s_prefix,
            n_layers,
        )

    def _expanded_prefix(self, batch_size: int):
        """Return (prefix_pad_masks_B, past_key_values_B) — both are .expand
        views over the B=1 cache. `denoise_step` deepcopies past_key_values
        internally, so the materialization happens at most once per denoise
        step rather than once per prefix forward.
        """
        assert self._past_key_values_1 is not None
        assert self._prefix_pad_masks_1 is not None
        pkv = self._past_key_values_1
        # Shallow-copy the cache shell + each layer object, then swap in
        # expanded K/V views. Avoids mutating the cached B=1 originals.
        new_layers = []
        for layer in pkv.layers:
            new_layer = copy.copy(layer)
            if getattr(layer, "is_initialized", True) and layer.keys.numel() > 0:
                new_layer.keys = layer.keys.expand(batch_size, -1, -1, -1)
                new_layer.values = layer.values.expand(batch_size, -1, -1, -1)
            new_layers.append(new_layer)
        # DynamicCache.__init__ doesn't accept a `layers` kwarg in this
        # transformers version; construct an empty cache of the same class then
        # swap in our expanded layers.
        new_pkv = type(pkv)()
        new_pkv.layers = new_layers
        pad_b = self._prefix_pad_masks_1.expand(batch_size, -1)
        return pad_b, new_pkv

    # ---------------------------------------------------------- model running
    def _ensure_setup(self) -> None:
        if not self._setup_done:
            raise RuntimeError("Call .setup() (or use as a context manager) first.")

    def _images_tokens_masks(
        self, batch_size: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run the policy's image prep on a B-repeated copy of the captured batch."""
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

        assert self._batch is not None
        batch_b = self._repeat_batch(self._batch, batch_size)
        images, img_masks = self._policy._preprocess_images(batch_b)
        tokens = batch_b[OBS_LANGUAGE_TOKENS]
        masks = batch_b[OBS_LANGUAGE_ATTENTION_MASK]
        return images, img_masks, tokens, masks

    @staticmethod
    def _repeat_batch(batch: dict[str, Any], n: int) -> dict[str, Any]:
        """Return a shallow copy of `batch` with each tensor entry expanded
        along dim 0 to size `n`. Non-tensor entries (e.g. 'task' list) are
        replicated. Uses `expand` for tensors so this is allocation-free."""
        out: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.shape[0] == n:
                    out[k] = v
                elif v.shape[0] == 1:
                    expand_shape = (n,) + tuple(-1 for _ in range(v.ndim - 1))
                    out[k] = v.expand(*expand_shape)
                else:
                    raise ValueError(
                        f"Batch tensor {k!r} has leading dim {v.shape[0]}; cannot expand to {n}."
                    )
            elif isinstance(v, list):
                if len(v) == 1:
                    out[k] = v * n
                elif len(v) == n:
                    out[k] = v
                else:
                    raise ValueError(
                        f"Batch list {k!r} has length {len(v)}; cannot replicate to {n}."
                    )
            else:
                out[k] = v
        return out

    @torch.no_grad()
    def _fm_denoise_loop(
        self, noise: torch.Tensor, *, capture_trajectory: bool
    ) -> dict[str, Any]:
        """Run only the FM denoise loop using the cached B=1 prefix expanded
        to noise's batch size. Replaces `model.sample_actions` for the
        resample/reproduce paths so we never re-run the costly prefix forward
        (which materializes a (B, heads, S, S) fp32 attention matrix in the
        eager attention impl — the original OOM source).
        """
        model = self._policy.model
        b = int(noise.shape[0])
        pad_b, pkv_b = self._expanded_prefix(b)

        num_steps = int(self.num_inference_steps)
        dt = -1.0 / num_steps
        x_t = noise

        cap_times: list[float] = []
        cap_x_t: list[torch.Tensor] = []
        cap_v_t: list[torch.Tensor] = []

        for step in range(num_steps):
            t = 1.0 + step * dt
            timestep = torch.tensor(t, dtype=torch.float32, device=self.device).expand(b)
            if capture_trajectory:
                cap_x_t.append(x_t.detach().to("cpu", torch.float32))
                cap_times.append(t)
            v_t = model.denoise_step(
                prefix_pad_masks=pad_b,
                past_key_values=pkv_b,
                x_t=x_t,
                timestep=timestep,
            )
            if capture_trajectory:
                cap_v_t.append(v_t.detach().to("cpu", torch.float32))
            x_t = x_t + dt * v_t

        # Match stored_x_t (T+1) by appending the post-last-step x at t=0.
        if capture_trajectory:
            cap_x_t.append(x_t.detach().to("cpu", torch.float32))

        chunk = x_t[:, :, : self._action_dim].detach().to("cpu", torch.float32)
        out: dict[str, Any] = {"chunk_actions": chunk}
        if capture_trajectory:
            out["x_t_in"] = torch.stack(cap_x_t, dim=1)
            out["v_t"] = torch.stack(cap_v_t, dim=1)
            out["times"] = torch.tensor(cap_times, dtype=torch.float32)
        return out

    @torch.no_grad()
    def reproduce(self, *, capture_trajectory: bool = False) -> dict[str, Any]:
        """Re-run the FM head with the *stored* noise; compare to stored chunk.

        Returns dict with:
          chunk_actions:    (chunk_size, action_dim) — newly produced chunk
          stored_chunk:     (chunk_size, action_dim) — what the recording has
          max_abs_err:      scalar — over all (chunk_size, action_dim) entries
          mean_abs_err:     scalar
          x_t / v_t / times: if capture_trajectory (matched against stored_x_t etc.)
        """
        self._ensure_setup()
        noise = self.stored_noise.to(self.device, dtype=torch.float32).unsqueeze(0)
        res = self._fm_denoise_loop(noise, capture_trajectory=capture_trajectory)
        chunk = res["chunk_actions"].squeeze(0)
        stored = self.stored_chunk_actions.to(torch.float32)
        diff = (chunk - stored).abs()
        out: dict[str, Any] = {
            "chunk_actions": chunk,
            "stored_chunk": stored,
            "max_abs_err": float(diff.max().item()),
            "mean_abs_err": float(diff.mean().item()),
        }
        if capture_trajectory:
            out["times"] = res["times"]
            out["x_t_in"] = res["x_t_in"].squeeze(0)  # drop the B=1 dim
            out["v_t"] = res["v_t"].squeeze(0)
        return out

    def _make_noise_gen(self, seed: int | None) -> torch.Generator:
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        return gen

    def _sample_noise_micro(
        self, b: int, gen: torch.Generator | None
    ) -> torch.Tensor:
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=(b, self.chunk_size, self.max_action_dim),
            generator=gen,
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def _run_micro_batch(
        self, noise_b: torch.Tensor, *, capture_trajectory: bool
    ) -> dict[str, Any]:
        """One forward at batch=b through the FM denoise loop only — prefix is
        reused from the B=1 cache built in setup(). Returns CPU float32
        tensors so the caller can free GPU state immediately."""
        res = self._fm_denoise_loop(noise_b, capture_trajectory=capture_trajectory)
        out: dict[str, Any] = {"chunk_actions": res["chunk_actions"]}
        if capture_trajectory:
            out["x_t_in"] = res["x_t_in"]
            out["v_t"] = res["v_t"]
            out["times"] = res["times"]
        return out

    @torch.no_grad()
    def resample(
        self,
        n_samples: int,
        *,
        noises: torch.Tensor | None = None,
        seed: int | None = None,
        capture_trajectory: bool = True,
        micro_batch: int | None = None,
    ) -> dict[str, Any]:
        """Re-run the FM head with N freshly drawn noises (or user-supplied ones).

        Accumulates all N samples on CPU before returning. Host RAM peak ≈
        `n_samples * per_sample_bytes` (~136 KB/sample with trajectory, ~8 KB
        without); GPU peak is bounded by `micro_batch`.

        Args:
          n_samples: how many chunks to sample. Ignored if `noises` is given.
          noises: optional pre-sampled noises of shape (N, chunk_size, max_action_dim).
            Use this to e.g. include the stored noise as sample 0, or to sweep
            structured perturbations.
          seed: RNG seed for the new noises. Independent torch.Generator on
            self.device; deterministic across runs given the same seed.
          capture_trajectory: also collect (times, x_t, v_t) per denoise step.
            Adds ~T small CPU copies per micro-batch — keep on for analysis,
            turn off if you only want chunks (~17× cheaper on disk/RAM).
          micro_batch: forward at most this many samples at a time through the
            transformer to bound GPU memory (default: no chunking).

        Returns dict with:
          noises:        (N, chunk_size, max_action_dim) float32 cpu
          chunk_actions: (N, chunk_size, action_dim)     float32 cpu
          times:         (T,)                            (if capture_trajectory)
          x_t_in:        (N, T+1, chunk_size, max_action_dim)
          v_t:           (N, T,   chunk_size, max_action_dim)
        `x_t_in[:, -1]` is the post-last-step value at t=0 (matches stored_x_t).
        """
        self._ensure_setup()

        provided = noises is not None
        if provided:
            if noises.ndim != 3 or noises.shape[1:] != (self.chunk_size, self.max_action_dim):
                raise ValueError(
                    f"noises must be (N, {self.chunk_size}, {self.max_action_dim}); got {tuple(noises.shape)}"
                )
            n_samples = int(noises.shape[0])
        else:
            if n_samples <= 0:
                raise ValueError("n_samples must be > 0 when noises is None.")

        mb = int(micro_batch) if micro_batch else n_samples
        if mb <= 0:
            raise ValueError("micro_batch must be > 0.")

        gen = self._make_noise_gen(seed) if not provided else None
        all_noises: list[torch.Tensor] = []
        all_chunks: list[torch.Tensor] = []
        all_x_t: list[torch.Tensor] = []
        all_v_t: list[torch.Tensor] = []
        all_times: torch.Tensor | None = None

        for start in range(0, n_samples, mb):
            end = min(start + mb, n_samples)
            b = end - start
            if provided:
                noise_b = noises[start:end].to(self.device, dtype=torch.float32).contiguous()
            else:
                noise_b = self._sample_noise_micro(b, gen)
            res = self._run_micro_batch(noise_b, capture_trajectory=capture_trajectory)
            all_noises.append(noise_b.detach().to("cpu", torch.float32))
            all_chunks.append(res["chunk_actions"])
            if capture_trajectory:
                all_x_t.append(res["x_t_in"])
                all_v_t.append(res["v_t"])
                if all_times is None:
                    all_times = res["times"]
            del noise_b, res  # drop GPU tensors before next iter

        out: dict[str, Any] = {
            "noises": torch.cat(all_noises, dim=0),
            "chunk_actions": torch.cat(all_chunks, dim=0),
        }
        if capture_trajectory:
            out["times"] = all_times
            out["x_t_in"] = torch.cat(all_x_t, dim=0)
            out["v_t"] = torch.cat(all_v_t, dim=0)
        return out

    # -------------------------------------------------------- future hook
    def rollout_remainder(
        self,
        override_chunk_actions: torch.Tensor,  # (chunk_size, action_dim) or (n_action_steps, action_dim)
        *,
        max_steps: int | None = None,
        return_observations: bool = False,
    ) -> dict[str, Any]:
        """**Not implemented.** Continue env stepping from the chunk position
        using `override_chunk_actions` for the first `n_action_steps` steps,
        then let the policy roll out normally until done or max_steps.

        Planned design (see also `fmaccel/replay.py` for the env-stepping
        template and `lerobot.scripts.lerobot_eval.rollout` for the policy
        loop):

        1. Snapshot env state at chunk position. (We currently advance the env
           in `setup()` and leave it parked there; for per-sample remainders
           we'll need to either (a) reset+replay to position for each call —
           simple but ~O(target_env_step) per call — or (b) use MuJoCo's
           `env.unwrapped.sim.get_state() / set_state()` to snapshot once and
           restore per call.
        2. Step env n_action_steps with `override_chunk_actions[:n_action_steps]`
           via the same postprocessor chain used in `_advance_to_chunk`.
        3. From that point, `policy.reset()` is wrong (would clobber any
           cached prefix); instead, drain the policy's `_action_queue`, then
           loop: build batch from current obs (same pipeline as setup), call
           `policy.predict_action_chunk(batch)` for each new chunk, step env
           n_action_steps with the first n_action_steps of that chunk.
        4. Stop on `terminated|truncated` or after `max_steps` env steps.
        5. Return `{actions, reward, terminated, truncated, success, seeds,
           observations?}` — mirroring `LiberoReplayer.replay_actions`'s
           return shape so downstream code can reuse render/analysis utils.
        """
        raise NotImplementedError(
            "rollout_remainder is a planned extension — see the docstring for "
            "the intended design. Implement when needed; the rest of this "
            "session intentionally leaves the env open for snapshotting."
        )

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        if self._envs is not None:
            from lerobot.envs import close_envs
            close_envs(self._envs)
            self._envs = None
            self._env = None
        # Don't drop an injected (shared) policy — its owner still uses it.
        if not self._injected_policy:
            self._policy = None
        self._batch = None
        self._setup_done = False


def resample_to_npz(
    recording_path: str,
    rollout_idx: int,
    env_idx: int,
    chunk_idx: int,
    output_path: str,
    *,
    n_samples: int,
    seed: int | None = None,
    capture_trajectory: bool = True,
    micro_batch: int | None = None,
    device: str = "cuda",
    obs_size: int = 360,
    num_inference_steps: int | None = None,
    include_reproduce: bool = True,
    batch_cache_dir: Path | str | None = None,
    use_batch_cache: bool = True,
    use_context: bool = True,
    reproduce_tol: float = 5e-2,
) -> dict[str, Any]:
    """Convenience: load → session → reproduce + resample → write one npz.

    Returns the in-memory dict written to disk (without the heavy arrays
    duplicated — useful as a smoke-test return).
    """
    recording = FMRecording.load(recording_path)
    payload: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {
        "recording_path": str(recording_path),
        "rollout_idx": int(rollout_idx),
        "env_idx": int(env_idx),
        "chunk_idx": int(chunk_idx),
        "n_samples": int(n_samples),
        "seed": -1 if seed is None else int(seed),
        "capture_trajectory": bool(capture_trajectory),
        "device": device,
        "obs_size": int(obs_size),
    }

    with ChunkResampleSession(
        recording,
        rollout_idx=rollout_idx,
        env_idx=env_idx,
        chunk_idx=chunk_idx,
        device=device,
        obs_size=obs_size,
        num_inference_steps=num_inference_steps,
        batch_cache_dir=batch_cache_dir,
        use_batch_cache=use_batch_cache,
        use_context=use_context,
    ) as sess:
        meta["used_context"] = bool(sess._use_context)
        # Stored slice — saved alongside resamples so the npz is self-contained.
        payload["stored_chunk_actions"] = sess.stored_chunk_actions.numpy()
        payload["stored_noise"] = sess.stored_noise.numpy()
        payload["stored_x_t"] = sess.stored_x_t.numpy()
        payload["stored_v_t"] = sess.stored_v_t.numpy()
        payload["stored_times"] = sess.stored_times.numpy()
        meta["task_id"] = sess.task_id
        meta["task_group"] = sess.suite_name
        meta["seed_env"] = sess.seed
        meta["env_step_at_chunk_start"] = int(
            sess.rollout.env_step_at_chunk_start[chunk_idx]
        )
        meta["task_desc"] = str(sess.rollout.task_descs[chunk_idx, env_idx])
        meta["action_dim"] = int(sess._action_dim)
        meta["chunk_size"] = sess.chunk_size
        meta["max_action_dim"] = sess.max_action_dim
        meta["num_inference_steps"] = sess.num_inference_steps

        if include_reproduce:
            rep = sess.reproduce(capture_trajectory=capture_trajectory)
            payload["reproduce_chunk_actions"] = rep["chunk_actions"].numpy()
            meta["reproduce_max_abs_err"] = rep["max_abs_err"]
            meta["reproduce_mean_abs_err"] = rep["mean_abs_err"]
            meta["reproduce_tol"] = float(reproduce_tol)
            meta["reproduce_ok"] = bool(rep["max_abs_err"] <= reproduce_tol)
            if capture_trajectory:
                payload["reproduce_x_t_in"] = rep["x_t_in"].numpy()
                payload["reproduce_v_t"] = rep["v_t"].numpy()
                payload["reproduce_times"] = rep["times"].numpy()
            logger.info(
                "reproduce: max_abs_err=%.3e mean_abs_err=%.3e (used_context=%s)",
                rep["max_abs_err"], rep["mean_abs_err"], meta.get("used_context"),
            )
            if not meta["reproduce_ok"]:
                logger.warning(
                    "REPRODUCE GATE FAILED: max_abs_err=%.3e > tol=%.1e. The resample is "
                    "conditioned on a DRIFTED observation (used_context=%s) — posterior / "
                    "multimodality analysis on this file is untrustworthy. Re-record with "
                    "capture_context=True and resample with use_context=True.",
                    rep["max_abs_err"], reproduce_tol, meta.get("used_context"),
                )

        if n_samples > 0:
            res = sess.resample(
                n_samples=n_samples,
                seed=seed,
                capture_trajectory=capture_trajectory,
                micro_batch=micro_batch,
            )
            payload["resample_noises"] = res["noises"].numpy()
            payload["resample_chunk_actions"] = res["chunk_actions"].numpy()
            if capture_trajectory:
                payload["resample_x_t_in"] = res["x_t_in"].numpy()
                payload["resample_v_t"] = res["v_t"].numpy()
                payload["resample_times"] = res["times"].numpy()

    import json
    import pathlib
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, meta_json=np.asarray(json.dumps(meta)), **payload)
    logger.info("wrote %s (n_samples=%d, keys=%d)", out_path, n_samples, len(payload))
    return {"meta": meta, "keys": list(payload.keys())}
