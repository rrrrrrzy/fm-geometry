"""``LiberoDataset`` — LIBERO via lerobot's vectorized env (``vec_env`` mode).

Owns the LIBERO/lerobot env assembly (``EnvConfig`` choice "libero" + ``make_env``
+ ``make_env_pre_post_processors``); the model-policy assembly lives in the model
adapter's ``build_eval``. Imports lerobot, so resolved only via
``get_dataset("libero")`` under the project environment.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from fmaccel.datasets.base import DatasetAdapter, TaskSpec

logger = logging.getLogger(__name__)

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")

# The 4 standard eval suites the aggregate "libero_all" group expands to (the ones
# scripts/build_libero_all_dataset.py merges into libero_all_v30_grip). Excludes libero_90.
# The LIBERO env's create_envs accepts a comma-separated `task` string, so eval on the
# aggregate needs no per-suite loop — one env dict spans all four.
LIBERO_ALL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIBERO_ALL = "libero_all"

# π₀.₅ base / base-finetuned camera convention: the policy config declares
# {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}. The LIBERO env natively exposes the
# agentview + eye-in-hand cameras (default keys observation.images.image / .image2); this
# mapping renames them to the policy's base_0_rgb / left_wrist_0_rgb so the env features are
# a subset of the policy's (right_wrist_0_rgb has no LIBERO source and is empty-padded by
# the model at inference). It is the EVAL counterpart of the fine-tune's --rename_map
# (image→base_0_rgb, wrist_image→left_wrist_0_rgb), so closed-loop eval feeds the policy
# exactly the camera wiring it was trained on.
#
# But the *openpi-converted* pi05-libero-hf checkpoint keeps the env-native keys
# (image / image2 / empty_camera_0). One hard-coded mapping can therefore only fit one of
# the two checkpoint families; the wrong one makes make_policy raise a visual-feature
# mismatch. So the default is "auto" — resolve_camera_mapping() reads the checkpoint config
# and picks the wiring that matches it (see that method).
PI05_CAMERA_MAPPING = {"agentview_image": "base_0_rgb",
                       "robot0_eye_in_hand_image": "left_wrist_0_rgb"}

_OBS_IMG_PREFIX = "observation.images."


class LiberoDataset(DatasetAdapter):
    name = "libero"
    action_dim = 7
    execution_mode = "vec_env"

    def __init__(self, *, task_group: str | None = None,
                 task_ids: Sequence[int] | None = None, render_size: int = 360,
                 camera_name_mapping: dict[str, str] | str | None = "auto") -> None:
        task_group = task_group or "libero_spatial"
        if task_group not in SUITES and task_group != LIBERO_ALL:
            raise ValueError(
                f"--task-group must be one of {SUITES} or {LIBERO_ALL!r}; got {task_group!r}")
        # Accepted values:
        #   "auto" (default) → resolved from the checkpoint at eval time (resolve_camera_mapping);
        #                      starts env-native until then.
        #   "pi05"           → PI05_CAMERA_MAPPING (base / base-finetuned checkpoints).
        #   None / "native"  → env-native image / image2 keys (pi05-libero-hf).
        #   custom dict      → robosuite-cam → suffix, passed through verbatim.
        self._camera_mapping_arg = camera_name_mapping
        self.camera_name_mapping = self._coerce_mapping(camera_name_mapping)
        # A single suite defaults to task 0 (historical smoke default); the aggregate
        # "libero_all" defaults to EVERY task of EVERY suite (task_ids=None → env selects all).
        if task_ids is not None:
            default_ids: list[int] | None = list(task_ids)
        elif task_group == LIBERO_ALL:
            default_ids = None
        else:
            default_ids = [0]
        super().__init__(task_group=task_group, task_ids=default_ids, render_size=render_size)

    @staticmethod
    def _coerce_mapping(m: dict[str, str] | str | None) -> dict[str, str] | None:
        if m == "pi05":
            return dict(PI05_CAMERA_MAPPING)
        if m in (None, "native", "auto"):
            return None  # "auto" starts env-native; resolve_camera_mapping refines it
        return m

    @staticmethod
    def _exterior_wrist_suffixes(input_features: dict[str, Any]) -> tuple[str | None, str | None]:
        """``(exterior, wrist)`` image-key suffixes a checkpoint declares: the first two
        VISUAL features after dropping the model-padded empty / right-wrist slots, with the
        ``observation.images.`` prefix stripped. Mirrors ``Pi05Adapter._expected_image_slots``
        but reads the raw (weights-free) checkpoint config rather than a loaded policy."""
        keys = [k for k, v in input_features.items()
                if "VISUAL" in str(getattr(v, "type", "")) and
                "empty" not in k and "right_wrist" not in k]
        sfx = [k[len(_OBS_IMG_PREFIX):] if k.startswith(_OBS_IMG_PREFIX) else k for k in keys]
        return (sfx[0] if sfx else None, sfx[1] if len(sfx) > 1 else None)

    def resolve_camera_mapping(self, checkpoint: str | None) -> dict[str, str] | None:
        """When constructed with ``camera_name_mapping="auto"`` (the default), pick the
        env→policy camera wiring from the *checkpoint's* declared image slots, so the env
        emits exactly the visual keys ``make_policy`` validates against:
          - pi05-libero-hf (openpi-converted) declares image/image2 → env-native (None)
          - base / base-finetuned declare base_0_rgb/left_wrist_0_rgb → PI05_CAMERA_MAPPING
        No-op unless the arg was "auto". Call once before ``env_config()``; on any failure
        it warns and leaves the env-native default so make_policy can raise a clear error."""
        if self._camera_mapping_arg != "auto" or checkpoint is None:
            return self.camera_name_mapping
        ext = wr = None
        try:
            import lerobot.policies  # noqa: F401  # registers the pi05 config subclass
            from lerobot.configs import PreTrainedConfig
            cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
            ext, wr = self._exterior_wrist_suffixes(cfg.input_features or {})
        except Exception as e:  # malformed/missing config → fall back to env-native
            logger.warning("libero: camera-mapping auto-detect failed for %s (%s); "
                           "using env-native image/image2.", checkpoint, e)
        if ext and wr and (ext, wr) != ("image", "image2"):
            self.camera_name_mapping = {"agentview_image": ext, "robot0_eye_in_hand_image": wr}
        else:
            self.camera_name_mapping = None
        logger.info("libero: camera mapping auto-resolved from %s -> %s",
                    checkpoint, self.camera_name_mapping or "env-native (image/image2)")
        return self.camera_name_mapping

    @property
    def _env_suites(self) -> tuple[str, ...]:
        """The suite name(s) this group expands to. ``libero_all`` → the 4 standard suites
        (LIBERO env's create_envs joins them from a comma-separated ``task`` string);
        any single suite → itself."""
        return LIBERO_ALL_SUITES if self.task_group == LIBERO_ALL else (self.task_group,)

    def list_tasks(self) -> list[TaskSpec]:
        # Explicit task_ids: one TaskSpec per id (per suite for the aggregate). None (the
        # aggregate default = all tasks): enumerate each suite's full task list via the
        # LIBERO benchmark (this method is env-side, so importing libero here is fine).
        if self.task_ids is not None:
            return [TaskSpec(task_id=t, task_group=g)
                    for g in self._env_suites for t in self.task_ids]
        from lerobot.envs.libero import _get_suite
        specs: list[TaskSpec] = []
        for g in self._env_suites:
            for tid in range(len(_get_suite(g).tasks)):
                specs.append(TaskSpec(task_id=tid, task_group=g))
        return specs

    def env_config(self) -> Any:
        from lerobot.envs.configs import EnvConfig
        cls = EnvConfig.get_choice_class("libero")
        return cls(
            task=",".join(self._env_suites),          # env splits comma-separated suites
            task_ids=list(self.task_ids) if self.task_ids is not None else None,
            observation_height=self.render_size,
            observation_width=self.render_size,
            camera_name_mapping=self.camera_name_mapping,
        )

    def build_envs(self, *, batch_size: int, use_async_envs: bool) -> Any:
        from lerobot.envs import make_env
        return make_env(self.env_config(), n_envs=batch_size, use_async_envs=use_async_envs)

    def build_env_processors(self, env_cfg: Any, policy_cfg: Any) -> tuple[Any, Any]:
        from lerobot.envs import make_env_pre_post_processors
        return make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    def close(self, envs: Any) -> None:
        from lerobot.envs import close_envs
        close_envs(envs)
