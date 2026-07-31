"""The dataset/env adapter contract.

Abstracts the three execution modes the repo runs:
  * ``"vec_env"``      — a lerobot vectorized env + ``eval_policy_all`` (LIBERO).
  * ``"zmq"``          — a sim in a separate process/environment, driven over a ZMQ bridge
                          (the policy server runs alongside the policy, the sim in its own
                          environment).
  * ``"teacher_forced"`` — replay a dataset's held-out states through the policy
                          (no env), used for toy FM recording.

The eval pipeline dispatches on :attr:`execution_mode`. For ``vec_env`` an adapter
must provide ``env_config`` / ``build_envs`` / ``build_env_processors`` / ``close``;
the other modes implement their own driver (wired in a later phase). Keep heavy
backend imports inside the methods so the contract stays import-light.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence


@dataclass
class TaskSpec:
    task_id: int
    task_group: str
    description: str | None = None


class DatasetAdapter(ABC):
    name: ClassVar[str] = "?"
    action_dim: ClassVar[int] = 0
    execution_mode: ClassVar[str] = "vec_env"

    def __init__(self, *, task_group: str | None = None,
                 task_ids: Sequence[int] | None = None, render_size: int = 360) -> None:
        self.task_group = task_group
        self.task_ids = list(task_ids) if task_ids is not None else None
        self.render_size = int(render_size)

    @property
    def run_slug(self) -> str:
        """Slug used in the run-id (``<date>_<model>_<slug>_<tag>``)."""
        return self.task_group or self.name

    def list_tasks(self) -> list[TaskSpec]:
        raise NotImplementedError

    # ---- teacher_forced labeling (label_accel) ----------------------------
    def episode_len(self, ep: dict) -> int:
        """Number of labelable frames in a demo-episode dict (default: rows of
        ``state_raw``). Image datasets that carry their own length can override."""
        return len(ep["state_raw"])

    def chunk_inputs(self, ep: dict, t: int, samples: int = 1) -> dict:
        """Raw policy obs for the chunk launched at frame ``t`` of ``ep``, repeated
        ``samples`` times (same observation, fresh FM noise per row). Default: the
        state-only conditioning the state toys consume directly; image policies (π₀.₅
        on LIBERO) override to assemble the camera observation. Whatever model-specific
        finishing (normalize/tokenize/device) is needed is the model adapter's
        :meth:`~fmaccel.models.base.FMModelAdapter.preprocess`."""
        import numpy as np

        state = ep["state_raw"][t]
        if samples > 1:
            state = np.repeat(np.asarray(state, np.float32)[None], samples, axis=0)
        return {"observation.state": state, "task": [self.task_desc(ep["state_raw"][t])] * samples}

    # ---- vec_env mode -----------------------------------------------------
    def env_config(self) -> Any:
        raise NotImplementedError(f"{self.name} is not a vec_env dataset")

    def build_envs(self, *, batch_size: int, use_async_envs: bool) -> Any:
        raise NotImplementedError(f"{self.name} is not a vec_env dataset")

    def build_env_processors(self, env_cfg: Any, policy_cfg: Any) -> tuple[Any, Any]:
        raise NotImplementedError(f"{self.name} is not a vec_env dataset")

    def close(self, envs: Any) -> None:
        pass
