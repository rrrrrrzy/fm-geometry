"""Eval pipeline: run a policy on a benchmark, recording the FM denoising trajectory.

This is the *producer* of everything downstream: with ``--record-fm --record-context`` it writes
every Euler iterate of every decision, plus the exact conditioning each decision saw, into one
self-describing run directory (``outputs/runs/<run-id>/``: ``eval/`` + ``fm/`` + ``videos/``).
Every score in the paper is then computed post-hoc from that directory — no policy needed.

Dispatches on ``dataset.execution_mode``. The ``vec_env`` path (LIBERO) wraps lerobot's
``eval_policy_all``, with the env assembly coming from the dataset adapter and the policy
assembly from the model adapter's ``build_eval``; ``teacher_forced`` replays a demo stream
instead of rolling out.

Note ``--disable-compile`` is required alongside ``--record-fm``: ``torch.compile`` captures the
graph before the recording hooks fire, so a compiled action-sampling path records nothing.
"""

from __future__ import annotations

import logging
from typing import Any

from fmaccel.core import runs
from fmaccel.core.io import write_json
from fmaccel.recording.recorder import FMRecorder
from fmaccel.registry import get_model

logger = logging.getLogger(__name__)


def run_eval(
    *,
    model: str,
    dataset: Any,                 # a built DatasetAdapter
    checkpoint: str | None = None,
    device: str = "cuda",
    seed: int = 1000,
    tag: str | None = None,
    n_episodes: int = 1,
    batch_size: int = 1,
    use_async_envs: bool | None = None,
    max_episodes_rendered: int = 1,
    max_parallel_tasks: int = 1,
    n_action_steps: int | None = 10,
    num_inference_steps: int | None = None,
    disable_compile: bool = False,
    compile_mode: str = "max-autotune",
    record_fm: bool = False,
    record_context: bool = False,
    file_prefix: str = "",
    worker_id: str | None = None,
    into_run: str | None = None,
) -> runs.RunDir:
    if dataset.execution_mode == "teacher_forced":
        return _run_teacher_forced(
            model=model, dataset=dataset, checkpoint=checkpoint, device=device, seed=seed, tag=tag,
            num_inference_steps=num_inference_steps, record_fm=record_fm, record_context=record_context,
            file_prefix=file_prefix, worker_id=worker_id,
            max_rollouts=None,
        )
    if dataset.execution_mode != "vec_env":
        raise NotImplementedError(
            f"this release ships the vec_env (LIBERO closed-loop) and teacher_forced paths; "
            f"{dataset.name} is '{dataset.execution_mode}'."
        )

    import torch
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.utils.random_utils import set_seed

    if record_fm and not disable_compile:
        logger.warning("--record-fm forces --disable-compile (compile bypasses the FM hooks).")
        disable_compile = True

    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create the run dir EARLY (or join an existing shared run for multi-GPU shards).
    if into_run is not None:
        run = runs.resolve_run(into_run)
    else:
        run = runs.create_run(
            model={"name": model, "checkpoint": str(checkpoint) if checkpoint else None},
            dataset={"name": dataset.name, "task_group": dataset.task_group,
                     "task_ids": dataset.task_ids, "run_slug": dataset.run_slug},
            stage="eval", tag=tag,
            args={"n_episodes": n_episodes, "batch_size": batch_size, "seed": seed,
                  "n_action_steps": n_action_steps, "num_inference_steps": num_inference_steps,
                  "render_size": dataset.render_size, "disable_compile": disable_compile,
                  "record_fm": record_fm, "record_context": record_context,
                  "worker_id": worker_id},
        )
    logger.info("run: %s -> %s", run.run_id, run.root)
    worker_tag = (file_prefix.rstrip("_") or worker_id) if into_run is not None else None

    ModelAdapter = get_model(model)
    if use_async_envs is None:
        use_async_envs = batch_size > 1

    # LIBERO: the env's camera keys must match the checkpoint's declared image slots for
    # make_policy's visual-feature validation (image/image2 for pi05-libero-hf vs
    # base_0_rgb/left_wrist_0_rgb for base-finetuned). Auto-resolve the env↔ckpt wiring
    # from the checkpoint config (cheap, weights-free) BEFORE building the env.
    resolve_cams = getattr(dataset, "resolve_camera_mapping", None)
    if resolve_cams is not None:
        ckpt_for_cams = (ModelAdapter.resolve_checkpoint(checkpoint)
                         if hasattr(ModelAdapter, "resolve_checkpoint") else checkpoint)
        resolve_cams(ckpt_for_cams)

    env_cfg = dataset.env_config()
    envs = dataset.build_envs(batch_size=batch_size, use_async_envs=use_async_envs)
    adapter = ModelAdapter.build_eval(
        checkpoint, env_cfg=env_cfg, device=device, action_dim=dataset.action_dim,
        n_action_steps=n_action_steps, num_inference_steps=num_inference_steps,
        disable_compile=disable_compile, compile_mode=compile_mode,
    )
    env_pre, env_post = dataset.build_env_processors(env_cfg, adapter.policy.config)

    # Snapshot the resolved model config + checkpoint into run.json so downstream
    # analysis stages can rebuild the exact policy (even if checkpoint defaulted).
    resolved_ckpt = getattr(adapter, "checkpoint", None) or (str(checkpoint) if checkpoint else None)
    if into_run is None:  # don't let shard workers race on the shared run.json
        run.meta.model["checkpoint"] = resolved_ckpt
        pc = adapter.policy.config
        cfg_snap = {
            k: getattr(pc, k, None) for k in
            ("type", "num_inference_steps", "chunk_size", "n_action_steps", "max_action_dim", "compile_model")
        }
        if cfg_snap.get("num_inference_steps") is None:
            cfg_snap["num_inference_steps"] = getattr(pc, "num_steps", None)  # some configs name it num_steps
        run.meta.model["config"] = cfg_snap
        run.write_meta()

    videos_dir = run.videos_dir if max_episodes_rendered > 0 else None

    recorder = None
    if record_fm:
        recorder = FMRecorder(
            adapter=adapter, output_dir=run.fm_dir, envs=envs,
            run_metadata={
                "source": "libero_eval", "policy_path": resolved_ckpt,
                "task_group": dataset.task_group, "task_ids": dataset.task_ids,
                "seed": seed, "worker_id": worker_id, "run_id": run.run_id,
                "model": run.meta.model, "dataset": run.meta.dataset,
            },
            file_prefix=file_prefix,
            manifest_name=f"manifest_{worker_tag}.json" if worker_tag else "manifest.json",
            capture_context=record_context,
        )
        recorder.attach()
        logger.info("FM recording -> %s (context=%s)", run.fm_dir, record_context)

    try:
        with torch.no_grad():
            info = eval_policy_all(
                envs=envs, policy=adapter.policy,
                env_preprocessor=env_pre, env_postprocessor=env_post,
                preprocessor=adapter.preprocessor, postprocessor=adapter.postprocessor,
                n_episodes=n_episodes, max_episodes_rendered=max_episodes_rendered,
                videos_dir=videos_dir, start_seed=seed, max_parallel_tasks=max_parallel_tasks,
            )
    finally:
        if recorder is not None:
            try:
                recorder.flush()
            except Exception:
                logger.exception("FMRecorder.flush() raised; recording may be partial.")
            recorder.detach()
        dataset.close(envs)

    eval_info_name = f"eval_info_{worker_tag}.json" if worker_tag else "eval_info.json"
    write_json(run.eval_dir / eval_info_name, info)
    if into_run is None:
        run.record_stage("eval", {"overall": info.get("overall"), "n_episodes": n_episodes})
    logger.info("eval done: %s", info.get("overall"))
    return run


def _run_teacher_forced(*, model: str, dataset: Any, checkpoint: str | None, device: str, seed: int,
                        tag: str | None, num_inference_steps: int | None, record_fm: bool,
                        record_context: bool,
                        file_prefix: str, worker_id: str | None, max_rollouts: int | None) -> runs.RunDir:
    """Drive a policy over a dataset's held-out states (no env), recording FM into a
    run-dir. Used for the toy: train → record → analyze, all in the new system."""
    import numpy as np
    import torch

    adapter = get_model(model).build(
        checkpoint, device=device, action_dim=dataset.action_dim,
        num_inference_steps=num_inference_steps,
    )
    resolved_ckpt = getattr(adapter, "checkpoint", None) or (str(checkpoint) if checkpoint else None)
    run = runs.create_run(
        model={"name": model, "checkpoint": resolved_ckpt,
               "config": {"num_inference_steps": adapter.config.num_inference_steps,
                          "chunk_size": adapter.config.chunk_size, "action_dim": adapter.config.action_dim}},
        dataset={"name": dataset.name, "task_group": dataset.task_group,
                 "run_slug": dataset.run_slug, "mode": "teacher_forced"},
        stage="eval", tag=tag,
        args={"seed": seed, "record_fm": record_fm, "record_context": record_context},
    )
    logger.info("teacher-forced run: %s -> %s", run.run_id, run.root)
    torch.manual_seed(seed)
    np.random.seed(seed)

    recorder = FMRecorder(
        adapter=adapter, output_dir=run.fm_dir, envs=None,
        run_metadata={"source": "teacher_forced", "policy_path": resolved_ckpt,
                      "task_group": dataset.task_group, "seed": seed, "worker_id": worker_id,
                      "run_id": run.run_id, "model": run.meta.model, "dataset": run.meta.dataset},
        file_prefix=file_prefix, capture_context=record_context,
    )
    recorder.attach()
    recorder.set_task(0, dataset.task_group)
    n_rollouts = 0
    try:
        for states in dataset.heldout_states(max_rollouts=max_rollouts):
            adapter.policy.reset()
            for t in range(len(states)):
                adapter.policy.select_action({"observation.state": states[t],
                                              "task": [dataset.task_desc(states[t])]})
            n_rollouts += 1
        recorder.flush()
    finally:
        recorder.detach()
    run.record_stage("eval", {"mode": "teacher_forced", "n_rollouts": n_rollouts})
    logger.info("teacher-forced recording done: %d rollouts -> %s", n_rollouts, run.fm_dir)
    return run
