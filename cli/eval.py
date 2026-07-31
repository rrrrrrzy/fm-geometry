#!/usr/bin/env python
"""Eval a model on a dataset, optionally recording FM denoising trajectories.

    python cli/eval.py --model pi05 --dataset libero \
        --task-group libero_spatial --task-ids 0 --n-episodes 1 \
        --record-fm --record-context --disable-compile

Writes one self-describing run dir at outputs/runs/<run-id>/ (run.json + eval/ +
fm/ + videos/). Downstream stages take it via `--run <run-id>`.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  # repo path + .env

from fmaccel.core import args as A
from fmaccel.registry import get_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    A.add_model(p)
    A.add_dataset(p)
    A.add_common(p)
    A.add_render(p)
    A.add_fm_inference(p)
    A.add_recording(p)
    p.add_argument("--max-episodes-rendered", type=int, default=1, help="rollouts to dump as mp4 (0=none)")
    p.add_argument("--max-parallel-tasks", type=int, default=1, help="ThreadPool size across task_ids")
    p.add_argument("--use-async-envs", action=argparse.BooleanOptionalAction, default=None,
                   help="AsyncVectorEnv (default: on iff batch-size>1)")
    p.add_argument("--compile-mode", default="max-autotune",
                   choices=("default", "reduce-overhead", "max-autotune"))
    p.add_argument("--worker-id", default=None, help="stored in run/FM metadata (e.g. gpu0)")
    p.add_argument("--file-prefix", default="", help="FM rollout-file prefix for multi-GPU shards")
    p.add_argument("--dataset-root", default=None, help="dataset source root (a LeRobot dataset), for adapters that need one")
    p.add_argument("--dataset-mode", default=None,
                   help="execution mode, for adapters that support more than one "
                        "(default: the adapter's own; LIBERO is closed-loop vec-env)")
    p.add_argument("--into-run", default=None, help="write into an existing run (multi-GPU shard worker)")
    args = p.parse_args()

    # default seed: eval historically used 1000 (add_common defaults to 0)
    seed = args.seed if args.seed else 1000

    DatasetCls = get_dataset(args.dataset)
    # Pass only the kwargs this dataset's constructor accepts (LIBERO has no
    # dataset_root) — keeps one generic eval script across datasets.
    import inspect
    candidate = {"task_group": args.task_group, "task_ids": args.task_ids,
                 "render_size": args.render_size, "dataset_root": args.dataset_root,
                 "mode": args.dataset_mode}
    accepted = set(inspect.signature(DatasetCls.__init__).parameters)
    dataset = DatasetCls(**{k: v for k, v in candidate.items() if k in accepted and v is not None})

    from fmaccel.eval import run_eval

    run = run_eval(
        model=args.model, dataset=dataset, checkpoint=args.checkpoint,
        device=args.device, seed=seed, tag=args.tag,
        n_episodes=args.n_episodes, batch_size=args.batch_size,
        use_async_envs=args.use_async_envs, max_episodes_rendered=args.max_episodes_rendered,
        max_parallel_tasks=args.max_parallel_tasks,
        n_action_steps=args.n_action_steps if args.n_action_steps is not None else 10,
        num_inference_steps=args.num_inference_steps,
        disable_compile=args.disable_compile, compile_mode=args.compile_mode,
        record_fm=args.record_fm, record_context=args.record_context,
        file_prefix=args.file_prefix, worker_id=args.worker_id, into_run=args.into_run,
    )
    print(f"run_id: {run.run_id}")
    print(f"dir:    {run.root}")


if __name__ == "__main__":
    main()
