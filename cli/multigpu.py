#!/usr/bin/env python
"""One multi-GPU launcher for sharded stages (replaces the 3 bash wrappers).

Spawns one single-GPU worker per GPU, then merges. Two stages:

  eval      — round-robin task_ids across GPUs into ONE shared run dir, then merge
              per-worker eval_info + FM manifests.
      python cli/multigpu.py eval --gpus 0,1,2,3 \
          --model pi05 --dataset libero --task-group libero_object --task-ids 0..9 \
          --record-fm --record-context

  posterior — balanced chunk shards over an existing run, then merge shards.
      python cli/multigpu.py posterior --gpus 0,1,2,3 --run <run-id> --samples 512

Worker stdout/stderr go to <run>/gpu<i>.log.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from fmaccel.core import runs, sharding

REPO = Path(__file__).resolve().parent.parent


def _expand_ids(spec: list[str]) -> list[int]:
    """Accept "0 1 2" or "0..9" forms."""
    out: list[int] = []
    for tok in spec:
        if ".." in tok:
            a, b = tok.split("..")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def _worker_env(gid: int, threads: int, extra: dict | None = None) -> dict:
    """Worker process env: pin the GPU and CAP the CPU thread pools. Without the caps
    each torch process defaults OMP/MKL/BLAS to the full core count, so N concurrent
    workers oversubscribe the CPU N-fold (loadavg ≫ cores, ~45k ctxsw/s) and STARVE
    the GPUs — measured as the dominant sharded-eval bottleneck (GPU sm duty ~50%, half the
    per-frame wall-time stuck in trivial Python/CPU prep). ``threads`` ≈ cores /
    concurrent-procs restores a ~1:1 core mapping."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gid),
           "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO),
           "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "OPENBLAS_NUM_THREADS": str(threads), "NUMEXPR_NUM_THREADS": str(threads)}
    if extra:
        env.update(extra)
    return env


def _auto_threads(nslots: int, override: int | None) -> int:
    """CPU threads per worker: explicit ``override`` or cores // concurrent-procs."""
    if override is not None:
        return max(1, override)
    return max(1, (os.cpu_count() or 16) // max(1, nslots))


def _run_workers(cmds: list[tuple[int, list[str], Path]], threads: int = 1) -> None:
    procs = []
    for gid, cmd, log in cmds:
        env = _worker_env(gid, threads)
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"  GPU {gid} (threads={threads}): {' '.join(cmd)}  -> {log}")
        procs.append((gid, subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)))
    fail = 0
    for gid, p in procs:
        if p.wait() != 0:
            print(f"  ✗ GPU {gid} FAILED (see log)", file=sys.stderr)
            fail += 1
        else:
            print(f"  ✓ GPU {gid} done")
    if fail:
        raise SystemExit(f"{fail} worker(s) failed.")


def stage_eval(args, passthrough: list[str]) -> None:
    task_ids = _expand_ids(args.task_ids)
    gpus = sharding.parse_gpus(args.gpus)
    shards = sharding.round_robin(task_ids, len(gpus))
    # Create the shared run up front.
    from fmaccel.registry import get_dataset
    ds = get_dataset(args.dataset)(task_group=args.task_group, task_ids=task_ids)
    run = runs.create_run(model={"name": args.model, "checkpoint": args.checkpoint},
                          dataset={"name": args.dataset, "task_group": args.task_group,
                                   "task_ids": task_ids, "run_slug": ds.run_slug},
                          stage="eval", tag=args.tag,
                          args={"gpus": args.gpus, "passthrough": passthrough})
    print(f"shared run: {run.run_id} -> {run.root}")
    # Support duplicate gpu ids (multi-process-per-GPU for CPU-bound envs): each worker
    # gets a unique rank-based file_prefix so per-worker eval_info + FM manifests don't
    # collide. worker_id = "gpu{gid}r{rank}" when a GPU appears more than once, else "gpu{gid}".
    from collections import Counter as _Counter
    gpu_counts = _Counter(gpus)
    worker_seen: dict[int, int] = {}
    cmds = []
    for gid, shard in zip(gpus, shards):
        if not shard:
            continue
        r = worker_seen.get(gid, 0)
        worker_seen[gid] = r + 1
        wid = f"gpu{gid}r{r}" if gpu_counts[gid] > 1 else f"gpu{gid}"
        cmd = [sys.executable, "cli/eval.py", "--model", args.model, "--dataset", args.dataset,
               "--task-group", args.task_group, "--task-ids", *map(str, shard),
               "--into-run", run.run_id, "--file-prefix", f"{wid}_", "--worker-id", wid,
               *passthrough]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        cmds.append((gid, cmd, run.root / f"{wid}.log"))
    _run_workers(cmds, _auto_threads(len(gpus), args.threads_per_proc))

    # Merge eval_info + FM manifests.
    ei = sorted(run.eval_dir.glob("eval_info_gpu*.json"))
    if ei:
        from fmaccel.core.io import write_json
        merged = sharding.merge_eval_info(ei)
        write_json(run.eval_dir / "eval_info.json", merged)
        run.record_stage("eval", {"overall": merged["overall"], "gpus": args.gpus})
        print("merged overall:", merged["overall"])
    fm = sorted(run.fm_dir.glob("manifest_gpu*.json"))
    if fm:
        merged_m = sharding.merge_fm_manifests(fm)
        (run.fm_dir / "manifest.json").write_text(json.dumps(merged_m, indent=2, default=str))
        print(f"merged {len(fm)} FM manifests -> {run.fm_dir / 'manifest.json'} ({len(merged_m['rollouts'])} rollouts)")


def stage_posterior(args, passthrough: list[str]) -> None:
    gpus = sharding.parse_gpus(args.gpus)
    run = runs.resolve_run(args.run)
    n = len(gpus)
    cmds = []
    for i, gid in enumerate(gpus):
        cmd = [sys.executable, "cli/posterior.py", "--run", run.run_id, "--samples", str(args.samples),
               "--num-shards", str(n), "--shard-index", str(i), *passthrough]
        cmds.append((gid, cmd, run.root / f"posterior_gpu{gid}.log"))
    _run_workers(cmds, _auto_threads(len(gpus), args.threads_per_proc))

    # Merge shard npzs.
    import numpy as np
    import pandas as pd
    shard_npz = sorted(run.posterior_dir.glob("shard_*/posterior_metrics.npz"))
    if not shard_npz:
        raise SystemExit("no shard outputs found")
    metas, payloads = [], []
    for p in shard_npz:
        with np.load(p, allow_pickle=True) as d:
            metas.append(json.loads(str(d["meta_json"])))
            payloads.append({k: d[k] for k in d.files if k != "meta_json"})
    merged = {k: np.concatenate([pl[k] for pl in payloads], axis=0) for k in payloads[0]}
    idx = np.stack([merged["rollout_idx"], merged["env_idx"], merged["chunk_idx"]], axis=1)
    _, uniq = np.unique(idx, axis=0, return_index=True)
    uniq.sort()
    merged = {k: v[uniq] for k, v in merged.items()}
    out_meta = dict(metas[0]); out_meta["merged_from"] = [str(p) for p in shard_npz]
    np.savez(run.posterior_dir / "posterior_metrics.npz",
             meta_json=np.asarray(json.dumps(out_meta)), **merged)
    scalar = [k for k in merged if merged[k].ndim == 1]
    pd.DataFrame({k: merged[k] for k in scalar}).to_csv(run.posterior_dir / "posterior_metrics.csv", index=False)
    run.record_stage("posterior", {"n_rows": int(len(merged["chunk_idx"])), "gpus": args.gpus})
    print(f"merged {len(shard_npz)} shards -> {run.posterior_dir} ({len(merged['chunk_idx'])} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    pe = sub.add_parser("eval", help="round-robin task sharding")
    pe.add_argument("--gpus", required=True)
    pe.add_argument("--model", required=True)
    pe.add_argument("--dataset", required=True)
    pe.add_argument("--task-group", required=True)
    pe.add_argument("--task-ids", nargs="+", required=True, help='e.g. "0 1 2" or "0..9"')
    pe.add_argument("--checkpoint", default=None)
    pe.add_argument("--tag", default=None)
    pe.add_argument("--threads-per-proc", type=int, default=None,
                    help="cap each worker's OMP/MKL/BLAS threads (default: cores // concurrent-procs)")

    pp = sub.add_parser("posterior", help="balanced chunk sharding")
    pp.add_argument("--gpus", required=True)
    pp.add_argument("--run", required=True)
    pp.add_argument("--samples", type=int, default=512)
    pp.add_argument("--threads-per-proc", type=int, default=None,
                    help="cap each worker's OMP/MKL/BLAS threads (default: cores // concurrent-procs)")

    args, passthrough = p.parse_known_args()
    passthrough = [a for a in passthrough if a != "--"]   # drop the optional "--" separator
    if args.stage == "eval":
        stage_eval(args, passthrough)
    else:
        stage_posterior(args, passthrough)


if __name__ == "__main__":
    main()
