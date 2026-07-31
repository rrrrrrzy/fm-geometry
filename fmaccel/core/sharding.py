"""Multi-GPU shard math (the subprocess launcher lives in P5 / run_multigpu.py).

Two splitting strategies, matching the old bash wrappers:

- :func:`round_robin` — task-level sharding for eval (task ``i`` -> gpu ``i % n``),
  so a partially-failed GPU loses whole tasks, not interleaved episodes.
- :func:`balanced` — contiguous balanced chunks for the posterior sweep, where
  per-item cost is roughly uniform and even load matters more than locality.

Both return a list of ``n`` lists (some possibly empty if ``len(items) < n``).
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def round_robin(items: Sequence[T], n: int) -> list[list[T]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    shards: list[list[T]] = [[] for _ in range(n)]
    for i, it in enumerate(items):
        shards[i % n].append(it)
    return shards


def balanced(items: Sequence[T], n: int) -> list[list[T]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    items = list(items)
    total = len(items)
    shards: list[list[T]] = []
    start = 0
    for s in range(n):
        size = total // n + (1 if s < total % n else 0)
        shards.append(items[start:start + size])
        start += size
    return shards


def parse_gpus(spec: str | None) -> list[int]:
    """Parse ``"0,1,2,3"`` (or ``"0 1 2 3"``) into ``[0,1,2,3]``."""
    if not spec:
        return []
    return [int(x) for x in spec.replace(",", " ").split()]


# --------------------------------------------------------------------- merges
def merge_eval_info(paths: Sequence) -> dict:
    """Concatenate per-worker eval_info.json (disjoint task_ids) + recompute
    per_group / overall aggregates."""
    import json
    import statistics
    from collections import defaultdict

    def _mean(xs):
        return float(statistics.fmean(xs)) if xs else float("nan")

    per_task = []
    for p in paths:
        per_task.extend(json.loads(open(p).read()).get("per_task", []))
    acc = defaultdict(lambda: {"successes": [], "sum_rewards": [], "max_rewards": []})
    for t in per_task:
        m = t.get("metrics", {})
        for key in ("successes", "sum_rewards", "max_rewards"):
            acc[t["task_group"]][key].extend(m.get(key, []))
    per_group, os_, osum, omax = {}, [], [], []
    for g, a in acc.items():
        s = a["successes"]
        per_group[g] = {"n_episodes": len(s), "pc_success": _mean([1.0 if x else 0.0 for x in s]) * 100,
                        "avg_sum_reward": _mean(a["sum_rewards"]), "avg_max_reward": _mean(a["max_rewards"])}
        os_ += s; osum += a["sum_rewards"]; omax += a["max_rewards"]
    overall = {"n_episodes": len(os_), "pc_success": _mean([1.0 if x else 0.0 for x in os_]) * 100,
               "avg_sum_reward": _mean(osum), "avg_max_reward": _mean(omax)}
    return {"per_task": per_task, "per_group": per_group, "overall": overall}


def merge_fm_manifests(paths: Sequence) -> dict:
    """Merge per-worker v3 FM manifests: concat rollouts with a fresh global
    rollout_id, keep dims from the first. (Rollout npz files don't collide —
    each worker used a distinct file_prefix.)"""
    import json

    manifests = [json.loads(open(p).read()) for p in paths]
    base = manifests[0]
    rollouts, rid = [], 0
    has_ctx = False
    for m in manifests:
        has_ctx = has_ctx or m.get("has_context", False)
        for ro in m["rollouts"]:
            ro = dict(ro)
            ro["worker_rollout_id"] = ro.get("rollout_id")
            ro["rollout_id"] = rid
            rid += 1
            rollouts.append(ro)
    out = dict(base)
    out.update({"rollouts": rollouts, "has_context": has_ctx,
                "source_manifests": [str(p) for p in paths]})
    return out
