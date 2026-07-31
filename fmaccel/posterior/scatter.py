"""Posterior stage: resample every chunk K times + compute scatter metrics.

Thin wrapper over :func:`fmaccel.posterior.sweep_to_files` pointed at
``<run>/fm`` and writing ``<run>/posterior/`` (or a ``shard_<i>/`` subdir for
multi-GPU sharding).
"""

from __future__ import annotations

from typing import Any, Sequence

from fmaccel.core import runs


def run_posterior(run: Any, *, samples: int = 2048, micro_batch: int | None = None,
                  rollouts: Sequence[int] | None = None, envs: Sequence[int] | None = None,
                  chunks: Sequence[int] | None = None, max_chunks: int | None = None,
                  num_shards: int = 1, shard_index: int = 0, seed: int = 0,
                  device: str = "cuda", stats_file: str | None = None) -> tuple[Any, dict]:
    from fmaccel.posterior import sweep_to_files

    rd = run if isinstance(run, runs.RunDir) else runs.resolve_run(run)
    out_dir = rd.posterior_dir if num_shards == 1 else (rd.posterior_dir / f"shard_{shard_index}")
    summary = sweep_to_files(
        str(rd.fm_dir), out_dir, k=samples, micro_batch=micro_batch,
        rollouts=rollouts, envs=envs, chunks=chunks, max_chunks=max_chunks,
        num_shards=num_shards, shard_index=shard_index, seed=seed, device=device,
        stats_file=stats_file,
    )
    rd.record_stage("posterior", {"samples": samples, "shard": f"{shard_index}/{num_shards}"})
    return out_dir, summary
