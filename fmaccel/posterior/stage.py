"""Resample stage: resample one recorded chunk N times, write into the run-dir.

Thin wrapper over :func:`fmaccel.posterior.resample.resample_to_npz` that points it at
``<run>/fm`` and writes to ``<run>/resample/``. The policy is rebuilt from the
recording's (now resolved) ``policy_path`` in context mode — no env, no replay.
"""

from __future__ import annotations

from typing import Any

from fmaccel.core import runs


def parse_chunk_spec(spec: str) -> tuple[int, int, int]:
    """Parse ``"r0/e0/c3"`` (or ``"0/0/3"``) into ``(rollout, env, chunk)``."""
    parts = spec.replace("r", "").replace("e", "").replace("c", "").split("/")
    if len(parts) != 3:
        raise ValueError(f"--chunk must be r<R>/e<E>/c<C>, got {spec!r}")
    return tuple(int(x) for x in parts)  # type: ignore[return-value]


def run_resample(run: Any, *, rollout_idx: int, env_idx: int, chunk_idx: int,
                 samples: int, seed: int | None = None, micro_batch: int | None = None,
                 capture_trajectory: bool = True, num_inference_steps: int | None = None,
                 device: str = "cuda") -> tuple[Any, dict]:
    from fmaccel.posterior.resample import resample_to_npz

    rd = run if isinstance(run, runs.RunDir) else runs.resolve_run(run)
    seed_tag = "rand" if seed is None else str(seed)
    suffix = "" if capture_trajectory else "_chunks"
    out_path = rd.resample_dir / f"r{rollout_idx}_e{env_idx}_c{chunk_idx}_n{samples}_s{seed_tag}{suffix}.npz"
    res = resample_to_npz(
        recording_path=str(rd.fm_dir), rollout_idx=rollout_idx, env_idx=env_idx, chunk_idx=chunk_idx,
        output_path=str(out_path), n_samples=samples, seed=seed,
        capture_trajectory=capture_trajectory, micro_batch=micro_batch, device=device,
        num_inference_steps=num_inference_steps, use_context=True,
    )
    rd.record_stage("resample", {"chunk": f"r{rollout_idx}/e{env_idx}/c{chunk_idx}", "samples": samples})
    return out_path, res
