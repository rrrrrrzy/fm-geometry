#!/usr/bin/env python
"""Per-step k-candidate chunk divergence along a *recorded* episode.

Reads a producing run's FM recording, replays each recorded episode in the gym
env (deterministic), and at every visited observation draws k action chunks (k FM
noises) and records the MAX pairwise chunk distance (mean per-action L2 over the
chunk). Headline output is ``max_pairwise_dist[_roN].npy`` ``[n_steps, 1]``.

    python cli/divergence.py --run <run-id> \
        --samples 32 --rollouts 0 --seed 0

Writes into the source run's ``<run>/chunk_divergence/`` (npy + npz + png + meta).
A reproduce gate (recorded noise must reproduce the recorded chunk at the
reconstructed observation) flags any episode whose replay drifted.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  # repo path + .env

from fmaccel.core import args as A


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    A.add_run(p)                  # --run (the producing run; reads its run.json + fm/)
    A.add_common(p)               # --device/--seed/--tag (seed = resample-noise RNG)
    A.add_sampling(p, default_samples=32)   # --samples == k candidate chunks per step; --micro-batch
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="override FM denoise steps (default: the recording's)")
    p.add_argument("--first-actions", type=int, default=None,
                   help="restrict the chunk-pair distance to the first N actions of the chunk "
                        "(default: whole chunk; pass n_action_steps to measure the executed plan's spread)")
    p.add_argument("--reproduce-every", type=int, default=0,
                   help="context mode: run the reproduce fidelity gate every N chunks (default 0 = chunk 0 "
                        "per rollout only). Reproduce is exact for captured-context recordings, so skipping "
                        "the redundant rest ~halves per-chunk cost. Pass 1 to gate every chunk.")
    p.add_argument("--rollouts", type=int, nargs="+", default=None,
                   help="recorded rollout indices to replay (default: all)")
    p.add_argument("--max-rollouts", type=int, default=None, help="cap number of rollouts replayed")
    p.add_argument("--dataset-root", default=None,
                   help="dataset source root for the teacher-forced producer "
                        "LeRobot dataset whose held-out demo states conditioned the toy) — reloads the "
                        "exact per-chunk conditioning states, no gym replay (default: adapter's root)")
    p.add_argument("--max-steps", type=int, default=None, help="cap env steps per episode (default: to done)")
    p.add_argument("--no-save-chunks", dest="save_chunks", action="store_false",
                   help="skip storing the per-step k chunks in the npz (smaller output)")
    p.add_argument("--per-action-divergence", dest="per_action_divergence",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="also store the within-chunk per-action divergence profile "
                        "(per_action_{max,mean}_dist [n, window]) in the npz so you can see WHERE in the "
                        "chunk the candidate plans diverge (default: on; --no-per-action-divergence to skip)")
    p.add_argument("--checkpoint", default=None,
                   help="rebuild the FM head from THIS checkpoint instead of the one recorded in the "
                        "manifest — for when the recording's checkpoint was relocated on disk (the "
                        "reproduce gate re-verifies the substituted weights still reproduce the chunk)")
    p.add_argument("--dim-scale-file", default=None,
                   help="JSON file containing one run-pooled per-action-dimension scale. Use the SAME "
                        "file for every --rollouts shard so multi-GPU divergence keeps one common metric.")
    p.add_argument("--no-progress", dest="progress", action="store_false", help="disable tqdm progress bars")
    args = p.parse_args()

    from fmaccel.posterior.divergence import run_chunk_divergence

    dim_scale = None
    if args.dim_scale_file:
        import json
        from pathlib import Path
        dim_scale = json.loads(Path(args.dim_scale_file).read_text())

    rd, summary = run_chunk_divergence(
        args.run, k=args.samples, rollouts=args.rollouts, max_rollouts=args.max_rollouts,
        max_steps=args.max_steps, device=args.device, seed=args.seed,
        save_chunks=args.save_chunks, num_inference_steps=args.num_inference_steps,
        micro_batch=args.micro_batch, first_actions=args.first_actions,
        reproduce_every=args.reproduce_every, dataset_root=args.dataset_root,
        per_action_divergence=args.per_action_divergence,
        checkpoint_override=args.checkpoint, progress=args.progress,
        dim_scale_override=dim_scale,
    )
    print(f"run:  {rd.run_id}")
    print(f"dir:  {rd.chunk_divergence_dir}")
    print(f"mode: {summary.get('mode', 'gym')}  first_actions={summary.get('first_actions')}")
    print(f"reproduce: worst_err={summary['worst_reproduce_err']:.2e} ok={summary['reproduce_ok']}")
    for ep in summary["episodes"]:
        units = ep.get("n_steps", ep.get("n_chunks"))   # gym = env steps, context = chunks
        succ = f" success={ep['success']}" if "success" in ep else ""
        print(f"  ro{ep['rollout_id']}: units={units}{succ} "
              f"peak_div={ep['peak_max_dist']:.2f}@{ep['peak_step']} mean_div={ep['mean_max_dist']:.2f} "
              f"reproduce_err={ep['max_reproduce_err']:.2e}")


if __name__ == "__main__":
    main()
