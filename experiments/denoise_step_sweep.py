#!/usr/bin/env python
"""main_exp — denoise-step sweep for π₀.₅ × LIBERO-all: how does the accel-prefix
↔ resample-divergence rank correlation depend on how many Euler denoise steps ``T``
the FM head actually runs?

For each ``T ∈ TS`` (default 2, 4, 6, 8, 10, 15, 20) this driver

  1. Runs an 8-GPU eval (``cli/multigpu.py eval``) with
     ``--num-inference-steps T --record-fm --record-context`` on the pi05 LIBERO-all
     fine-tune. Produces one run dir per T.
  2. Shards ``chunk_divergence`` (k=32, first-actions=n_action_steps=10) across the
     8 GPUs on the recorded rollouts, then loads the resulting per-episode npz.
  3. Runs ``chunk_geometry`` single-process on all rollouts, which writes the
     accel-prefix ρ curve into the run's ``chunk_geometry/meta.json``.
  4. Aggregates every T's ρ curve into a T-vs-ρ figure + json + markdown table under
     ``outputs/denoise_step_sweep/``.

By default we run 10 episodes per task × 40 tasks = 400 rollouts per T (bounded so
seven Ts finish on an 8-GPU node in a few hours). Every
per-T run reuses the shared checkpoint / dataset / policy config; only T changes.

    python experiments/denoise_step_sweep.py \\
        --gpus 0,1,2,3,4,5,6,7 \\
        --steps 2 4 6 8 10 15 20 --n-episodes 10 \\
        --checkpoint outputs/finetunes/pi05_libero_all/baseline/checkpoints/010000/pretrained_model_ema

The aggregate step (``--mode aggregate``) skips the runs and just re-reads existing
``chunk_geometry/meta.json`` for each T's run id, so the figure can be rebuilt
without re-running the eval — pass ``--tag`` matching whatever was used at record
time to resolve the run ids automatically, or ``--set-run T=run-id`` per T.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401,E402

from fmaccel.core import runs, sharding  # noqa: E402

DEFAULT_STEPS = [2, 4, 6, 8, 10, 15, 20]
DEFAULT_CHECKPOINT = (
    "outputs/finetunes/pi05_libero_all/baseline/checkpoints/010000/pretrained_model_ema"
)
DEFAULT_OUT = "outputs/denoise_step_sweep"
DEFAULT_TAG_FMT = "denoise-sweep-T{T:02d}"


def _worker_env(gid: int, threads: int) -> dict:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gid),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(REPO),
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }


def _run_workers(cmds: list[tuple[int, list[str], Path]], threads: int) -> None:
    procs: list[tuple[int, subprocess.Popen, Path]] = []
    for gid, cmd, log in cmds:
        log.parent.mkdir(parents=True, exist_ok=True)
        env = _worker_env(gid, threads)
        print(f"  GPU {gid} threads={threads}: {' '.join(shlex.quote(c) for c in cmd)}  -> {log}",
              flush=True)
        procs.append((gid, subprocess.Popen(cmd, env=env,
                                            stdout=open(log, "w"), stderr=subprocess.STDOUT), log))
    fail = 0
    for gid, p, log in procs:
        rc = p.wait()
        if rc != 0:
            print(f"  ✗ GPU {gid} FAILED rc={rc} (see {log})", file=sys.stderr, flush=True)
            fail += 1
        else:
            print(f"  ✓ GPU {gid} done", flush=True)
    if fail:
        raise SystemExit(f"{fail} worker(s) failed.")


def _run_eval_for_T(*, T: int, gpus: list[int], checkpoint: str, n_episodes: int,
                    tag: str, threads: int, seed: int, disable_compile: bool) -> str:
    """Kick off cli/multigpu.py eval for this T and return the created run_id."""
    cmd = [
        sys.executable, str(REPO / "scripts" / "run_multigpu.py"), "eval",
        "--gpus", ",".join(str(g) for g in gpus),
        "--model", "pi05", "--dataset", "libero",
        "--task-group", "libero_all",
        "--task-ids", "0..9",
        "--checkpoint", checkpoint,
        "--tag", tag,
        "--threads-per-proc", str(threads),
        "--",
        "--n-episodes", str(n_episodes),
        "--num-inference-steps", str(T),
        "--seed", str(seed),
        "--record-fm", "--record-context",
    ]
    if disable_compile:
        cmd.append("--disable-compile")
    print(f"[T={T}] eval launcher: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run(cmd, env=env, check=True)
    del proc
    # Discover the run just created by tag (unique by construction: the tag carries T).
    run_dir = None
    for rd in sorted((REPO / "outputs" / "runs").iterdir()):
        if not rd.is_dir():
            continue
        if not (rd / "run.json").exists():
            continue
        if rd.name.endswith(f"_{tag}"):
            run_dir = rd
            break
    if run_dir is None:
        raise RuntimeError(f"[T={T}] eval finished but no run with tag {tag!r} found under outputs/runs/")
    print(f"[T={T}] eval run: {run_dir.name}", flush=True)
    return run_dir.name


def _rollout_ids(run_id: str) -> list[int]:
    from fmaccel.recording import format as fmt
    rd = runs.resolve_run(run_id)
    manifest = fmt.read_manifest(rd.fm_dir)
    return [int(r["rollout_id"]) for r in manifest["rollouts"]]


def _shard_chunk_divergence(*, run_id: str, gpus: list[int], k: int, first_actions: int,
                            micro_batch: int, threads: int) -> None:
    """Run cli/divergence.py in parallel per GPU, each getting a disjoint slice
    of the recording's rollout ids. Every shard writes chunk_divergence_ro<id>.npz into
    the shared <run>/chunk_divergence/ dir (naming is by rollout_id, so no collision)."""
    ids = _rollout_ids(run_id)
    shards = sharding.round_robin(ids, len(gpus))
    log_root = runs.resolve_run(run_id).root
    cmds: list[tuple[int, list[str], Path]] = []
    for gid, shard in zip(gpus, shards):
        if not shard:
            continue
        cmd = [
            sys.executable, str(REPO / "scripts" / "chunk_divergence.py"),
            "--run", run_id,
            "--samples", str(k),
            "--first-actions", str(first_actions),
            "--micro-batch", str(micro_batch),
            "--rollouts", *[str(x) for x in shard],
            "--no-progress",
        ]
        cmds.append((gid, cmd, log_root / f"chunk_divergence_gpu{gid}.log"))
    print(f"[cd] {len(ids)} rollouts across {len(cmds)} GPU shards "
          f"(sizes: {[len(s) for s in shards]}) — k={k}, first_actions={first_actions}",
          flush=True)
    _run_workers(cmds, threads)


def _run_chunk_geometry(run_id: str) -> None:
    log = runs.resolve_run(run_id).root / "chunk_geometry.log"
    cmd = [sys.executable, str(REPO / "scripts" / "chunk_geometry.py"),
           "--run", run_id, "--max-rollouts", "10000"]
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1"}
    print(f"[cg] {' '.join(shlex.quote(c) for c in cmd)}  -> {log}", flush=True)
    with open(log, "w") as f:
        subprocess.run(cmd, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)


def _load_geometry(run_id: str) -> dict:
    rd = runs.resolve_run(run_id)
    meta = json.loads((rd.chunk_geometry_dir / "meta.json").read_text())
    prr = [None if r is None else float(r) for r in (meta.get("run_rho_prefix_accel_vs_div") or [])]
    ns = [int(x) for x in (meta.get("prefix_n_steps") or [])]
    best_k, best_rho = None, None
    for k, r in zip(ns, prr):
        if r is None:
            continue
        if best_rho is None or r > best_rho:
            best_rho, best_k = float(r), int(k)
    return {
        "run_id": run_id,
        "T": int(meta.get("chunk_size", 0)),  # placeholder; caller overrides
        "num_inference_steps": len(ns) + 1 if ns else None,
        "prefix_n_steps": ns,
        "run_rho_prefix_accel_vs_div": prr,
        "run_rho_accel_vs_div": meta.get("run_rho_accel_vs_div"),
        "run_rho_accel_full_vs_div": meta.get("run_rho_accel_full_vs_div"),
        "run_rho_straightness_vs_div": meta.get("run_rho_straightness_vs_div"),
        "n_pooled_chunks": int(meta.get("n_pooled_chunks", 0)),
        "n_rollouts": int(meta.get("n_rollouts", 0)),
        "first_actions": int(meta.get("first_actions", 0)),
        "chunk_size": int(meta.get("chunk_size", 0)),
        "n_action_steps": int(meta.get("n_action_steps", 0)),
        "action_dim": int(meta.get("action_dim", 0)),
        "best_prefix_k": best_k,
        "best_prefix_rho": best_rho,
    }


def _plot_sweep(records: list[dict], out_base: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = sorted(records, key=lambda r: r["T"])
    if not records:
        print("no records to plot")
        return

    cmap = plt.get_cmap("viridis")
    Ts = [r["T"] for r in records]
    tmin, tmax = min(Ts), max(Ts)

    # ---- Fig A: prefix-ρ curve per T (x = k/T, y = ρ) ----
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for r in records:
        ns = np.asarray(r["prefix_n_steps"], int)
        rho = np.asarray([np.nan if x is None else x for x in r["run_rho_prefix_accel_vs_div"]], float)
        T = int(r["T"])
        depth = ns / T
        color = cmap((T - tmin) / max(1, (tmax - tmin)))
        ax.plot(depth, rho, "-o", ms=5, lw=1.9, color=color,
                label=f"T={T:>2}  (n={r['n_pooled_chunks']}, peak ρ={r['best_prefix_rho']:+.3f} @ k≤{r['best_prefix_k']})")
        if r["best_prefix_k"] is not None and r["best_prefix_rho"] is not None:
            ax.plot([r["best_prefix_k"] / T], [r["best_prefix_rho"]], marker="*",
                    ms=13, color=color, mec="k", mew=0.5, zorder=5)
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.set_xlabel("fraction of the noise→clean denoise path folded into the accel prefix  (k / T)")
    ax.set_ylabel("Spearman ρ  (prefix-accel vs resample-GT chunk divergence)")
    ax.set_title("π₀.₅ · LIBERO-all — denoise-step sweep\n"
                 "prefix-accel ρ curve, one line per denoise-step count T   ·   ★ = peak per T")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}_prefix_curves.{ext}", dpi=150)
    plt.close(fig)

    # ---- Fig B: T-vs-peak ρ and T-vs-full-path ρ ----
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    Ts_arr = np.asarray([r["T"] for r in records], float)
    peak = np.asarray([np.nan if r["best_prefix_rho"] is None else r["best_prefix_rho"] for r in records], float)
    full = np.asarray([np.nan if r["run_rho_accel_vs_div"] is None else r["run_rho_accel_vs_div"] for r in records], float)
    ax.plot(Ts_arr, peak, "-o", ms=8, lw=2.0, color="C0", label="peak prefix ρ (headline)")
    ax.plot(Ts_arr, full, "-s", ms=6, lw=1.5, color="C3", alpha=0.75,
            label="full-path accel ρ (last cutoff)")
    for r in records:
        if r["best_prefix_rho"] is not None:
            ax.annotate(f"@k≤{r['best_prefix_k']}", (r["T"], r["best_prefix_rho"]),
                        xytext=(4, 6), textcoords="offset points", fontsize=8, color="C0")
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.set_xlabel("denoise steps  T  (num_inference_steps)")
    ax.set_ylabel("Spearman ρ  (accel vs resample-divergence)")
    ax.set_title("π₀.₅ · LIBERO-all — accel↔divergence ρ vs denoise-step count T")
    ax.set_xticks(Ts_arr.astype(int))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}_T_vs_rho.{ext}", dpi=150)
    plt.close(fig)


def _write_markdown(records: list[dict], out_path: Path, meta: dict) -> None:
    records = sorted(records, key=lambda r: r["T"])
    lines = []
    lines.append("# π₀.₅ × LIBERO-all — denoise-step (T) sweep of accel-prefix ↔ resample-divergence ρ\n")
    lines.append(f"- checkpoint: `{meta['checkpoint']}`")
    lines.append(f"- n_episodes/task: {meta['n_episodes']}  ·  seed: {meta['seed']}")
    lines.append(f"- chunk_divergence: k={meta['k']}, first_actions={meta['first_actions']}, "
                 f"n_action_steps={meta['n_action_steps']}")
    lines.append(f"- GPUs: {meta['gpus']}\n")
    lines.append("| T | run_id | n_pooled_chunks | n_rollouts | ρ full-path | **peak prefix ρ** | @ k steps |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in records:
        fp = "—" if r["run_rho_accel_vs_div"] is None else f"{r['run_rho_accel_vs_div']:+.3f}"
        bp = "—" if r["best_prefix_rho"] is None else f"**{r['best_prefix_rho']:+.3f}**"
        bk = "—" if r["best_prefix_k"] is None else f"≤{r['best_prefix_k']}"
        lines.append(f"| {r['T']} | `{r['run_id']}` | {r['n_pooled_chunks']} | {r['n_rollouts']} | "
                     f"{fp} | {bp} | {bk} |")
    lines.append("\n## Prefix ρ curves (one row per T)\n")
    for r in records:
        pairs = [(k, rho) for k, rho in zip(r["prefix_n_steps"], r["run_rho_prefix_accel_vs_div"])]
        cells = []
        for k, rho in pairs:
            s = "—" if rho is None else (f"**{rho:+.3f}**" if k == r["best_prefix_k"] else f"{rho:+.3f}")
            cells.append(f"k≤{k}: {s}")
        lines.append(f"- **T={r['T']}** (num_inference_steps={r['num_inference_steps']}): "
                     + "  ".join(cells))
    out_path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="GPU ids for eval + chunk_divergence sharding")
    p.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS,
                   help="denoise step counts to sweep (num_inference_steps values)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="pi05 fine-tune checkpoint (relative to repo or absolute)")
    p.add_argument("--n-episodes", type=int, default=10, help="episodes per task (× 40 tasks = rollouts/T)")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--k", type=int, default=32, help="chunk_divergence samples (k)")
    p.add_argument("--first-actions", type=int, default=10,
                   help="chunk_divergence executed-window length (n_action_steps)")
    p.add_argument("--micro-batch", type=int, default=16,
                   help="chunk_divergence per-forward batch cap")
    p.add_argument("--threads-per-proc", type=int, default=8,
                   help="OMP/MKL threads per worker (cap CPU oversubscription)")
    p.add_argument("--tag-fmt", default=DEFAULT_TAG_FMT,
                   help="tag format string with {T}; used to name each per-T run")
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory for the aggregate figure/table")
    p.add_argument("--mode", choices=("full", "eval", "cd", "cg", "aggregate", "resume"),
                   default="resume",
                   help="'full'=re-run every stage. 'resume'=skip stages already done (default). "
                        "'eval' only. 'cd' chunk_divergence only. 'cg' chunk_geometry only. "
                        "'aggregate' only re-reads existing meta.json — pair with --set-run.")
    p.add_argument("--set-run", action="append", default=[],
                   help="pin a T's run id: --set-run 4=2026-07-21_pi05_libero-all_denoise-sweep-T04 (repeatable)")
    p.add_argument("--no-disable-compile", dest="disable_compile", action="store_false", default=True,
                   help="don't pass --disable-compile (default: pass it — FM hooks break under compile)")
    args = p.parse_args()

    ckpt = args.checkpoint
    if not Path(ckpt).is_absolute():
        ckpt = str(REPO / ckpt)
    gpus = sharding.parse_gpus(args.gpus)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)

    # Pre-populate run ids from --set-run overrides (T=run-id form).
    pinned: dict[int, str] = {}
    for kv in args.set_run:
        k, _, v = kv.partition("=")
        if k and v:
            pinned[int(k)] = v.strip()

    records: list[dict] = []
    for T in args.steps:
        tag = args.tag_fmt.format(T=T)
        print(f"\n=== T={T} (tag={tag}) ===", flush=True)

        run_id = pinned.get(T)
        if run_id is None:
            # Try to find an existing run by tag (resume mode)
            for rd in sorted((REPO / "outputs" / "runs").iterdir()):
                if rd.is_dir() and rd.name.endswith(f"_{tag}") and (rd / "run.json").exists():
                    run_id = rd.name
                    break

        need_eval = args.mode in ("full", "eval") or (args.mode == "resume" and run_id is None)
        if need_eval:
            print(f"[T={T}] running eval …", flush=True)
            t0 = time.time()
            run_id = _run_eval_for_T(
                T=T, gpus=gpus, checkpoint=ckpt, n_episodes=args.n_episodes,
                tag=tag, threads=args.threads_per_proc, seed=args.seed,
                disable_compile=args.disable_compile,
            )
            print(f"[T={T}] eval done in {time.time() - t0:.0f}s -> {run_id}", flush=True)
        else:
            if run_id is None:
                raise SystemExit(f"[T={T}] no run id found (mode={args.mode!r}); "
                                 f"re-run with --mode eval or --set-run {T}=<run-id>")
            print(f"[T={T}] eval skipped (using {run_id})", flush=True)

        rd = runs.resolve_run(run_id)
        cd_meta = rd.root / "chunk_divergence" / "meta.json"
        need_cd = args.mode in ("full", "cd") or (args.mode == "resume" and not cd_meta.exists())
        # `resume` also re-runs cd if the recording length changed (rare — but the meta records
        # its num_inference_steps, so a mismatch means a stale cd result vs a fresh eval).
        if need_cd:
            print(f"[T={T}] running chunk_divergence (8-way shard) …", flush=True)
            t0 = time.time()
            _shard_chunk_divergence(
                run_id=run_id, gpus=gpus, k=args.k, first_actions=args.first_actions,
                micro_batch=args.micro_batch, threads=args.threads_per_proc,
            )
            print(f"[T={T}] chunk_divergence done in {time.time() - t0:.0f}s", flush=True)
        else:
            print(f"[T={T}] chunk_divergence skipped (meta exists at {cd_meta})", flush=True)

        cg_meta = rd.chunk_geometry_dir / "meta.json"
        need_cg = args.mode in ("full", "cg") or (args.mode == "resume" and not cg_meta.exists())
        if need_cg:
            print(f"[T={T}] running chunk_geometry …", flush=True)
            t0 = time.time()
            _run_chunk_geometry(run_id)
            print(f"[T={T}] chunk_geometry done in {time.time() - t0:.0f}s", flush=True)
        else:
            print(f"[T={T}] chunk_geometry skipped", flush=True)

        rec = _load_geometry(run_id)
        rec["T"] = int(T)
        records.append(rec)
        print(f"[T={T}] ρ full={rec['run_rho_accel_vs_div']}  "
              f"peak={rec['best_prefix_rho']} @ k≤{rec['best_prefix_k']}   "
              f"(n={rec['n_pooled_chunks']} chunks / {rec['n_rollouts']} rollouts)", flush=True)

    # ---- aggregate ----
    out_base = out / "denoise_step_sweep"
    _plot_sweep(records, out_base)
    aggregate = {
        "checkpoint": ckpt,
        "n_episodes_per_task": args.n_episodes,
        "seed": args.seed,
        "k": args.k,
        "first_actions": args.first_actions,
        "gpus": args.gpus,
        "records": records,
    }
    (out / "denoise_step_sweep.json").write_text(json.dumps(aggregate, indent=2, default=str))

    _write_markdown(records, out / "denoise_step_sweep.md", meta={
        "checkpoint": ckpt, "n_episodes": args.n_episodes, "seed": args.seed,
        "k": args.k, "first_actions": args.first_actions,
        "n_action_steps": records[0]["n_action_steps"] if records else args.first_actions,
        "gpus": args.gpus,
    })

    print(f"\nwrote:")
    print(f"  {out}/denoise_step_sweep_prefix_curves.png (+ .pdf)")
    print(f"  {out}/denoise_step_sweep_T_vs_rho.png (+ .pdf)")
    print(f"  {out}/denoise_step_sweep.json")
    print(f"  {out}/denoise_step_sweep.md")


if __name__ == "__main__":
    main()
