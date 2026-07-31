#!/usr/bin/env python
"""Main-experiment figure: the FM flow field of ONE real π₀.₅ action chunk (PCA view).

The companion figure ``flowfield_toy.py`` draws the geometry of FM uncertainty
on a *2-D toy* net, where the action space is already plottable. This script draws the
**same picture for a real recorded π₀.₅/LIBERO chunk** — so the reader sees that the toy
cartoon is faithful to the production model, not just a didactic sketch.

We take one episode + one action chunk from an existing FM recording (default: the
``2026-07-08_pi05_libero-all`` fdrec run), rebuild that chunk's *exact recorded
conditioning* with :class:`fmaccel.posterior.resample.ChunkResampleSession` (no env, no replay —
the captured-context path the whole failure-detection pipeline relies on), and resample
``N`` independent FM noises through the head, **capturing the full denoise trajectory**
(``x_t``/``v_t``) of every sample.

Because π₀.₅'s action chunk is high-dimensional (``chunk_size × action_dim``), we restrict
to the **execution window** — the first ``n_action_steps`` actions of the chunk, the only
ones the closed-loop actually runs (RTC off), i.e. the exact sub-chunk the
``chunk_geometry``/``chunk_divergence`` stages score — z-score each action dim exactly as
``whole_chunk_curvature`` does, and **PCA-project the flattened execution window to 2-D**.
The recorded denoise trajectories then live on a plane we can draw with the same
streamplot + posterior-cloud + highlighted-path layout as the toy figure. Every sampled
noise→action path is one denoising trajectory; the pink cloud is the resampled action
posterior (the Monte-Carlo resample GT restricted to the execution window); the box
reports its 2-D spread. The chunk is chosen (by default) as the highest-divergence chunk
of the episode, so it reads like the *uncertain* panel of the toy figure.

A **reproduce gate** (feed the recorded noise back through the rebuilt head) runs first;
a small max-abs error proves the conditioning was reconstructed correctly, so the flow
field is the real chunk's, not a drifted one (mirrors ``resample`` / ``chunk_divergence``).

Run (canonical env; a few minutes on one GPU)::

    CUDA_VISIBLE_DEVICES=0 --no-capture-output \\
        python experiments/flowfield_pi05.py

    # pick a specific episode + chunk:
    ... flowfield_pi05.py --rollout-id 61 --chunk-idx 12 --n-traj 2048

Outputs (into ``--out``): ``flowfield_pi05.png/.pdf`` (the single panel,
same layout as the toy figure's right panel) + ``summary.json`` (the chosen
episode/chunk, reproduce error, PCA variance explained, and both the 2-D-view and the
true high-dim standardized posterior spread).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # scripts/ -> for _bootstrap
_sys.path.insert(0, str(_Path(__file__).resolve().parent))          # experiments/ -> for the ref module
import _bootstrap  # noqa: F401,E402  (repo root on sys.path + .env)

# The toy figure's plotting kit — reused verbatim so this figure is byte-for-byte the same
# format: same palette / fonts / streamplot styling / highlighted-path + posterior-cloud
# layout / annotation box / bottom legend. We only relabel the axes (PC 1/2, not raw dims).
import flowfield_toy as ref  # noqa: E402


DEFAULT_RUN = "pi05_libero_all_mainexp"


# ============================================================== chunk selection
def _pick_chunk(rd, rollout_id: int, recording, action_dim: int, n_action_steps: int) -> tuple[int, dict]:
    """Choose the action chunk to feature (highest-divergence by default → an *uncertain*
    field, like the toy figure's OOD panel). Prefer the divergence GT the ``chunk_geometry``
    stage already wrote for this episode; else fall back to the whole-chunk denoise ``accel``
    read straight off the recording (free, no resample). Returns ``(chunk_idx, info)``."""
    import numpy as np
    from fmaccel.geometry.accel import whole_chunk_curvature, _valid_n_chunks

    ro = recording.rollouts[0]
    nv = _valid_n_chunks(ro)
    info: dict = {"selection": None, "n_valid_chunks": int(nv)}

    geo = rd.root / "chunk_geometry" / f"chunk_geometry_ro{rollout_id}.npz"
    if geo.exists():
        with np.load(geo, allow_pickle=True) as g:
            div = np.asarray(g["divergence_max_full"], np.float32)
        if div.size:
            c = int(np.argmax(div))
            info.update(selection="max chunk_divergence (chunk_geometry npz)",
                        divergence_per_chunk=[float(x) for x in div], chosen_divergence=float(div[c]))
            return c, info

    # fallback: highest whole-chunk denoise accel over the executed window (first n_action_steps)
    x_t = np.asarray(ro.x_t[:nv, :, 0, :n_action_steps, :], np.float32)     # (nv, T+1, K, maxA)
    accel = whole_chunk_curvature(x_t, action_dim)["accel"]
    c = int(np.argmax(accel))
    info.update(selection="max whole-chunk accel (fm recording, no divergence npz found)",
                accel_per_chunk=[float(x) for x in accel], chosen_accel=float(accel[c]))
    return c, info


# ============================================================== PCA of the execution window
def _fit_pca_2d(X: "Any") -> tuple["Any", "Any", "Any"]:
    """Deterministic 2-component PCA of ``X`` ``(M, D)`` via SVD.

    Returns ``(mean (D,), W (D, 2), evr (2,))``: the fit mean, the top-2 component matrix
    (columns are unit PCs; ``(X-mean) @ W`` projects to 2-D), and the fraction of total
    variance each of the two PCs explains. Signs are fixed (each PC's largest-magnitude
    entry made positive) so reruns render identically.
    """
    import numpy as np

    mean = X.mean(axis=0)
    Xc = X - mean
    # economy SVD: Vt rows are the principal directions, S**2 ∝ per-component variance.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = np.ascontiguousarray(Vt[:2].T).astype(np.float32)                  # (D, 2)
    for j in range(2):                                                     # deterministic sign
        k = int(np.argmax(np.abs(W[:, j])))
        if W[k, j] < 0:
            W[:, j] = -W[:, j]
    var = (S.astype(np.float64) ** 2)
    evr = (var[:2] / var.sum()).astype(np.float32) if var.sum() > 0 else np.zeros(2, np.float32)
    return mean.astype(np.float32), W, evr


def build_real_record(res: dict, action_dim: int, n_action_steps: int, *, pca_on: str) -> tuple[dict, dict]:
    """Turn a resample result into the same ``rec`` dict the toy figure's ``_panel`` consumes.

    ``res`` is :meth:`ChunkResampleSession.resample` output: ``x_t_in`` ``(N, T+1, K, maxA)``
    and ``times`` ``(T,)``. We slice the **execution window** (first ``n_action_steps`` chunk
    positions, first ``action_dim`` dims), flatten it, z-score each flattened dim over all
    (sample, denoise-step) points — the SAME normalization ``whole_chunk_curvature`` uses —
    and PCA-project to 2-D. The flow-field vectors are the actual per-step Euler displacements
    of the projected paths (so arrows follow the true noise→action denoising motion, sign and
    all), matching what the toy figure's ``vfn``-sampled velocities encode.

    Returns ``(rec, stats)`` where ``stats`` carries the PCA variance explained and both the
    2-D-view and the true high-dim standardized posterior spreads.
    """
    import numpy as np
    from fmaccel.geometry.accel import whole_chunk_curvature

    x = np.asarray(res["x_t_in"], np.float32)                              # (N, T+1, K, maxA)
    N, Tp1 = x.shape[0], x.shape[1]
    exec_win = x[:, :, :n_action_steps, :action_dim]                       # (N, T+1, ns, act)
    flat = exec_win.reshape(N, Tp1, -1)                                    # (N, T+1, ns*act)
    D = flat.shape[-1]

    # per-dim z-score over (sample, step) — identical to whole_chunk_curvature's flat.std((0,1))
    sd = flat.reshape(-1, D).std(axis=0) + ref_EPS
    xz = flat / sd                                                          # (N, T+1, D)

    # PCA basis: on all trajectory points (dominant noise→action contraction flow, like the
    # toy figure's converging streamlines) or on the endpoints only (posterior's own spread).
    fit_pts = xz.reshape(-1, D) if pca_on == "all" else xz[:, -1, :]
    mean, W, evr = _fit_pca_2d(fit_pts)

    traj2d = ((xz - mean) @ W).astype(np.float32)                          # (N, T+1, 2)
    traj = np.transpose(traj2d, (1, 0, 2))                                 # (T+1, N, 2) step-major
    step = traj[1:] - traj[:-1]                                            # (T, N, 2) Euler displacement / step
    endpoints = traj[-1]                                                   # (N, 2) resampled actions (posterior)
    noise_pts = traj[0]                                                    # (N, 2) s=1 noise starts

    # per-trajectory accel in the 2-D view (only used to pick a representative highlighted path)
    accel2d = whole_chunk_curvature(traj2d[:, :, None, :], 2)["accel"]     # (N,)

    # progress colour channel (unused by the solid-navy streamplot, kept for _bin_field's API)
    progress = np.asarray(res.get("times", np.linspace(1.0, 0.0, Tp1 - 1)), np.float32)
    if progress.shape[0] != Tp1 - 1:
        progress = np.linspace(1.0, 0.0, Tp1 - 1, dtype=np.float32)

    spread_2d = float(endpoints.std(axis=0).mean())                        # posterior spread in the 2-D view
    spread_highdim = float(xz[:, -1, :].std(axis=0).mean())               # true mean_std in standardized D-space

    rec = {
        "pos": traj[:-1], "vel": step, "progress": progress,               # -> _bin_field (streamplot field)
        "traj": traj, "noise_pts": noise_pts, "endpoints": endpoints,      # faint cloud + highlighted path
        "means": None,                                                     # no GT modes on a real chunk (OOD-like)
        "accel": accel2d,                                                  # -> highlight_index
        "spread_mean_std": spread_2d,                                      # -> annotation box
    }
    stats = {
        "n_samples": int(N), "flattened_dim": int(D),
        "pca_on": pca_on, "pca_explained_variance_ratio": [float(x) for x in evr],
        "pca_var_explained_2d": float(evr.sum()),
        "posterior_spread_2d_view": spread_2d,
        "posterior_spread_highdim_standardized": spread_highdim,
    }
    return rec, stats


ref_EPS = 1e-12


# ============================================================================ plotting
def make_figure(rec, title, out_dir, *, grid, min_count, n_faint):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xlim, ylim = ref._shared_limits((rec,))                                # one square window (noise + endpoints)
    with plt.rc_context(ref.FONT_RC):
        # 2x2 layout with τ-colorbar on the RIGHT — mirrors the toy individual panels so
        # this figure sits next to the toy figure's panels in the
        # paper without any layout drift.
        fig = plt.figure(figsize=(6.2, 6.2))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.045], height_ratios=[1.0, 0.09],
                              wspace=0.05, hspace=0.24)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        lax = fig.add_subplot(gs[1, :])
        ref._panel(ax, rec, title, None, grid=grid, min_count=min_count, n_faint=n_faint,
                   xlim=xlim, ylim=ylim, show_noise_cloud=False)
        ax.set_xlabel("PC 1", fontsize=9, color=ref.INK_MUTED)
        ax.set_ylabel("PC 2", fontsize=9, color=ref.INK_MUTED)
        ref._add_tau_colorbar_to_ax(fig, cax, orientation="vertical")
        lax.axis("off")
        handles, lbls = ax.get_legend_handles_labels()
        lax.legend(handles, lbls, loc="center", ncol=len(lbls), frameon=False, fontsize=9,
                   borderaxespad=0.0)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"flowfield_pi05.{ext}", dpi=200,
                        bbox_inches="tight", transparent=True)
        plt.close(fig)


# ================================================================================ args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=DEFAULT_RUN, help="run-id (or RunDir) of the producing FM recording")
    p.add_argument("--rollout-id", type=int, default=0, help="which recorded episode (rollout_id)")
    p.add_argument("--chunk-idx", type=int, default=None,
                   help="which action chunk (default: auto = the episode's highest-divergence chunk)")
    p.add_argument("--n-traj", type=int, default=1024, help="resampled-noise trajectories at the chunk")
    p.add_argument("--micro-batch", type=int, default=256, help="FM-head micro-batch (bounds GPU memory)")
    p.add_argument("--seed", type=int, default=0, help="resample noise seed")
    p.add_argument("--num-inference-steps", type=int, default=None, help="override T (default: recording's)")
    p.add_argument("--device", default=None, help="cuda / cpu (default cuda)")
    p.add_argument("--pca-on", choices=("all", "endpoints"), default="all",
                   help="fit PCA on all trajectory points (dominant flow) or the endpoints (posterior spread)")
    p.add_argument("--grid", type=int, default=22, help="streamplot bin resolution")
    p.add_argument("--min-count", type=int, default=4, help="drop flow-field cells with fewer samples")
    p.add_argument("--n-faint", type=int, default=0,
                   help="faint trajectory cloud lines (default 0 = streamlines only; raise to overlay sampled paths)")
    p.add_argument("--reproduce-tol", type=float, default=None,
                   help="reproduce-gate tolerance (default: adapter's, 1e-2 for π₀.₅)")
    p.add_argument("--out", default="outputs/flowfield/pi05", help="output directory")
    return p.parse_args()


def main():
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    import numpy as np

    from fmaccel.core import runs
    from fmaccel.recording.loader import FMRecording
    from fmaccel.registry import get_model
    from fmaccel.posterior.resample import ChunkResampleSession, build_resample_policy

    args = parse_args()
    device = args.device or "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rd = runs.resolve_run(args.run)
    meta = rd.meta
    model_name = str(meta.model.get("name"))
    policy_path = meta.model.get("checkpoint")

    # Load ONLY this episode (memory-bounded: each context sidecar is tens of MB).
    recording = FMRecording.load(rd.fm_dir, rollout_ids=[args.rollout_id])
    if not recording.rollouts:
        raise SystemExit(f"run {rd.run_id!r} has no rollout_id={args.rollout_id} in its recording")
    dims = recording.dims
    action_dim = int(dims["action_dim"])
    n_action_steps = int(dims["n_action_steps"])
    nis = int(args.num_inference_steps) if args.num_inference_steps is not None else int(dims["num_inference_steps"])
    ro = recording.rollouts[0]

    chunk_idx, sel = (_pick_chunk(rd, args.rollout_id, recording, action_dim, n_action_steps)
                      if args.chunk_idx is None else (int(args.chunk_idx), {"selection": "user --chunk-idx"}))
    print(f"run={rd.run_id}  rollout_id={args.rollout_id}  task_id={int(ro.task_id)} "
          f"task_group={ro.task_group}")
    print(f"chunk_idx={chunk_idx}  ({sel['selection']})  "
          f"env_step_at_chunk_start={int(ro.env_step_at_chunk_start[chunk_idx])}")

    reproduce_tol = (args.reproduce_tol if args.reproduce_tol is not None
                     else float(getattr(get_model(model_name), "resample_reproduce_tol", 1e-2)))

    # Rebuild the FM head once, then resample this chunk from its exact recorded context.
    policy = build_resample_policy(str(policy_path), device=device,
                                   n_action_steps=n_action_steps, num_inference_steps=nis)
    with ChunkResampleSession(recording, rollout_idx=0, env_idx=0, chunk_idx=chunk_idx,
                              device=device, num_inference_steps=nis, use_context=True,
                              policy=policy) as sess:
        rep = sess.reproduce()
        repro_ok = bool(rep["max_abs_err"] <= reproduce_tol)
        print(f"reproduce gate: max_abs_err={rep['max_abs_err']:.3e} mean_abs_err={rep['mean_abs_err']:.3e} "
              f"tol={reproduce_tol:.1e} -> {'OK' if repro_ok else 'FAILED'}")
        if not repro_ok:
            print("  WARNING: reproduce gate FAILED — the rebuilt context did not reproduce the recorded "
                  "chunk, so this flow field may be conditioned on a drifted observation.")
        print(f"resampling N={args.n_traj} trajectories (micro_batch={args.micro_batch}, T={nis}) ...")
        res = sess.resample(n_samples=args.n_traj, seed=args.seed, capture_trajectory=True,
                            micro_batch=args.micro_batch)

    rec, stats = build_real_record(res, action_dim, n_action_steps, pca_on=args.pca_on)
    print(f"PCA({args.pca_on}) -> 2D: var explained = {stats['pca_var_explained_2d']:.1%} "
          f"({stats['pca_explained_variance_ratio']})")
    print(f"posterior spread: 2D-view={stats['posterior_spread_2d_view']:.3f}  "
          f"high-dim(standardized)={stats['posterior_spread_highdim_standardized']:.3f}")

    task_desc = str(ro.task_descs[chunk_idx, 0])
    instr = task_desc.split("Task:", 1)[-1].split(", State:", 1)[0].strip() if "Task:" in task_desc else ""
    title = r"$\pi_{0.5}$ / LIBERO — action-chunk flow field (PCA)"
    make_figure(rec, title, out_dir, grid=args.grid, min_count=args.min_count, n_faint=args.n_faint)

    summary = {
        "source_run": rd.run_id, "model": model_name, "checkpoint": str(policy_path),
        "rollout_id": int(args.rollout_id), "task_id": int(ro.task_id), "task_group": ro.task_group,
        "task_instruction": instr, "chunk_idx": int(chunk_idx),
        "env_step_at_chunk_start": int(ro.env_step_at_chunk_start[chunk_idx]),
        "chunk_selection": sel,
        "execution_window": {"n_action_steps": n_action_steps, "action_dim": action_dim,
                             "flattened_dim": stats["flattened_dim"]},
        "num_inference_steps": nis, "n_traj": int(args.n_traj), "seed": int(args.seed),
        "reproduce_max_abs_err": rep["max_abs_err"], "reproduce_mean_abs_err": rep["mean_abs_err"],
        "reproduce_tol": reproduce_tol, "reproduce_ok": repro_ok,
        **stats,
        "note": ("resampled-action posterior (execution window) of one recorded π₀.₅ chunk, "
                 "z-scored per action dim like whole_chunk_curvature, PCA-projected to 2-D; "
                 "flow field = per-step Euler displacements of the projected denoise paths. "
                 "Same layout as the toy figure's right panel."),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote: {out_dir}/flowfield_pi05.png/.pdf, summary.json")


if __name__ == "__main__":
    main()
