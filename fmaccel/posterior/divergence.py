"""Chunk-divergence stage: the resample ground truth along a *recorded* episode.

This is an **analysis stage on an existing run**, not a fresh rollout. At every decision the
policy actually made, we rebuild that decision's exact conditioning from the ``--record-context``
sidecar and draw ``k`` action chunks with ``k`` independent FM noises (one batched forward). How
far apart those candidate plans land is ``D_resample`` — the paper's uncertainty ground truth,
and what the free ``accel`` proxy is validated against. The trace therefore says *where along the
policy's own trajectory its action posterior is spread*.

**Fidelity gate.** Reconstructing the conditioning is the whole ballgame: resampling at a
*drifted* observation measures a posterior the policy never faced. So at each chunk start we feed
the *recorded* noise back through the FM head at the reconstructed conditioning and compare
against the recorded chunk. A small reproduce error proves the reconstruction is right; a large
one means the numbers for that episode are untrustworthy, and the stage says so (mirroring
``resample``'s gate). Recordings without captured context are refused outright rather than
approximated.

Distance contract (per-dim-standardized action units):
  * each action dim is first divided by its **run-pooled winsorized std** (std after clipping the
    dim's extreme 1% tails, over every selected rollout's recorded actions, in the candidates' own
    space) so heterogeneous dims (translation vs gripper) weigh comparably instead of the
    largest-scale dim dominating — winsorized so one diverged rollout can't hijack the scale; see
    ``_pooled_dim_scale``
  * action distance      = Euclidean ``L2`` between two corresponding standardized actions
  * chunk-pair distance  = mean of that over the aligned actions
  * recorded per step    = **max** over all ``k*(k-1)/2`` candidate pairs

Output (into the source run's ``<run>/chunk_divergence/``), one set per episode:
``max_pairwise_dist[_roN].npy`` ``[n_steps, 1]`` (the headline) + a richer ``.npz`` (max/mean
spread, the k chunks, states, executed actions, coverage, chunk-start mask, per-step reproduce
error) + a ``divergence[_roN].png`` + a top-level ``meta.json``. With ``per_action_divergence``
on (default) the ``.npz`` also carries ``per_action_max_dist`` / ``per_action_mean_dist``
``[n, window]`` — the within-chunk divergence profile the scalars collapse with a chunk-mean — so
the trace shows *where in the chunk* the candidate plans diverge.

``first_actions`` restricts the chunk-pair distance to the first N actions of the chunk. With
``n_action_steps < chunk_size`` those are the only actions ever executed, so
``first_actions=n_action_steps`` measures the *executed* plan's spread — which is what the paper
reports (both geometric scores and the GT are computed over the executed window only).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from fmaccel.core import runs
from fmaccel.core.io import write_json
from fmaccel.recording.loader import FMRecording
from fmaccel.registry import get_model

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- math
def sample_candidates(policy: Any, state: Any, k: int, *, micro_batch: int | None = None) -> "Any":
    """Draw ``k`` raw (un-normalized) action chunks ``(k, chunk, act)`` from one state.

    The ``k`` rows share the conditioning (the single normalized state) but get
    ``k`` *independent* FM noises (``sample_actions`` draws ``randn`` per row), so
    this is the "k different noises in parallel" step that probes the action
    posterior at ``state``. ``micro_batch`` bounds how many rows go through the FM
    head at once (the toy is tiny, so the default single batch is usually fine).
    """
    import numpy as np
    import torch

    with torch.no_grad():
        st = torch.as_tensor(np.asarray(state, dtype=np.float32), device=policy.device)
        cond = policy.norm_state.normalize(st).reshape(1, -1)         # (1, obs_dim)
        mb = int(micro_batch) if micro_batch else int(k)
        rows: list[torch.Tensor] = []
        for start in range(0, int(k), mb):
            b = min(start + mb, int(k)) - start
            cond_b = cond.expand(b, -1).contiguous()                  # (b, obs_dim)
            chunk_norm = policy.model.sample_actions(cond_b)          # (b, chunk, act) normalized
            rows.append(policy.norm_action.unnormalize(chunk_norm).detach().cpu())
        all_raw = torch.cat(rows, dim=0)                              # (k, chunk, act) raw
    return all_raw.numpy().astype(np.float32)


def pairwise_chunk_spread(
    chunks: Any, scale: Any | None = None, *, return_per_action: bool = False
) -> tuple:
    """``(max, mean)`` over candidate pairs of the mean per-action L2 distance.

    ``chunks`` is ``(k, chunk, act)``. Returns the **max** (the headline metric)
    and the **mean** (cheap, kept for context) of the ``k*(k-1)/2`` chunk-pair
    distances, where a chunk-pair distance is the mean over the ``chunk`` aligned
    actions of the Euclidean distance between the two actions.

    ``scale`` ``(act,)`` **per-dim-standardizes** the distance: each action dim is
    divided by ``scale[dim]`` before the L2, so dims with different natural scales
    weigh comparably instead of the largest-scale dim dominating. Pass the
    run-pooled per-dim std from ``_pooled_dim_scale``; ``None`` keeps the raw
    (un-standardized) distance.

    With ``return_per_action=True`` also returns the **per-action** (within-chunk)
    breakdown ``(max, mean, pa_max, pa_mean)``: ``pa_max``/``pa_mean`` are ``(chunk,)``
    float32, the max/mean over the same ``k*(k-1)/2`` candidate pairs of the L2
    distance *at each action position* — i.e. the per-action distances that the
    headline scalars collapse with a chunk-mean, kept so the trace shows *where in
    the chunk* the candidate plans diverge. Note the aggregation order differs from
    the scalars (scalar = max-over-pairs of mean-over-chunk; ``pa_max`` = max-over-pairs
    *per* action), so ``pa_max.mean()`` does not equal the scalar ``max``.
    """
    import numpy as np

    k = chunks.shape[0]
    chunk_len = int(chunks.shape[1])
    if k < 2:
        if return_per_action:
            z = np.zeros(chunk_len, np.float32)
            return 0.0, 0.0, z, z.copy()
        return 0.0, 0.0
    diff = chunks[:, None] - chunks[None, :]               # (k, k, chunk, act)
    if scale is not None:
        diff = diff / np.asarray(scale, dtype=np.float32)  # (act,) broadcast — per-dim standardize
    per_action = np.linalg.norm(diff, axis=-1)            # (k, k, chunk)  L2 per standardized action
    iu = np.triu_indices(k, 1)                            # unique unordered pairs
    pair_per_action = per_action[iu[0], iu[1], :]        # (n_pairs, chunk)  per-pair, per-action L2
    vals = pair_per_action.mean(axis=-1)                 # (n_pairs,)  mean over the chunk
    mx, mn = float(vals.max()), float(vals.mean())
    if return_per_action:
        pa_max = pair_per_action.max(axis=0).astype(np.float32)    # (chunk,) max over pairs per action
        pa_mean = pair_per_action.mean(axis=0).astype(np.float32)  # (chunk,) mean over pairs per action
        return mx, mn, pa_max, pa_mean
    return mx, mn


def _pooled_dim_scale(samples: Any, eps: float = 1e-6, clip_pct: float = 1.0) -> Any:
    """Run-pooled per-action-dim **robust** scale to standardize the chunk/action distance.

    ``samples`` is ``(..., act)`` recorded actions pooled over **every** selected
    rollout, so the scale is one fixed vector for the whole run — divergence is then
    comparable across episodes (which the pooled run-level correlations rely on), not
    re-scaled per episode. Per dim the unit is a **winsorized std**: each dim is clipped
    to its ``[clip_pct, 100-clip_pct]`` percentiles before ``std``, so a single
    catastrophically-diverged rollout — e.g. a toy episode where the policy blows up to
    ``|action|~1e8`` — is clipped away and can't hijack the scale (plain std over the run
    is otherwise dominated by <1% of pathological steps). Winsorizing (not IQR) is the
    right robust estimator here: clipping only the extreme ``clip_pct`` keeps clean data
    at ~its plain std, whereas ``IQR/1.349`` also reweights heavy-tailed / bimodal clean
    dims (π₀.₅'s actions are non-Gaussian, so IQR there shifts the per-dim weighting and
    the headline ρ; winsorized std does not). Near-constant dims (scale < ``eps``) are
    left un-rescaled (1.0). Returns ``(act,)`` float32 — divide ``diff[..., dim]`` by it.
    """
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-1])
    lo, hi = np.percentile(flat, [clip_pct, 100.0 - clip_pct], axis=0)
    sd = np.clip(flat, lo, hi).std(axis=0)                  # winsorized (clipped-tail) std
    return np.where(sd < eps, 1.0, sd).astype(np.float32)


# ------------------------------------------------------------------------ plotting
def _plot_divergence(max_div: Any, mean_div: Any, chunk_start: Any, out_path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.arange(len(max_div))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(steps, max_div, lw=1.8, color="C3", label="max pairwise chunk distance")
    ax.plot(steps, mean_div, lw=1.1, color="C0", alpha=0.7, label="mean pairwise chunk distance")
    starts = steps[np.asarray(chunk_start, dtype=bool)]
    if len(starts):
        ax.scatter(starts, np.asarray(max_div)[starts], s=14, color="0.35", zorder=3,
                   label="chunk start (re-plan)")
    if len(max_div):
        peak = int(np.argmax(max_div))
        ax.axvline(peak, color="0.6", ls="--", lw=1.0)
        ax.annotate(f"peak @ step {peak}\n{float(max_div[peak]):.1f}",
                    xy=(peak, float(max_div[peak])), xytext=(6, -2),
                    textcoords="offset points", fontsize=9, color="C3")
    ax.set_xlabel("env step along the recorded episode")
    ax.set_ylabel("chunk distance (raw action units)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_context_divergence(env_step: Any, max_div: Any, mean_div: Any, out_path, title: str,
                             ylabel: str = "chunk distance (normalized action units)") -> None:
    """Per-chunk divergence vs env step (one resample per re-plan; no dense stride)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.asarray(env_step)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, max_div, "-o", ms=3, lw=1.8, color="C3", label="max pairwise chunk distance")
    ax.plot(x, mean_div, "-o", ms=2, lw=1.1, color="C0", alpha=0.7, label="mean pairwise chunk distance")
    if len(max_div):
        peak = int(np.argmax(max_div))
        ax.axvline(x[peak], color="0.6", ls="--", lw=1.0)
        ax.annotate(f"peak @ env {int(x[peak])}\n{float(max_div[peak]):.2f}",
                    xy=(x[peak], float(max_div[peak])), xytext=(6, -2),
                    textcoords="offset points", fontsize=9, color="C3")
    ax.set_xlabel("env step at chunk re-plan")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------- context driver
def _valid_n_chunks_ctx(ro: Any, env_idx: int = 0) -> int:
    """Chunks up to (and including) the one holding the first env-step ``done``.

    Mirrors ``chunk_geometry._valid_n_chunks`` so the two stages resample / read the
    exact same set of chunks per episode.
    """
    import numpy as np

    done = np.asarray(ro.terminated[:, env_idx] | ro.truncated[:, env_idx], dtype=bool)
    hits = np.flatnonzero(done)
    if not len(hits):
        return int(ro.n_chunks)
    d_step = int(hits[0])
    starts = np.asarray(ro.env_step_at_chunk_start)
    return int(max(1, min(int(np.searchsorted(starts, d_step, side="right")), ro.n_chunks)))


def _context_units(recording: FMRecording, idxs: Sequence[int]) -> list[tuple[int, int]]:
    """Return every physical rollout/batch-slot pair as one eval episode.

    With vectorized eval, one FM rollout file stores ``B`` independent episodes. Treating only
    slot zero silently drops ``B-1`` episodes and custom-context resampling also cannot expand a
    recorded ``B>1`` conditioning tensor directly to ``k`` candidates.
    """
    return [(ri, env_idx) for ri in idxs for env_idx in range(recording.rollouts[ri].batch_size)]


def _context_unit_suffix(ro: Any, env_idx: int, *, multiple: bool) -> str:
    if not multiple:
        return ""
    base = f"_ro{int(ro.rollout_id)}"
    return f"{base}_e{env_idx}" if ro.batch_size > 1 else base


def _slice_custom_context(ctx_arrays: dict[str, Any], env_idx: int) -> dict[str, Any]:
    """Select one env while preserving a singleton batch axis for custom adapters.

    Most custom schemas are ``(chunk, B, ...)``; image lists use
    ``(chunk, ncam, B, ...)``. The adapter can then faithfully expand B=1 to k.
    """
    import numpy as np

    out: dict[str, Any] = {}
    for key, value in ctx_arrays.items():
        arr = np.asarray(value)
        batch_axis = 2 if key in {"ctx_images", "ctx_img_masks"} else 1
        sl = [slice(None)] * arr.ndim
        sl[batch_axis] = slice(env_idx, env_idx + 1)
        out[key] = arr[tuple(sl)]
    return out


def _run_context_divergence(
    rd: runs.RunDir,
    recording: FMRecording,
    *,
    model_name: str,
    checkpoint: Any,
    dataset_name: str,
    k: int,
    rollouts: Sequence[int] | None,
    max_rollouts: int | None,
    device: str,
    seed: int,
    save_chunks: bool,
    num_inference_steps: int,
    micro_batch: int | None,
    reproduce_tol: float,
    first_actions: int | None,
    reproduce_every: int,
    per_action_divergence: bool,
    checkpoint_override: str | None = None,
    dim_scale_override: Sequence[float] | None = None,
    progress: bool = True,
) -> tuple[runs.RunDir, dict]:
    """Context-resample chunk divergence for ``has_context`` recordings (π₀.₅ on
    LIBERO): for each selected rollout, rebuild every chunk's exact recorded
    conditioning and draw ``k`` candidate chunks at that re-plan observation (no env,
    no replay), recording the per-chunk max/mean pairwise spread over the first
    ``first_actions`` actions. Writes the same ``chunk_divergence`` npz/png format the
    ``chunk_geometry`` stage consumes (one row per chunk, ``chunk_start`` all True)."""
    import numpy as np
    from tqdm.auto import tqdm

    dims = recording.dims
    chunk_size = int(dims["chunk_size"])
    action_dim = int(dims["action_dim"])
    # Some adapters record dims.action_dim == max_action_dim (set from the model's padded
    # head width) while the resample + stored chunk_actions live at the smaller real output
    # dim. The per-dim scale and the (k,chunk,act)-vs-scale
    # broadcast must use the REAL dim, so clamp to the recorded chunk_actions last axis.
    if recording.rollouts:
        real_adim = int(np.asarray(recording.rollouts[0].chunk_actions).shape[-1])
        if real_adim != action_dim:
            logger.info("chunk_divergence: dims.action_dim=%d but recorded chunk_actions has %d dims; "
                        "using %d (the real output dim).", action_dim, real_adim, real_adim)
            action_dim = real_adim
    n_action_steps = int(dims["n_action_steps"])
    nis = int(num_inference_steps)
    fa = max(1, min(int(first_actions), chunk_size)) if first_actions else chunk_size
    # An explicit override wins over the manifest's recorded policy_path (used when the
    # recording's checkpoint has been relocated and the manifest path is now dangling).
    policy_path = (checkpoint_override or recording.manifest.get("policy_path")
                   or recording.manifest.get("checkpoint") or checkpoint)
    if not policy_path:
        raise KeyError("recording manifest has no policy_path/checkpoint; cannot rebuild the FM head")

    # `recording` is pre-filtered to the requested rollouts by run_chunk_divergence's loader, so
    # iterate its positions (outputs are named by ro.rollout_id, which the records still carry).
    idxs = list(range(len(recording.rollouts)))
    if max_rollouts is not None:
        idxs = idxs[: int(max_rollouts)]

    logger.info("chunk_divergence (context) on run %s: %d/%d rollouts, k=%d, first_actions=%d, nis=%d",
                rd.run_id, len(idxs), len(recording.rollouts), k, fa, nis)
    out_dir = rd.chunk_divergence_dir

    # Two context-resample backends, picked by the adapter's resample contract:
    #   * default π₀.₅ schema (supports_context_resample=False) — the generic
    #     ChunkResampleSession over a build_resample_policy()-loaded PI05Policy;
    #   * custom-context adapters (supports_context_resample=True) — drive their
    #     own sample_with_context via adapter.resample_context_chunk off the rollout's
    #     ctx_arrays sidecar (ChunkResampleSession is hardcoded to π₀.₅'s schema + PI05Policy).
    # Each backend exposes the same `_resample_one(ri, c, do_repro) -> (chunks (k,chunk,action_dim),
    # reproduce_err)` closure so the per-chunk distance/recording code below is backend-agnostic.
    adapter_cls = get_model(model_name)
    if getattr(adapter_cls, "supports_context_resample", False):
        adapter = adapter_cls.build(
            str(policy_path), device=device, action_dim=action_dim, num_inference_steps=nis,
        )

        def _resample_one(ri: int, env_idx: int, c: int, do_repro: bool):
            ctx = recording.rollouts[ri].ctx_arrays
            if ctx is None:
                raise KeyError(
                    f"rollout {ri} has no captured context (ctx_arrays); custom-context resample needs "
                    f"a recording made with --record-context. has_context="
                    f"{recording.manifest.get('has_context')}")
            unit_ctx = _slice_custom_context(ctx, env_idx)
            chunks = adapter.resample_context_chunk(
                unit_ctx, c, k=k, seed=seed + c, num_inference_steps=nis
            )
            chunks = np.asarray(chunks, np.float32)[:, :, :action_dim]      # (k, chunk, action_dim)
            rerr = float("nan")
            if do_repro:
                recorded_noise = recording.rollouts[ri].noise[c, env_idx : env_idx + 1]
                redo = adapter.reproduce_context_chunk(
                    unit_ctx, c, recorded_noise, num_inference_steps=nis
                )
                redo = np.asarray(redo, np.float32)[0, :, :action_dim]      # (chunk, action_dim)
                # Compare against the recorded FM ENDPOINT (x_t[-1]), NOT chunk_actions: reproduce
                # re-runs the FM Euler loop and returns its endpoint, so the recorded denoise
                # endpoint is the apples-to-apples reference. chunk_actions is the POSTPROCESSED
                # executed slice — for some adapters that is decoded/unnormalized and only
                # n_action_steps long vs the full chunk_size FM chunk, so it can mismatch in both
                # length and space. x_t[-1] is the same space+length as redo for every
                # custom-context adapter.
                recorded_end = np.asarray(
                    recording.rollouts[ri].x_t[c, -1, env_idx], np.float32)[:, :action_dim]  # (chunk, adim)
                n = min(redo.shape[0], recorded_end.shape[0])
                rerr = float(np.abs(redo[:n] - recorded_end[:n]).max())
            return chunks, rerr
    else:
        from fmaccel.posterior.resample import ChunkResampleSession, build_resample_policy

        # Load the FM weights ONCE and inject into every per-chunk session (from_pretrained
        # is the only multi-second step and is identical across chunks of one recording).
        policy = build_resample_policy(
            str(policy_path), device=device, n_action_steps=n_action_steps, num_inference_steps=nis,
        )

        def _resample_one(ri: int, env_idx: int, c: int, do_repro: bool):
            with ChunkResampleSession(
                recording, rollout_idx=ri, env_idx=env_idx, chunk_idx=c, device=device,
                num_inference_steps=nis, use_context=True, policy=policy,
            ) as sess:
                rerr = float(sess.reproduce()["max_abs_err"]) if do_repro else float("nan")
                res = sess.resample(n_samples=k, seed=seed + c, capture_trajectory=False,
                                    micro_batch=micro_batch)
            return res["chunk_actions"].numpy().astype(np.float32), rerr   # (k, chunk, action_dim)

    units = _context_units(recording, idxs)
    suffix_all = len(units) > 1
    episodes: list[dict] = []
    worst_reproduce = 0.0

    # Per-dim standardization scale: run-pooled over every selected rollout's recorded
    # chunk actions. These are already in the FM-head normalized space (= the resampled
    # candidates' space here), so each action dim divides the distance by one fixed std
    # and dims weigh comparably (and divergence stays comparable across episodes).
    if dim_scale_override is not None:
        dim_scale = np.asarray(dim_scale_override, np.float32).reshape(-1)
        if dim_scale.shape != (action_dim,) or not np.all(np.isfinite(dim_scale)) or np.any(dim_scale <= 0):
            raise ValueError(
                f"dim_scale_override must contain {action_dim} finite positive values; got {dim_scale!r}"
            )
        scale_source = "caller-supplied run-pooled scale (shared across rollout shards)"
    else:
        scale_rows = [
            np.asarray(
                recording.rollouts[ri].chunk_actions[
                    : _valid_n_chunks_ctx(recording.rollouts[ri], env_idx), env_idx
                ],
                np.float32,
            ).reshape(-1, action_dim)
            for ri, env_idx in units
        ]
        dim_scale = _pooled_dim_scale(np.concatenate(scale_rows, axis=0)) if scale_rows else None
        scale_source = "computed from the selected rollouts"
    logger.info("per-dim distance scale (run-pooled, normalized space): %s",
                np.array2string(dim_scale, precision=3) if dim_scale is not None else "none")

    ro_bar = tqdm(units, desc="chunk_divergence(ctx)", unit="ep", disable=not progress, dynamic_ncols=True)
    for ri, env_idx in ro_bar:
        ro = recording.rollouts[ri]
        nv = _valid_n_chunks_ctx(ro, env_idx)
        env_step = np.asarray(ro.env_step_at_chunk_start[:nv], dtype=np.int64)
        max_div = np.zeros(nv, np.float32)
        mean_div = np.zeros(nv, np.float32)
        pa_max_div = np.zeros((nv, fa), np.float32) if per_action_divergence else None  # [n_chunks, window]
        pa_mean_div = np.zeros((nv, fa), np.float32) if per_action_divergence else None
        repro_err = np.full(nv, np.nan, np.float32)
        chunks_log = np.zeros((nv, k, chunk_size, action_dim), np.float32) if save_chunks else None
        ep_worst = 0.0

        ch_bar = tqdm(range(nv), desc=f"  ro{int(ro.rollout_id)}", unit="chunk",
                      leave=False, disable=not progress, dynamic_ncols=True)
        for c in ch_bar:
            # Reproduce is a fidelity GATE, not part of the GT: for captured-context
            # recordings it is exact (max_abs_err≈0), so a full B=1 denoise on every
            # chunk just to re-confirm that is ~half the per-chunk cost for no signal.
            # Run it on chunk 0 of each rollout (+ every `reproduce_every` chunks if set)
            # to keep a drift gate while skipping the redundant rest.
            do_repro = c == 0 or (reproduce_every > 0 and c % reproduce_every == 0)
            chunks_raw, rerr = _resample_one(ri, env_idx, c, do_repro)     # (k, chunk, action_dim)
            if per_action_divergence:
                mx, mn, pa_mx, pa_mn = pairwise_chunk_spread(
                    chunks_raw[:, :fa], scale=dim_scale, return_per_action=True)
                pa_max_div[c] = pa_mx
                pa_mean_div[c] = pa_mn
            else:
                mx, mn = pairwise_chunk_spread(chunks_raw[:, :fa], scale=dim_scale)  # per-dim-standardized spread
            max_div[c] = mx
            mean_div[c] = mn
            repro_err[c] = rerr
            if do_repro:
                ep_worst = max(ep_worst, rerr)
            if save_chunks:
                chunks_log[c] = chunks_raw                                # full chunk kept for re-windowing
            ch_bar.set_postfix(div=f"{mx:.2f}", refresh=False)
        ch_bar.close()

        suffix = _context_unit_suffix(ro, env_idx, multiple=suffix_all)
        max_arr = max_div.reshape(-1, 1)
        mean_arr = mean_div.reshape(-1, 1)
        start_arr = np.ones(nv, dtype=bool)                               # every unit is a re-plan
        np.save(out_dir / f"max_pairwise_dist{suffix}.npy", max_arr)
        npz: dict[str, Any] = {
            "max_pairwise_dist": max_arr,                                 # [n_chunks, 1]
            "mean_pairwise_dist": mean_arr,                               # [n_chunks, 1]
            "env_step": env_step,                                         # [n_chunks] env step at re-plan
            "chunk_start": start_arr,                                     # [n_chunks] (all True)
            "reproduce_err": repro_err,                                   # [n_chunks]
            "first_actions": np.int64(fa),
            "dim_scale": (np.asarray(dim_scale, np.float32)               # [act] per-dim standardization
                          if dim_scale is not None else np.ones(action_dim, np.float32)),
        }
        if per_action_divergence:
            npz["per_action_max_dist"] = pa_max_div                       # [n_chunks, window] within-chunk profile
            npz["per_action_mean_dist"] = pa_mean_div                     # [n_chunks, window]
        if save_chunks:
            npz["chunks"] = chunks_log                                    # [n_chunks, k, chunk, act]
        np.savez_compressed(out_dir / f"chunk_divergence{suffix}.npz", **npz)
        peak = int(np.argmax(max_arr)) if nv else -1
        repro_ok = ep_worst <= reproduce_tol
        _plot_context_divergence(
            env_step, max_div, mean_div, out_dir / f"divergence{suffix}.png",
            title=(f"{model_name}/{dataset_name} ro{int(ro.rollout_id)}: k={k} chunk spread "
                   f"over first {fa}/{chunk_size} actions"),
        )
        worst_reproduce = max(worst_reproduce, ep_worst)
        if not repro_ok:
            logger.warning(
                "REPRODUCE GATE FAILED on ro%d: max reproduce err=%.3e > tol=%.1e. The recorded "
                "context did not reproduce the recorded chunk, so the resampled spread is untrustworthy.",
                int(ro.rollout_id), ep_worst, reproduce_tol,
            )
        ep_summary = {
            "rollout_id": int(ro.rollout_id), "env_idx": int(env_idx),
            "task_id": int(ro.task_id), "n_chunks": int(nv),
            "chunk_size": chunk_size, "first_actions": fa, "peak_step": int(env_step[peak]) if peak >= 0 else -1,
            "peak_max_dist": float(max_arr[peak, 0]) if peak >= 0 else 0.0,
            "mean_max_dist": float(max_arr.mean()) if nv else 0.0,
            "max_reproduce_err": ep_worst, "reproduce_ok": repro_ok,
        }
        episodes.append(ep_summary)
        ro_bar.set_postfix(peak=f"{ep_summary['peak_max_dist']:.2f}", refresh=False)
        logger.info("ro %d: chunks=%d peak_div=%.3f@env%d mean_div=%.3f reproduce_err=%.2e",
                    int(ro.rollout_id), nv, ep_summary["peak_max_dist"], ep_summary["peak_step"],
                    ep_summary["mean_max_dist"], ep_worst)
    ro_bar.close()

    summary = {
        "source_run": rd.run_id, "model": model_name, "checkpoint": checkpoint, "dataset": dataset_name,
        "mode": "context", "k": k, "seed": seed, "n_rollouts": len(episodes), "chunk_size": chunk_size,
        "n_action_steps": n_action_steps, "num_inference_steps": nis, "first_actions": fa,
        "stride": "one resample per action chunk (at each re-plan's recorded context)",
        "trajectory": "exact recorded context (no env replay)",
        "distance": f"per-dim-standardized Euclidean L2 per action -> mean over first {fa} actions -> max over k*(k-1)/2 pairs",
        "standardization": "each action dim divided by its run-pooled winsorized std (std after clipping each dim's extreme 1% tails) over all selected rollouts' recorded chunk_actions (normalized space)",
        "dim_scale": [float(x) for x in dim_scale] if dim_scale is not None else None,
        "dim_scale_source": scale_source,
        "action_space": "normalized (FM-head output space; pi0.5 unnormalizes downstream)",
        "reproduce_tol": reproduce_tol, "worst_reproduce_err": worst_reproduce,
        "reproduce_ok": worst_reproduce <= reproduce_tol,
        "reproduce_gate": ("chunk 0 per rollout" if reproduce_every <= 0
                           else f"chunk 0 + every {reproduce_every} chunks per rollout"),
        "per_action_divergence": per_action_divergence,
        "per_action_distance": (
            f"per-action (within-chunk) pairwise L2 over the first {fa}/{chunk_size} actions, "
            "max/mean over k*(k-1)/2 pairs per action position -> per_action_{max,mean}_dist [n_chunks, "
            f"{fa}]" if per_action_divergence else None),
        "episodes": episodes,
        "outputs": {
            "headline": "max_pairwise_dist[_roN].npy  [n_chunks, 1]",
            "npz": "chunk_divergence[_roN].npz", "plot": "divergence[_roN].png",
            **({"per_action": "per_action_{max,mean}_dist in the npz  [n_chunks, window]"}
               if per_action_divergence else {}),
        },
    }
    write_json(out_dir / "meta.json", summary)
    rd.record_stage("chunk_divergence", {
        "mode": "context", "k": k, "n_rollouts": len(episodes), "first_actions": fa,
        "mean_max_dist": float(np.mean([e["mean_max_dist"] for e in episodes])) if episodes else 0.0,
        "worst_reproduce_err": worst_reproduce, "reproduce_ok": summary["reproduce_ok"],
    })
    logger.info("chunk_divergence (context) done -> %s (worst reproduce err=%.2e)", out_dir, worst_reproduce)
    return rd, summary


# --------------------------------------------------------------------------- driver
def run_chunk_divergence(
    run: Any,                     # run-id or RunDir of the *producing* run (its fm/ recording)
    *,
    k: int = 32,
    rollouts: Sequence[int] | None = None,
    max_rollouts: int | None = None,
    max_steps: int | None = None,
    device: str = "cuda",
    seed: int = 0,
    save_chunks: bool = True,
    num_inference_steps: int | None = None,
    micro_batch: int | None = None,
    reproduce_tol: float | None = None,
    first_actions: int | None = None,
    reproduce_every: int = 0,
    dataset_root: str | None = None,
    per_action_divergence: bool = True,
    checkpoint_override: str | None = None,
    dim_scale_override: Sequence[float] | None = None,
    progress: bool = True,
) -> tuple[runs.RunDir, dict]:
    """Read an existing run's recorded episodes; for each, resample ``k`` chunks and
    record the per-unit max chunk spread into ``<run>/chunk_divergence/``.

    ``checkpoint_override`` (optional): rebuild the FM head from THIS checkpoint path
    instead of the one recorded in the run/manifest. Use it when the recording's
    checkpoint has been relocated on disk (the manifest ``policy_path`` is now dangling)
    — the reproduce gate then re-verifies that the substituted weights still reproduce
    the recorded chunk (a large reproduce err means the override is the wrong model).

    Two producers, picked by the recording's ``has_context`` flag:
      * **gym replay** (``has_context=False``, a gym-env producer) — reset the gym env
        at the episode seed, replay the recorded executed actions, and resample at
        *every visited observation* (dense stride).
      * **recorded context** (``has_context=True``, π₀.₅ on LIBERO) — rebuild
        the exact recorded conditioning per chunk (no env, no replay) and resample at
        *each chunk's re-plan observation* (one k-candidate set per chunk).

    ``first_actions`` restricts the chunk-pair distance to the first N actions of the
    chunk (default: the whole chunk). With ``n_action_steps < chunk_size`` and RTC off,
    only the first ``n_action_steps`` actions of each chunk are ever executed, so
    ``first_actions=n_action_steps`` measures the spread of the *executed* plan.

    ``per_action_divergence`` (default ``True``) also stores the within-chunk
    ``per_action_{max,mean}_dist`` ``[n, window]`` profile in the npz — the per-action
    breakdown the headline scalars collapse with a chunk-mean — over the same
    ``first_actions`` window. Returns ``(run, summary)``."""
    import numpy as np
    import torch
    from tqdm.auto import tqdm

    rd = run if isinstance(run, runs.RunDir) else runs.resolve_run(run)
    meta = rd.meta
    model_name = str(meta.model.get("name"))
    checkpoint = checkpoint_override or meta.model.get("checkpoint")
    dataset_name = str(meta.dataset.get("name"))
    # Fidelity-gate tolerance: default from the adapter (an fp32 head reproduces to ~1e-2; a
    # bf16-autocast head needs a looser bound) unless the caller pinned one. Resolving the adapter class is import-light (lazy registry).
    if reproduce_tol is None:
        reproduce_tol = float(getattr(get_model(model_name), "resample_reproduce_tol", 1e-2))

    # Load ONLY this shard's rollouts (memory-bounded: each context sidecar is tens of MB, so
    # loading a 2000-rollout run at once would need ~100GB RAM). rollouts are rollout_ids;
    # hand them to the loader filter. The sub-functions then iterate the loaded list's positions
    # (they remap idxs to range(len(recording.rollouts))), and name outputs by ro.rollout_id.
    _want = None if rollouts is None else [int(i) for i in rollouts]
    recording = FMRecording.load(rd.fm_dir, rollout_ids=_want)
    if not recording.rollouts:
        raise ValueError(f"run {rd.run_id!r} has no recorded rollouts under {rd.fm_dir}")
    dims = recording.dims
    n_action_steps = int(dims["n_action_steps"])
    nis = int(num_inference_steps) if num_inference_steps is not None else int(dims["num_inference_steps"])

    # Recordings that carry exact captured context (π₀.₅ on LIBERO) rebuild the
    # recorded conditioning directly — no gym env, no replay — so they take the
    # context-resample path. Gym-env recordings (has_context=False) fall through
    # exact conditioning each decision saw, so no env and no replay are involved.
    if bool(recording.manifest.get("has_context", False)):
        return _run_context_divergence(
            rd, recording, model_name=model_name, checkpoint=checkpoint, dataset_name=dataset_name,
            k=k, rollouts=rollouts, max_rollouts=max_rollouts, device=device, seed=seed,
            save_chunks=save_chunks, num_inference_steps=nis, micro_batch=micro_batch,
            reproduce_tol=reproduce_tol, first_actions=first_actions,
            reproduce_every=reproduce_every, per_action_divergence=per_action_divergence,
            checkpoint_override=checkpoint_override, dim_scale_override=dim_scale_override,
            progress=progress,
        )

    raise NotImplementedError(
        f"run {rd.run_id!r} has has_context=False. The resample ground truth must be drawn at the "
        f"exact conditioning the policy faced, which is reconstructed from the --record-context "
        f"sidecar — re-record with `--record-fm --record-context`."
    )
