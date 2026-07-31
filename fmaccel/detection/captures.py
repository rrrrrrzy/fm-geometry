"""GPU capture stages: the three post-hoc reconstructions detectors need.

Everything in :mod:`fmaccel.detection.score` is lerobot-free and reads numpy off disk. Three
detector families need something the recorder does not store in the hot path, so this module
rebuilds each **post-hoc** from the ``--record-context`` sidecar (no env, no rollout) and writes it
next to the recording, where the scoring side picks it up:

* :func:`run_obs_emb` -> ``<run>/obs_emb/`` — pooled observation/prefix conditioning
  (``ChunkRecord.obs_emb``), for the embedding-OOD family: ``rnd_oe`` / ``logpzo`` /
  ``pca_kmeans`` / ``knn`` / ``mahalanobis`` / ``fiper``.
* :func:`run_hidden_states` -> ``<run>/hidden_states/`` — the action expert's last-layer hidden
  feature (``ChunkRecord.hidden['action_expert_last']``), for the supervised ``safe`` probe.
* :func:`run_fm_loss_score` -> ``<run>/fm_loss/`` — the Diff-DAgger re-noised flow-matching loss,
  which is model-in-the-loop and so cannot be read off the recorded denoise path at all.

All three need the policy on a GPU and an adapter that advertises the matching capability flag
(``supports_obs_emb`` / ``supports_hidden`` / ``supports_fm_velocity``); the consumer side stays
numpy-only. Each embedding is the SAME routine used at detector ``fit`` and ``score`` time — that
identity is the contract these stages exist to guarantee.

See ``docs/baselines.md`` for the per-detector fidelity notes.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from fmaccel.core import runs
from fmaccel.core.io import write_json
from fmaccel.detectors.base import ChunkRecord
from fmaccel.detectors.fm_loss import FmLossDetector
from fmaccel.posterior.divergence import (
    _context_unit_suffix,
    _context_units,
    _slice_custom_context,
    _valid_n_chunks_ctx,
)
from fmaccel.recording.loader import FMRecording
from fmaccel.registry import get_model

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- obs embeddings
def run_obs_emb(
    run: Any,
    *,
    rollouts: Sequence[int] | None = None,
    max_rollouts: int | None = None,
    device: str = "cuda",
    progress: bool = True,
) -> tuple[runs.RunDir, dict]:
    """Compute + store per-chunk obs embeddings for a context-capturing run.

    Writes ``<run>/obs_emb/obs_emb_ro<rollout_id>.npz`` (``obs_emb`` ``(n_chunks, d)``,
    ``rollout_id``, ``n_chunks``) + ``meta.json``. Returns ``(RunDir, summary)``."""
    from tqdm.auto import tqdm

    rd = run if isinstance(run, runs.RunDir) else runs.resolve_run(run)
    # Load ONLY this shard's rollouts (memory-bounded: each context sidecar is tens of MB, so
    # loading all 2000 needs ~100GB). rollouts here are rollout_ids → hand them straight to the
    # loader filter; the loaded list then IS exactly the requested slice.
    _want = None if rollouts is None else [int(i) for i in rollouts]
    recording = FMRecording.load(rd.fm_dir, rollout_ids=_want)
    if not recording.rollouts:
        raise ValueError(f"run {rd.run_id!r} has no recorded rollouts under {rd.fm_dir}")
    if not bool(recording.manifest.get("has_context", False)):
        raise ValueError(
            f"run {rd.run_id!r} has has_context=False; obs_emb needs a --record-context recording "
            f"(the prefix conditioning is reconstructed from the ctx_* sidecar).")

    model_name = str(rd.meta.model.get("name"))
    checkpoint = recording.manifest.get("policy_path") or recording.manifest.get("checkpoint") \
        or rd.meta.model.get("checkpoint")
    adapter_cls = get_model(model_name)
    if not getattr(adapter_cls, "supports_obs_emb", False):
        raise NotImplementedError(
            f"model {model_name!r} has supports_obs_emb=False (no embed_context_obs); the "
            f"embedding-OOD detectors are unavailable for it.")

    action_dim = int(recording.dims["action_dim"])
    nis = int(recording.dims["num_inference_steps"])
    adapter = adapter_cls.build(str(checkpoint), device=device, action_dim=action_dim,
                                num_inference_steps=nis)

    # Recording is already filtered to the requested rollouts (or all), so iterate its positions.
    idxs = list(range(len(recording.rollouts)))
    if max_rollouts is not None:
        idxs = idxs[: int(max_rollouts)]

    out_dir = rd.stage_dir("obs_emb")
    logger.info("obs_emb on run %s: %d/%d rollouts (model=%s)", rd.run_id, len(idxs),
                len(recording.rollouts), model_name)

    units = _context_units(recording, idxs)
    total_units = sum(int(r.get("batch_size", 1)) for r in recording.manifest["rollouts"])
    episodes: list[dict] = []
    emb_dim = None
    ro_bar = tqdm(units, desc="obs_emb", unit="ep", disable=not progress, dynamic_ncols=True)
    for ri, env_idx in ro_bar:
        ro = recording.rollouts[ri]
        ctx = ro.ctx_arrays
        if ctx is None:
            logger.warning("rollout %d has no ctx_arrays; skipping (no context sidecar).", ri)
            continue
        nv = _valid_n_chunks_ctx(ro, env_idx)
        unit_ctx = _slice_custom_context(ctx, env_idx) if ro.batch_size > 1 else ctx
        embs = np.stack([adapter.embed_context_obs(unit_ctx, c) for c in range(nv)]).astype(np.float32)
        emb_dim = int(embs.shape[1])
        suffix = _context_unit_suffix(ro, env_idx, multiple=total_units > 1)
        np.savez_compressed(out_dir / f"obs_emb{suffix}.npz",
                            obs_emb=embs, rollout_id=np.int64(int(ro.rollout_id)),
                            env_idx=np.int64(env_idx), n_chunks=np.int64(nv))
        episodes.append({"rollout_id": int(ro.rollout_id), "task_id": int(ro.task_id),
                         "env_idx": int(env_idx), "n_chunks": int(nv)})
        ro_bar.set_postfix(d=emb_dim, refresh=False)
    ro_bar.close()

    summary = {
        "source_run": rd.run_id, "model": model_name, "checkpoint": str(checkpoint),
        "n_rollouts": len(episodes), "emb_dim": emb_dim,
        "pooling": "masked-mean over embed_prefix tokens (valid pad_masks)",
        "episodes": episodes,
        "outputs": {"per_rollout": "obs_emb_ro<rollout_id>.npz (obs_emb [n_chunks, d])"},
    }
    write_json(out_dir / "meta.json", summary)
    rd.record_stage("obs_emb", {"n_rollouts": len(episodes), "emb_dim": emb_dim})
    logger.info("obs_emb done -> %s (%d rollouts, d=%s)", out_dir, len(episodes), emb_dim)
    return rd, summary


# ------------------------------------------------- action-expert hidden features
def run_hidden_states(
    run: Any,
    *,
    rollouts: Sequence[int] | None = None,
    max_rollouts: int | None = None,
    horizon_reduce: str = "mean",
    diff_reduce: str = "mean",
    device: str = "cuda",
    progress: bool = True,
) -> tuple[runs.RunDir, dict]:
    """Compute + store per-chunk action-expert last-hidden features for a context-capturing run.

    Writes ``<run>/hidden_states/hidden_ro<rollout_id>.npz`` (``hidden`` ``(n_chunks, d)``,
    ``rollout_id``, ``n_chunks``) + ``meta.json``. ``horizon_reduce`` / ``diff_reduce`` ∈
    {"mean","first","last"} are SAFE's grid-searched aggregation choices (default mean/mean =
    PizeroDatasetConfig). Returns ``(RunDir, summary)``."""
    from tqdm.auto import tqdm

    rd = run if isinstance(run, runs.RunDir) else runs.resolve_run(run)
    # Load ONLY this shard's rollouts (context sidecars are large; see obs_emb note).
    _want = None if rollouts is None else [int(i) for i in rollouts]
    recording = FMRecording.load(rd.fm_dir, rollout_ids=_want)
    if not recording.rollouts:
        raise ValueError(f"run {rd.run_id!r} has no recorded rollouts under {rd.fm_dir}")
    if not bool(recording.manifest.get("has_context", False)):
        raise ValueError(
            f"run {rd.run_id!r} has has_context=False; hidden_states needs a --record-context "
            f"recording (the prefix conditioning is reconstructed from the ctx_* sidecar).")

    model_name = str(rd.meta.model.get("name"))
    checkpoint = recording.manifest.get("policy_path") or recording.manifest.get("checkpoint") \
        or rd.meta.model.get("checkpoint")
    adapter_cls = get_model(model_name)
    if not getattr(adapter_cls, "supports_hidden", False):
        raise NotImplementedError(
            f"model {model_name!r} has supports_hidden=False (no embed_context_hidden); the SAFE "
            f"supervised probe is unavailable for it.")

    action_dim = int(recording.dims["action_dim"])
    nis = int(recording.dims["num_inference_steps"])
    adapter = adapter_cls.build(str(checkpoint), device=device, action_dim=action_dim,
                                num_inference_steps=nis)

    # Recording is already filtered to the requested rollouts (or all), so iterate its positions.
    idxs = list(range(len(recording.rollouts)))
    if max_rollouts is not None:
        idxs = idxs[: int(max_rollouts)]

    out_dir = rd.stage_dir("hidden_states")
    logger.info("hidden_states on run %s: %d/%d rollouts (model=%s, reduce=%s/%s)", rd.run_id,
                len(idxs), len(recording.rollouts), model_name, horizon_reduce, diff_reduce)

    units = _context_units(recording, idxs)
    total_units = sum(int(r.get("batch_size", 1)) for r in recording.manifest["rollouts"])
    episodes: list[dict] = []
    hid_dim = None
    ro_bar = tqdm(units, desc="hidden_states", unit="ep", disable=not progress, dynamic_ncols=True)
    for ri, env_idx in ro_bar:
        ro = recording.rollouts[ri]
        ctx = ro.ctx_arrays
        if ctx is None:
            logger.warning("rollout %d has no ctx_arrays; skipping (no context sidecar).", ri)
            continue
        nv = _valid_n_chunks_ctx(ro, env_idx)
        unit_ctx = _slice_custom_context(ctx, env_idx) if ro.batch_size > 1 else ctx
        feats = np.stack([
            adapter.embed_context_hidden(unit_ctx, c, horizon_reduce=horizon_reduce, diff_reduce=diff_reduce)
            for c in range(nv)
        ]).astype(np.float32)
        hid_dim = int(feats.shape[1])
        suffix = _context_unit_suffix(ro, env_idx, multiple=total_units > 1)
        np.savez_compressed(out_dir / f"hidden{suffix}.npz",
                            hidden=feats, rollout_id=np.int64(int(ro.rollout_id)),
                            env_idx=np.int64(env_idx), n_chunks=np.int64(nv))
        episodes.append({"rollout_id": int(ro.rollout_id), "task_id": int(ro.task_id),
                         "env_idx": int(env_idx), "n_chunks": int(nv)})
        ro_bar.set_postfix(d=hid_dim, refresh=False)
    ro_bar.close()

    summary = {
        "source_run": rd.run_id, "model": model_name, "checkpoint": str(checkpoint),
        "n_rollouts": len(episodes), "hidden_dim": hid_dim,
        "reduction": f"horizon={horizon_reduce}, diff_step={diff_reduce} (SAFE PizeroDatasetConfig)",
        "feature": "action_expert_last (suffix_out before action_out_proj)",
        "episodes": episodes,
        "outputs": {"per_rollout": "hidden_ro<rollout_id>.npz (hidden [n_chunks, d])"},
    }
    write_json(out_dir / "meta.json", summary)
    rd.record_stage("hidden_states", {"n_rollouts": len(episodes), "hidden_dim": hid_dim})
    logger.info("hidden_states done -> %s (%d rollouts, d=%s)", out_dir, len(episodes), hid_dim)
    return rd, summary


# ------------------------------------------------------ fm_loss (Diff-DAgger) score
def _velocity_contract_err(adapter: Any, ro: Any, action_dim: int, env_idx: int = 0) -> float:
    """Max ``|v_closure − v_recorded|`` over chunk 0's recorded denoise steps — the REAL fm_loss
    contract gate: the closure must reproduce the recorded velocity ``v_t`` bit-for-bit (validates
    the closure/sign/time wiring; the *absolute* fm_loss floor is a separate, model-dependent
    quantity — see the gate note below).

    TWO conventions (``adapter.fm_velocity_matches_recorded``):
      * pi05 (False): native ``x_s=s·noise+(1−s)·action``, ``s:1→0`` — the fm_loss clean-data
        closure returns ``−v_θ(x_τ, τ=1−s)``, so recover the recorded ``v_θ`` as ``−closure(x,1−s)``.
      * clean-data-native heads (True): native direction ``x_τ=(1−τ)·noise+τ·action``,
        ``v=action−noise`` at the SAME time the recorder stored — the closure returns ``+v_θ(x,s)``
        directly, so compare ``closure(x,s)`` to ``v_t`` with no flip / no time reversal."""
    matches = bool(getattr(adapter, "fm_velocity_matches_recorded", False))
    ctx = (_slice_custom_context(ro.ctx_arrays, env_idx)
           if ro.batch_size > 1 else ro.ctx_arrays)
    fmv = adapter.velocity_fn_for_context(ctx, 0)
    xt = np.asarray(ro.x_t[0, :, env_idx, :, :], np.float32)   # (T+1, chunk, max_adim)
    vt = np.asarray(ro.v_t[0, :, env_idx, :, :], np.float32)   # (T,   chunk, max_adim)
    times = np.asarray(ro.time[0], np.float32)           # (T,) recorded FM times
    errs = []
    for step in range(len(times)):
        s = float(times[step])
        t_eval = s if matches else (1.0 - s)             # clean-data-native time vs pi05's 1−s
        v_clean = np.asarray(fmv(xt[step][None, :, :], np.array([t_eval], np.float32)))  # (1,chunk,act)
        v_theta = v_clean[0] if matches else -v_clean[0]  # clean-data-native: no flip; pi05: negate
        errs.append(float(np.abs(v_theta - vt[step][:, :v_theta.shape[1]]).max()))
    return max(errs) if errs else float("nan")


def run_fm_loss_score(
    run: Any,
    *,
    n_exec: int | None = None,
    m_t: int = 16,
    m_noise: int = 2,
    rollouts: Sequence[int] | None = None,
    max_rollouts: int | None = None,
    device: str = "cuda",
    contract_tol: float = 0.12,
    progress: bool = True,
) -> tuple[runs.RunDir, dict]:
    """Compute + store the per-chunk fm_loss score for a context-capturing run.

    Writes ``<run>/fm_loss/fm_loss_ro<id>.npz`` (``fm_loss`` ``(n_chunks,)``) + ``meta.json``.
    Gates on the **velocity contract** (the closure must reproduce the recorded ``v_t``; see
    :func:`_velocity_contract_err`) rather than an absolute ID-loss threshold — the absolute
    fm_loss floor is NOT ~0 for a multimodal flow policy (the marginal velocity ``v_θ(x_τ,τ)``
    differs from the straight-line conditional ``â−x0`` even in-distribution), so a low-loss-ID
    gate is the wrong test here.

    ``contract_tol`` defaults to ``0.12`` (NOT ~0): bf16 vision/VLM towers are not
    bit-reproducible across the record→re-forward boundary (same drift the chunk_divergence
    reproduce gate sees, ~0.04–0.09, concentrated at the late low-``s`` steps where the
    multimodal gripper decision is steep). The closure is provably exact when run in the SAME
    forward (verified: matches recorded ``v_t`` to 0.0 at every step in-process); a *systematic*
    sign/time/cache wiring bug would be O(1), far above this floor. Returns ``(RunDir, summary)``."""
    from tqdm.auto import tqdm

    rd = run if isinstance(run, runs.RunDir) else runs.resolve_run(run)
    # Load ONLY this shard's rollouts (context sidecars are large; see obs_emb note).
    _want = None if rollouts is None else [int(i) for i in rollouts]
    recording = FMRecording.load(rd.fm_dir, rollout_ids=_want)
    if not recording.rollouts:
        raise ValueError(f"run {rd.run_id!r} has no recorded rollouts under {rd.fm_dir}")
    if not bool(recording.manifest.get("has_context", False)):
        raise ValueError(f"run {rd.run_id!r} has has_context=False; fm_loss needs --record-context.")

    model_name = str(rd.meta.model.get("name"))
    checkpoint = recording.manifest.get("policy_path") or recording.manifest.get("checkpoint") \
        or rd.meta.model.get("checkpoint")
    adapter_cls = get_model(model_name)
    if not getattr(adapter_cls, "supports_fm_velocity", False):
        raise NotImplementedError(
            f"model {model_name!r} has supports_fm_velocity=False (no velocity_fn_for_context); "
            f"the fm_loss detector is unavailable for it.")

    dims = recording.dims
    action_dim = int(dims["action_dim"])
    # Clamp to the real output dim (an adapter may record the model's padded head width while the
    # chunk lives at fewer dims). Scoring the FM loss over the zero-pad dims would add a constant
    # noise floor; the recorded
    # chunk_actions last axis is the truth (no-op where dims.action_dim already equals it).
    real_adim = int(np.asarray(recording.rollouts[0].chunk_actions).shape[-1])
    if real_adim != action_dim:
        logger.info("fm_loss: dims.action_dim=%d but recorded chunk_actions has %d dims; using %d.",
                    action_dim, real_adim, real_adim)
        action_dim = real_adim
    chunk_size = int(dims["chunk_size"])
    nis = int(dims["num_inference_steps"])
    if n_exec is None:
        n_exec = min(int(dims["n_action_steps"]), chunk_size)
    adapter = adapter_cls.build(str(checkpoint), device=device, action_dim=action_dim,
                                num_inference_steps=nis)
    det = FmLossDetector(m_t=int(m_t), m_noise=int(m_noise))

    # Recording is already filtered to the requested rollouts (or all), so iterate its positions.
    idxs = list(range(len(recording.rollouts)))
    if max_rollouts is not None:
        idxs = idxs[: int(max_rollouts)]

    out_dir = rd.stage_dir("fm_loss")

    # Velocity-contract gate (run ONCE): the closure must reproduce the recorded v_t bit-for-bit.
    units = _context_units(recording, idxs)
    total_units = sum(int(r.get("batch_size", 1)) for r in recording.manifest["rollouts"])
    contract_err = None
    for ri, env_idx in units:
        if recording.rollouts[ri].ctx_arrays is not None:
            contract_err = _velocity_contract_err(adapter, recording.rollouts[ri], action_dim, env_idx)
            break
    contract_ok = contract_err is not None and contract_err <= float(contract_tol)
    if contract_err is None:
        logger.warning("fm_loss: no rollout has ctx_arrays; cannot run the velocity-contract gate.")
    elif contract_ok:
        logger.info("fm_loss velocity-contract gate OK: max|v_closure - v_recorded| = %.2e <= %.1e",
                    contract_err, contract_tol)
    else:
        logger.warning(
            "fm_loss VELOCITY-CONTRACT GATE FAILED: max|v_closure - v_recorded| = %.3e > tol %.1e — the "
            "velocity callback is mis-wired (sign/time/cache); fm_loss numbers are UNTRUSTWORTHY. See "
            "detectors/fm_loss.py VELOCITY CONTRACT.", contract_err, contract_tol)

    episodes: list[dict] = []
    ro_bar = tqdm(units, desc="fm_loss", unit="ep", disable=not progress, dynamic_ncols=True)
    for ri, env_idx in ro_bar:
        ro = recording.rollouts[ri]
        ctx = ro.ctx_arrays
        if ctx is None:
            logger.warning("rollout %d has no ctx_arrays; skipping.", ri)
            continue
        nv = _valid_n_chunks_ctx(ro, env_idx)
        unit_ctx = _slice_custom_context(ctx, env_idx) if ro.batch_size > 1 else ctx
        scores = np.full(nv, np.nan, np.float32)
        for c in range(nv):
            fmv = adapter.velocity_fn_for_context(unit_ctx, c)
            xc = np.asarray(ro.x_t[c, :, env_idx, :, :], np.float32)  # (T+1, chunk, max_adim)
            rec = ChunkRecord(x_t=xc, action_dim=action_dim, n_exec=int(n_exec),
                              env_step=c, task_id=int(ro.task_id), extra={"fm_velocity": fmv})
            scores[c] = float(det.score(rec))
        suffix = _context_unit_suffix(ro, env_idx, multiple=total_units > 1)
        np.savez_compressed(out_dir / f"fm_loss{suffix}.npz",
                            fm_loss=scores, rollout_id=np.int64(int(ro.rollout_id)),
                            env_idx=np.int64(env_idx), n_chunks=np.int64(nv))
        episodes.append({"rollout_id": int(ro.rollout_id), "task_id": int(ro.task_id),
                         "env_idx": int(env_idx), "n_chunks": int(nv),
                         "mean_fm_loss": float(np.nanmean(scores))})
        ro_bar.set_postfix(loss=f"{np.nanmean(scores):.3f}", refresh=False)
    ro_bar.close()

    summary = {
        "source_run": rd.run_id, "model": model_name, "checkpoint": str(checkpoint),
        "n_rollouts": len(episodes), "m_t": int(m_t), "m_noise": int(m_noise),
        "n_exec": int(n_exec), "contract_err": contract_err, "contract_tol": float(contract_tol),
        "contract_ok": contract_ok,
        "note": ("fm_loss absolute floor is model-dependent (multimodal flow: marginal velocity != "
                 "straight-line conditional); the gate validates the velocity CLOSURE (reproduces "
                 "recorded v_t), not an absolute ID-loss threshold. Whether fm_loss discriminates "
                 "success vs failure is the empirical AUROC on the real run."),
        "episodes": episodes,
        "outputs": {"per_rollout": "fm_loss_ro<rollout_id>.npz (fm_loss [n_chunks])"},
    }
    write_json(out_dir / "meta.json", summary)
    rd.record_stage("fm_loss", {"n_rollouts": len(episodes), "contract_err": contract_err,
                                "contract_ok": contract_ok})
    logger.info("fm_loss done -> %s (%d rollouts, contract_err=%.2e ok=%s)", out_dir, len(episodes),
                contract_err if contract_err is not None else float("nan"), contract_ok)
    return rd, summary
