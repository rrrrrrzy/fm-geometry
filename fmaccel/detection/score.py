"""Post-hoc detector scoring: run failure-detectors over a recorded run, into the FD format.

The general post-hoc counterpart of the online server scoring: given a run that has an FM
recording (``<run>/fm/`` → the per-chunk denoise path ``x_t``) and optionally the resample
artifact (``<run>/chunk_divergence/*.npz`` → the ``k`` candidate plans per chunk), score every
requested :class:`~fmaccel.detectors.base.Detector` per chunk and emit the per-episode JSON
layout :mod:`fmaccel.detection.cusum` consumes
(``<out>/<task_group>/<Task>.json`` → ``tasks.<name>.episodes[].{success, n_chunks,
<name>_scores}``), then run the shared metric battery so accel/straightness/sparc (from ``x_t``)
and ace/oracle (from the resample candidates) are all compared identically.

This is where the **resample-based** detectors (ace/oracle) get scored — they read the ``k``
candidates the ``chunk_divergence`` stage already saved (numpy, no GPU, no model load), exactly the
artifact :mod:`fmaccel.geometry.accel` validates accel against. A detector whose signal
is missing for a rollout (e.g. ace with no ``chunk_divergence`` npz) is simply skipped for that
rollout; a temporal detector undefined at the first chunk is forward-filled.

Success label is a proxy from the recording (episode ended via ``terminated`` = task success vs only
``truncated`` = timeout); pass ``success_by_rollout`` to override from an external eval JSON. Pure
numpy (+ the detectors' own deps); reuses :func:`chunk_geometry._load_div_chunks` /
``_valid_n_chunks`` so the resample chunks are read identically to the geometry leg.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fmaccel.core import runs
from fmaccel.core.io import write_json
from fmaccel.detectors import get_detector
from fmaccel.detectors.base import ChunkRecord
from fmaccel.geometry.accel import _load_div_chunks, _valid_n_chunks
from fmaccel.detection.cusum import run_failure_detection_eval
from fmaccel.recording.loader import FMRecording

logger = logging.getLogger(__name__)


def _sanitize(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(s)).strip("_") or "task"


def _episode_success(ro: Any, env_idx: int = 0) -> bool:
    """Success proxy from the recording: the episode ended via ``terminated`` (task success)
    rather than only ``truncated`` (timeout) — LIBERO terminates
    on success. Override with an external label source when available (closed-loop eval JSON)."""
    return bool(np.asarray(ro.terminated[:, env_idx], bool).any())


def _unit_suffix(rollout_id: int, env_idx: int, batch_size: int) -> str:
    return f"_ro{int(rollout_id)}_e{env_idx}" if batch_size > 1 else f"_ro{int(rollout_id)}"


def _resample_subset_indices(stored_k: int, requested_k: int | None, *,
                             seed: int, rollout_id: int, env_idx: int) -> np.ndarray:
    """Deterministically choose a nested MC-resample subset for one rollout unit.

    ``chunk_divergence`` keeps the complete candidate axis, so k-ablation experiments do not
    need another model forward.  A seed-specific subset is shared by all decisions in one
    rollout (candidate order is immaterial to every detector), while rollout/env ids keep the
    draw stable if the caller scores only a subset of the recording.

    ``requested_k=None`` means use every stored candidate.  Asking for more candidates than the
    artifact contains is an error rather than a silent effective-k change.
    """
    stored_k = int(stored_k)
    if stored_k < 1:
        raise ValueError(f"stored resample candidate count must be positive, got {stored_k}")
    if requested_k is None:
        return np.arange(stored_k, dtype=np.int64)
    requested_k = int(requested_k)
    if requested_k < 1:
        raise ValueError(f"resample_k must be >=1 or None, got {requested_k}")
    if requested_k > stored_k:
        raise ValueError(
            f"requested resample_k={requested_k}, but the saved artifact has only k={stored_k}")
    if requested_k == stored_k:
        return np.arange(stored_k, dtype=np.int64)
    entropy = [int(seed) & 0xFFFFFFFF, int(rollout_id) & 0xFFFFFFFF, int(env_idx) & 0xFFFFFFFF]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return np.sort(rng.choice(stored_k, size=requested_k, replace=False)).astype(np.int64)


def _load_obs_emb(oe_dir: Path, rollout_id: int, env_idx: int = 0,
                  batch_size: int = 1) -> np.ndarray | None:
    """Per-chunk obs embeddings ``(n_chunks, d)`` for one rollout from the ``obs_emb`` stage, or
    ``None`` if absent (the embedding-OOD detectors are then skipped for that rollout)."""
    p = oe_dir / f"obs_emb{_unit_suffix(rollout_id, env_idx, batch_size)}.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return np.asarray(d["obs_emb"], np.float32)


def _load_hidden(hs_dir: Path, rollout_id: int, env_idx: int = 0,
                 batch_size: int = 1) -> np.ndarray | None:
    """Per-chunk reduced action-expert last-hidden feature ``(n_chunks, d)`` for one rollout from
    the ``hidden_states`` stage, or ``None`` if absent (the SAFE probe is then skipped)."""
    p = hs_dir / f"hidden{_unit_suffix(rollout_id, env_idx, batch_size)}.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return np.asarray(d["hidden"], np.float32)


def _load_precomputed(stage_dir: Path, prefix: str, key: str, rollout_id: int,
                      env_idx: int = 0, batch_size: int = 1) -> np.ndarray | None:
    """Per-chunk precomputed score stream ``(n_chunks,)`` from a GPU stage (e.g. ``fm_loss``), or
    ``None`` if absent. The model-in-the-loop detectors run their own GPU stage and land here, so
    the lerobot-free ``detector_score`` just reads the stream and routes it through the harness."""
    p = stage_dir / f"{prefix}{_unit_suffix(rollout_id, env_idx, batch_size)}.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return np.asarray(d[key], np.float32)


def build_chunk_records(
    x_t: np.ndarray,
    chunks_div: np.ndarray | None,
    n_valid: int,
    *,
    action_dim: int,
    n_exec: int,
    chunks_div_full: np.ndarray | None = None,
    task_id: int | None = None,
    task_group: str | None = None,
    dim_scale: np.ndarray | None = None,
    obs_emb: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
    hidden_key: str = "action_expert_last",
) -> list[ChunkRecord]:
    """One :class:`ChunkRecord` per chunk of ONE rollout (the unit the detectors read).

    ``x_t`` is ``(n_chunks, T+1, K, D)`` (env already sliced); ``chunks_div`` ``(n_div, k, fa, act)``
    or ``None``; ``obs_emb`` ``(n_chunks, d)`` (the pooled prefix embedding from the ``obs_emb``
    stage) or ``None``; ``hidden`` ``(n_chunks, d)`` (the reduced action-expert last-hidden feature
    from the ``hidden_states`` stage, for SAFE) or ``None``. Consecutive records also carry the
    PREVIOUS decision's resample set as ``prev_resample`` (for the temporal-MMD detector STAC). The
    records carry whatever signals are available; a detector whose ``requires`` aren't met is skipped
    per-chunk by the scorer.

    ``chunks_div_full`` ``(n_div, k, chunk, act)`` is the SAME resample set over the full chunk
    horizon (``chunks_div`` is its ``[:, :, :first_actions]`` executed-window slice). It populates
    ``resample_full`` / ``prev_resample_full`` — cross-time detectors need it because two consecutive
    *executed* windows cover disjoint absolute timesteps (see ``ChunkRecord``)."""
    extra = {"dim_scale": np.asarray(dim_scale, np.float32)} if dim_scale is not None else {}
    recs: list[ChunkRecord] = []
    prev_resample_c = None       # previous chunk's executed-window set (ace / oracle / fiper)
    prev_resample_full_c = None  # previous chunk's FULL-horizon set, threaded forward for STAC
    for c in range(int(n_valid)):
        resample_c = None
        if chunks_div is not None and c < len(chunks_div):
            resample_c = np.asarray(chunks_div[c], np.float32)  # (k, fa, act)
        resample_full_c = None
        if chunks_div_full is not None and c < len(chunks_div_full):
            resample_full_c = np.asarray(chunks_div_full[c], np.float32)  # (k, chunk, act)
        emb_c = None
        if obs_emb is not None and c < len(obs_emb):
            emb_c = np.asarray(obs_emb[c], np.float32)          # (d,)
        hid_c = None
        if hidden is not None and c < len(hidden):
            hid_c = {hidden_key: np.asarray(hidden[c], np.float32)}  # {key: (d,)}
        recs.append(ChunkRecord(
            x_t=np.asarray(x_t[c], np.float32), action_dim=action_dim, n_exec=n_exec,
            env_step=c, task_id=task_id, task_group=task_group,
            resample=resample_c, prev_resample=prev_resample_c, obs_emb=emb_c, hidden=hid_c,
            resample_full=resample_full_c, prev_resample_full=prev_resample_full_c,
            extra=dict(extra)))
        prev_resample_c = resample_c  # this decision's resample becomes the next decision's prev
        prev_resample_full_c = resample_full_c
    return recs


def score_records(recs: Sequence[ChunkRecord], dets: Sequence[Any]) -> dict[str, list[float]]:
    """Per-chunk detector streams for ONE rollout's :class:`ChunkRecord` list.

    Returns ``{detector_name: [per-chunk score]}`` only for detectors whose required signals were
    available across the rollout (leading-NaN forward-filled; interior gaps or no data → dropped).
    """
    from fmaccel.detectors.base import Detector as _Detector

    streams: dict[str, list[float]] = {}
    for d in dets:
        recurrent = type(d).score_stream is not _Detector.score_stream
        if recurrent:
            # A history-dependent probe (SAFE-LSTM: s_t = σ(LSTM(e_{0:t}))) must see the whole
            # ordered episode at once. Score the sub-sequence of records that carry its signals,
            # then scatter back to the full-length stream (NaN where a signal was missing).
            present = [i for i, rec in enumerate(recs) if all(rec.has(s) for s in d.requires)]
            vals = [float("nan")] * len(recs)
            if present:
                try:
                    seq = d.score_stream([recs[i] for i in present])
                    for pos, v in zip(present, seq):
                        vals[pos] = float(v)
                except Exception:
                    pass  # probe not fitted / unavailable -> all NaN (dropped below)
            streams[d.name] = vals
            continue
        # memoryless per-chunk detectors (accel/ace/stac/oracle/embedding family)
        col: list[float] = []
        for rec in recs:
            if not all(rec.has(s) for s in d.requires):
                col.append(float("nan"))  # a required signal is unavailable this chunk
                continue
            try:
                col.append(float(d.score(rec)))
            except Exception:
                # e.g. fm_loss without an injected velocity, or a learned detector not fitted —
                # treat as unavailable for this run (it gets dropped below), don't crash the loop.
                col.append(float("nan"))
        streams[d.name] = col

    out: dict[str, list[float]] = {}
    for name, vals in streams.items():
        arr = np.asarray(vals, float)
        finite = np.flatnonzero(np.isfinite(arr))
        if finite.size == 0:
            continue  # detector never available for this rollout (e.g. ace with no resample)
        if finite[0] > 0:  # forward-fill leading NaN (temporal detector undefined at chunk 0)
            arr[: finite[0]] = arr[finite[0]]
        if not np.isfinite(arr).all():
            continue  # interior gap (intermittent availability) — skip for safety
        out[name] = [round(float(v), 6) for v in arr]
    return out


def score_rollout(
    x_t: np.ndarray,
    chunks_div: np.ndarray | None,
    n_valid: int,
    *,
    action_dim: int,
    n_exec: int,
    dets: Sequence[Any],
    dim_scale: np.ndarray | None = None,
    obs_emb: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Back-compat shim: build ChunkRecords then score them (the pre-refactor one-call API)."""
    recs = build_chunk_records(x_t, chunks_div, n_valid, action_dim=action_dim, n_exec=n_exec,
                               dim_scale=dim_scale, obs_emb=obs_emb, hidden=hidden)
    return score_records(recs, dets)


def run_detector_score(
    run: Any,
    *,
    detectors: Sequence[str],
    n_exec: int | None = None,
    rollouts: Sequence[int] | None = None,
    max_rollouts: int | None = None,
    success_by_rollout: Mapping[Any, bool] | None = None,
    exclude_rollout_units: Sequence[Any] | None = None,
    out_dir: str | Path | None = None,
    label: str | None = None,
    run_harness: bool = True,
    fit_n_unsup: int | None = None,
    fit_n_sup: int | None = None,
    fit_n_sup_succ: int | None = None,
    fit_n_sup_fail: int | None = None,
    fit_frac: float = 0.5,
    fit_seed: int = 0,
    resample_k: int | None = 4,
    resample_seed: int = 0,
    cusum_calib_n: int = 10,
    cusum_calib_seed: int | None = None,   # None -> follow fit_seed (varies per repeat)
    cusum_calib_draws: int = 20,
    device: str = "cpu",
    accel_fixed_std: str | Path | np.ndarray | bool | None = True,
    accel_mode: str | None = None,
) -> tuple[runs.RunDir, dict]:
    """Score ``detectors`` over a recorded run and (optionally) run the shared FD harness.

    Reads ``<run>/fm/`` for ``x_t``, ``<run>/chunk_divergence/`` for the resample candidates, and
    ``<run>/obs_emb/`` for the pooled prefix embedding (when present). Detectors that need a
    ``fit`` are **calibrated on a disjoint split** that is EXCLUDED from the scored/eval set, so the
    comparison never trains and tests on the same episode. Two disjoint fit families:

    * **UNSUPERVISED** embedding-OOD (rnd_oe/logpzo/density/fiper) — held out from the SUCCESS-only
      rollouts carrying ``obs_emb``;
    * **SUPERVISED** SAFE — held out from the rollouts carrying ``hidden`` (spans BOTH classes).

    Resample-based detectors use ``resample_k=8`` candidates by default, selected without model
    re-execution from the complete saved candidate axis. ``resample_seed`` selects a reproducible
    subset independently for each rollout unit; use ``resample_k=None`` to consume every stored
    candidate (the pre-ablation legacy behaviour).

    Split size is controlled by **absolute rollout counts**. The UNSUPERVISED family uses
    ``fit_n_unsup`` success rollouts. For SUPERVISED SAFE, ``fit_n_sup_succ`` / ``fit_n_sup_fail``
    give an EXACT per-class held-out count (the paper protocol draws a BALANCED split — e.g. 4
    success + 4 failure — so the probe never sees a class-imbalanced calibration set); when both
    are ``None`` the legacy single-count ``fit_n_sup`` path is used (drawn from the pooled
    labeled set, then class-balanced only if it happened to be single-class). When a count is
    ``None`` it falls back to the legacy ``round(fit_frac · pool)`` fraction (default
    ``fit_frac=0.5``). Per-class counts are clamped to the available class size. ``fit_seed`` seeds
    the held-out sample (SAFE uses ``fit_seed+1`` so the two families draw independently). The two
    ``fit_ids`` sets are UNIONED and all removed from scoring. Training-free detectors
    (accel/straightness/sparc/ace/stac/oracle) see the full scored set. Writes
    ``<out>/<task_group>/<Task>.json`` (FD format) + the harness ``meta.json`` /
    ``compare_detectors.png``. Returns ``(RunDir, harness_summary | {})``.
    """
    rd = run if isinstance(run, runs.RunDir) else runs.resolve_run(run)
    recording = FMRecording.load(rd.fm_dir)
    if not recording.rollouts:
        raise ValueError(f"run {rd.run_id!r} has no recorded rollouts under {rd.fm_dir}")
    dims = recording.dims
    action_dim = int(dims["action_dim"])
    # Clamp to the real output dim: some adapters set dims.action_dim = max_action_dim (the model's
    # padded head width) while the stored chunk lives at fewer dims; scoring accel/straightness over
    # the zero-pad dims would dilute the signal. The recorded chunk_actions last axis is the ground
    # truth (a no-op where dims.action_dim already agrees with it).
    real_adim = int(np.asarray(recording.rollouts[0].chunk_actions).shape[-1])
    if real_adim != action_dim:
        logger.info("detector_score: dims.action_dim=%d but recorded chunk_actions has %d dims; "
                    "using %d (the real output dim).", action_dim, real_adim, real_adim)
        action_dim = real_adim
    n_action_steps = int(dims["n_action_steps"])
    chunk_size = int(dims["chunk_size"])
    if n_exec is None:
        n_exec = min(n_action_steps, chunk_size)
    cd_dir = rd.root / "chunk_divergence"
    oe_dir = rd.root / "obs_emb"
    hs_dir = rd.root / "hidden_states"
    fl_dir = rd.root / "fm_loss"

    # fm_loss is model-in-the-loop: it can't score on this lerobot-free path, so it's served as a
    # PRECOMPUTED stream from its own GPU stage (cli/fm_loss.py) rather than a live detector.
    precomputed = [n for n in detectors if n == "fm_loss"]
    # The learned detectors (rnd_oe / logpzo / safe, and fiper via its inner RND **rnd_kwargs)
    # train a torch net in fit(); pass the run device so they train on GPU (250-epoch CPU-Adam
    # over ~28k embeddings is hours). Others (ace/oracle/stac/density) are numpy/sklearn — no device.
    _DEVICE_AWARE = {"rnd_oe", "logpzo", "safe", "fiper"}

    # accel's default mode is accel_prefix:7 — a DENOISE-depth prefix read at layer 7, tuned for the
    # T=10 recording path (9 prefix layers). A shallow-denoise model (e.g. num_inference_steps=4
    # → only T-1=3 prefix layers) can't index depth 7, so score_chunks raises IndexError for EVERY chunk
    # and accel gets silently dropped. Auto-fall-back to accel_prefix:-1 (the LAST denoise layer ==
    # accel_exec, the cross-model-comparable readout; on pi05 corr(j7, j-1)≈0.90) whenever the recording
    # is too shallow for the default depth. A T=10 recording keeps the validated accel_prefix:7
    # byte-for-byte.
    n_inf = int(dims.get("num_inference_steps", n_action_steps))
    n_prefix_layers = max(1, n_inf - 1)  # whole_chunk_prefix_accel returns (k, T-1)

    def _accel_mode() -> str:
        from fmaccel.detectors.accel import AccelDetector
        # An explicit caller override (e.g. a shallow denoise-depth prefix `accel_prefix:1`) takes
        # precedence over the validated default; still subject to the shallow-recording fall-back below.
        default = accel_mode if accel_mode is not None else AccelDetector().mode  # "accel_prefix:7"
        base, sep, col = default.partition(":")
        if base == "accel_prefix" and sep:
            j = int(col)
            if not (-n_prefix_layers <= j < n_prefix_layers):
                logger.info("detector_score: accel default mode %r needs denoise depth %d but this "
                            "recording has only %d prefix layer(s) (num_inference_steps=%d); falling "
                            "back to 'accel_prefix:-1' (last layer == accel_exec, cross-model comparable).",
                            default, j, n_prefix_layers, n_inf)
                return "accel_prefix:-1"
        return default

    def _accel_std(mode: str) -> "np.ndarray | None":
        """The per-dim reference std for accel, matching score_chunks' internal window for ``mode``.

        ⚠️ SCALE (load-bearing, see detectors/accel.py + the accel_normalization_scale note): with
        fixed_std=None, AccelDetector.score self-normalizes EACH chunk (it feeds one chunk as a
        (1,T+1,…) batch), which does NOT match the offline per-episode/run-pooled label scale the
        proxy was validated on (corr≈0.30) and measurably depresses AUROC (pi05 LIBERO: 0.71 self
        vs 0.79 run-pooled). We restore the validated behaviour by z-scoring against a FIXED
        reference std — a caller may pass an external demo-std npz;
        here we default to the run-pooled std computed over ALL of this recording's chunks (no
        external artifact needed; equivalent scale). ``accel_fixed_std``: True → auto run-pooled
        (default); a path/ndarray → that external reference; False/None → self-norm (legacy).

        The std MUST match score_chunks' flattened dim KA = window*action_dim, where the window is the
        executed window for accel_prefix/accel_exec and the whole chunk for accel/accel_prefix_full."""
        if accel_fixed_std is None or accel_fixed_std is False:
            return None
        if isinstance(accel_fixed_std, np.ndarray):
            return np.asarray(accel_fixed_std, np.float32)
        if not isinstance(accel_fixed_std, bool):  # a path to an external reference-std npz
            with np.load(Path(accel_fixed_std)) as z:
                key = "std" if "std" in z else list(z.keys())[0]
                return np.asarray(z[key], np.float32)
        # accel_fixed_std is True -> compute the run-pooled per-dim std over every chunk of this run,
        # sliced to the window the mode uses, so KA matches score_chunks exactly.
        base = mode.partition(":")[0]
        whole = base in ("accel", "accel_prefix_full")  # whole-chunk window vs executed window
        w = int(chunk_size if whole else n_exec)
        paths: list[np.ndarray] = []
        for ro in recording.rollouts:
            for env_idx in range(ro.batch_size):
                nv = _valid_n_chunks(ro, env_idx)
                if nv <= 0:
                    continue
                # (nv, T+1, window, action_dim) -> flatten chunk axis to (nv, T+1, window*action_dim)
                xc = np.asarray(ro.x_t[:nv, :, env_idx, :w, :action_dim], np.float32)
                paths.append(xc.reshape(xc.shape[0], xc.shape[1], -1))
        if not paths:
            return None
        flat = np.concatenate(paths, axis=0)             # (N_total, T+1, KA)
        std = flat.std(axis=(0, 1)).astype(np.float32)   # (KA,)  run-pooled per-dim std
        logger.info("detector_score: accel using RUN-POOLED fixed_std (window=%d, KA=%d, over %d chunks) "
                    "to match the label scale (self-norm depresses AUROC ~0.08).",
                    w, std.shape[0], flat.shape[0])
        return std

    def _make(name: str):
        cls = get_detector(name)
        if name in _DEVICE_AWARE:
            # A repeat changes both the held-out fit rollouts and the learned detector's own RNG.
            return cls(device=device, seed=fit_seed)
        if name == "accel":
            m = _accel_mode()
            return cls(mode=m, fixed_std=_accel_std(m))
        return cls()

    dets = [_make(name) for name in detectors if name not in precomputed]
    # Two fit families: unsupervised (success-only calibration — embedding-OOD + fiper) and supervised
    # (SAFE — needs FAILURE-labeled episodes). Each trains on a DISJOINT held-out split, both excluded
    # from scoring; a detector is memoryless-scored unless it overrides score_stream (SAFE-LSTM).
    needs_fit_unsup = [d for d in dets if (not d.supervised) and ("obs_emb" in d.requires)]
    needs_fit_sup = [d for d in dets if d.supervised]  # SAFE (requires 'hidden' + labels)
    total_units = sum(ro.batch_size for ro in recording.rollouts)
    if rollouts is not None:
        idxs = [int(i) for i in rollouts]
    else:
        idxs = list(range(len(recording.rollouts)))
        if max_rollouts is not None:
            idxs = idxs[: int(max_rollouts)]

    out = Path(out_dir) if out_dir is not None else rd.stage_dir("detectors")

    # Pass 1: build every rollout's ChunkRecords once (carry obs_emb / hidden when the stage ran),
    # tag success, so the fit pass and the scoring pass read the SAME records.
    per_rollout: list[dict] = []
    n_with_div = n_with_emb = n_with_hidden = 0
    for ri in idxs:
        ro = recording.rollouts[ri]
        rid = int(ro.rollout_id)
        for env_idx in range(ro.batch_size):
            nv = _valid_n_chunks(ro, env_idx)
            sfx = _unit_suffix(rid, env_idx, ro.batch_size) if total_units > 1 else ""
            # One read, two windows: `chunks_full` spans the whole chunk horizon (cross-time
            # detectors need the part PAST the executed window), `chunks` is its executed-window
            # slice — bit-identical to the pre-fix load, so ace/oracle/fiper numerics are unchanged.
            chunks_full, _div_step, _fa, have_div, _scale = _load_div_chunks(
                cd_dir, sfx, chunk_size, full_window=True)
            chunks = None
            if have_div:
                subset = _resample_subset_indices(
                    chunks_full.shape[1], resample_k, seed=resample_seed,
                    rollout_id=rid, env_idx=env_idx)
                chunks_full = chunks_full[:, subset, ...]
                chunks = chunks_full[:, :, :_fa, :]
            n_with_div += int(have_div)
            obs_emb = _load_obs_emb(oe_dir, rid, env_idx, ro.batch_size)
            n_with_emb += int(obs_emb is not None)
            hidden = _load_hidden(hs_dir, rid, env_idx, ro.batch_size)
            n_with_hidden += int(hidden is not None)
            x_env = np.asarray(ro.x_t[:nv, :, env_idx, :, :], np.float32)  # (nv, T+1, K, D)
            group = str(ro.task_group or "all")
            task_desc = (str(ro.task_descs[0, env_idx])
                         if getattr(ro, "task_descs", None) is not None and ro.task_descs.size else "")
            recs = build_chunk_records(
                x_env, chunks if have_div else None, nv, action_dim=action_dim, n_exec=int(n_exec),
                chunks_div_full=chunks_full if have_div else None,
                task_id=int(ro.task_id), task_group=group,
                dim_scale=_scale if have_div else None, obs_emb=obs_emb, hidden=hidden)
            unit_id = (rid, env_idx) if ro.batch_size > 1 else rid
            succ = None
            if success_by_rollout:
                succ = success_by_rollout.get(unit_id, success_by_rollout.get(rid))
            pre: dict[str, np.ndarray] = {}
            for name in precomputed:                          # precomputed GPU-stage streams (fm_loss)
                arr = (_load_precomputed(fl_dir, name, name, rid, env_idx, ro.batch_size)
                       if name == "fm_loss" else None)
                if arr is not None:
                    pre[name] = arr[:nv]
            per_rollout.append({
                "rollout_id": unit_id, "n_chunks": int(nv), "recs": recs, "precomputed": pre,
                "success": bool(_episode_success(ro, env_idx) if succ is None else succ),
                "group": group, "task": _sanitize(task_desc or f"task{int(ro.task_id)}"),
                "has_emb": obs_emb is not None, "has_hidden": hidden is not None,
            })

    # Fit pass. Two disjoint held-out splits, both excluded from scoring:
    #  (a) UNSUPERVISED embedding-OOD family — success-only calibration;
    #  (b) SUPERVISED SAFE — needs BOTH classes (success + failure) with the 'hidden' feature.
    def _held_out_n(count: int | None, pool: int, lo: int) -> int:
        """Number of rollouts to hold out for a fit family: an explicit ABSOLUTE ``count`` (clamped to
        [lo, pool] and to pool-1 so >=1 rollout is always left to score), else the legacy
        round(fit_frac·pool) fraction with the same floor/cap."""
        n = int(count) if count is not None else int(round(float(fit_frac) * pool))
        return max(lo, min(n, pool - 1 if pool > lo else pool))

    fit_ids: set[Any] = set()
    fit_unsup_ids: set[Any] = set()
    fit_sup_ids: set[Any] = set()
    if needs_fit_unsup:
        emb_succ = [r for r in per_rollout if r["success"] and r["has_emb"]]
        if not emb_succ:
            logger.warning("detector_score: %d detector(s) need a success-only fit but no SUCCESS "
                           "rollout carries obs_emb (run cli/obs_emb.py first); dropped: %s",
                           len(needs_fit_unsup), [d.name for d in needs_fit_unsup])
        else:
            rng = np.random.default_rng(int(fit_seed))
            order = rng.permutation(len(emb_succ))
            n_fit = _held_out_n(fit_n_unsup, len(emb_succ), lo=1)
            fit_rollouts = [emb_succ[i] for i in order[:n_fit]]
            fit_unsup_ids = {r["rollout_id"] for r in fit_rollouts}
            fit_ids |= fit_unsup_ids
            fit_recs = [rec for r in fit_rollouts for rec in r["recs"] if rec.obs_emb is not None]
            for d in needs_fit_unsup:
                d.fit(fit_recs)
            logger.info("detector_score: fit %d embedding-OOD detector(s) on %d/%d success rollouts "
                        "(%d chunks, %s), held out from scoring: %s",
                        len(needs_fit_unsup), len(fit_rollouts), len(emb_succ), len(fit_recs),
                        f"fit_n_unsup={fit_n_unsup}" if fit_n_unsup is not None else f"fit_frac={fit_frac}",
                        [d.name for d in needs_fit_unsup])
    if needs_fit_sup:
        hid_eps = [r for r in per_rollout if r["has_hidden"]]
        succ_eps = [r for r in hid_eps if r["success"]]
        fail_eps = [r for r in hid_eps if not r["success"]]
        if not fail_eps or not succ_eps:
            logger.warning("detector_score: SAFE is supervised and needs BOTH success (%d) and "
                           "failure (%d) rollouts carrying the 'hidden' feature (run the hidden "
                           "capture stage first); dropped: %s", len(succ_eps), len(fail_eps),
                           [d.name for d in needs_fit_sup])
        elif fit_n_sup_succ is not None or fit_n_sup_fail is not None:
            # BALANCED per-class draw (the paper protocol: exactly N_succ + N_fail rollouts).
            # A per-class count of None defaults to the other's value, then to 4 (paper default).
            n_s_req = fit_n_sup_succ if fit_n_sup_succ is not None else (fit_n_sup_fail if fit_n_sup_fail is not None else 4)
            n_f_req = fit_n_sup_fail if fit_n_sup_fail is not None else (fit_n_sup_succ if fit_n_sup_succ is not None else 4)
            n_s = max(1, min(int(n_s_req), len(succ_eps)))
            n_f = max(1, min(int(n_f_req), len(fail_eps)))
            rng = np.random.default_rng(int(fit_seed) + 1)
            s_pick = [succ_eps[i] for i in rng.permutation(len(succ_eps))[:n_s]]
            f_pick = [fail_eps[i] for i in rng.permutation(len(fail_eps))[:n_f]]
            sup_fit = s_pick + f_pick
            fit_sup_ids = {r["rollout_id"] for r in sup_fit}
            fit_ids |= fit_sup_ids
            labeled = [([rec for rec in r["recs"] if rec.has("hidden")], r["success"]) for r in sup_fit]
            for d in needs_fit_sup:
                d.fit(labeled)
            logger.info("detector_score: fit %d SUPERVISED probe(s) on a BALANCED %d succ + %d fail "
                        "split (requested %s/%s) of %d succ / %d fail labeled rollouts, held out from "
                        "scoring: %s", len(needs_fit_sup), n_s, n_f, n_s_req, n_f_req,
                        len(succ_eps), len(fail_eps), [d.name for d in needs_fit_sup])
        else:
            rng = np.random.default_rng(int(fit_seed) + 1)
            order = rng.permutation(len(hid_eps))
            n_fit = _held_out_n(fit_n_sup, len(hid_eps), lo=2)
            sup_fit = [hid_eps[i] for i in order[:n_fit]]
            # ensure the fit split spans both classes (SAFE needs positives + negatives); dedup by id
            if all(r["success"] for r in sup_fit) or all(not r["success"] for r in sup_fit):
                first_fail = next((r for r in hid_eps if not r["success"]), None)
                first_succ = next((r for r in hid_eps if r["success"]), None)
                seen_ids = {r["rollout_id"] for r in sup_fit}
                sup_fit = [r for r in (first_succ, first_fail)
                           if r is not None and r["rollout_id"] not in seen_ids] + sup_fit
            fit_sup_ids = {r["rollout_id"] for r in sup_fit}
            fit_ids |= fit_sup_ids
            labeled = [([rec for rec in r["recs"] if rec.has("hidden")], r["success"]) for r in sup_fit]
            for d in needs_fit_sup:
                d.fit(labeled)
            logger.info("detector_score: fit %d SUPERVISED probe(s) on %d/%d labeled rollouts "
                        "(%d succ / %d fail, %s), held out from scoring: %s",
                        len(needs_fit_sup), len(sup_fit), len(hid_eps), sum(s for _, s in labeled),
                        sum(1 for _, s in labeled if not s),
                        f"fit_n_sup={fit_n_sup}" if fit_n_sup is not None else f"fit_frac={fit_frac}",
                        [d.name for d in needs_fit_sup])

    # Optional externally-defined exclusions make controlled sweeps pair their evaluation
    # population with an earlier experiment.  Unit ids are ints for B=1 recordings and
    # ``(rollout_id, env_idx)`` tuples for vectorized recordings; JSON round-trips tuples as
    # lists, so normalize both representations here.
    external_exclude_ids: set[Any] = set()
    for unit_id in exclude_rollout_units or ():
        if isinstance(unit_id, (list, tuple)):
            external_exclude_ids.add(tuple(int(v) for v in unit_id))
        else:
            external_exclude_ids.add(int(unit_id))

    # Scoring pass: every rollout NOT in the fit split or the caller's paired-population split.
    per_file: dict[tuple[str, str], list[dict]] = {}
    score_keys: set[str] = set()
    for r in per_rollout:
        if r["rollout_id"] in fit_ids or r["rollout_id"] in external_exclude_ids:
            continue  # held out for calibration — never scored/eval'd
        streams = score_records(r["recs"], dets)
        for name, arr in r["precomputed"].items():           # merge precomputed GPU-stage streams
            finite = np.asarray(arr, float)
            if np.isfinite(finite).any():
                # forward-fill any leading nan to match score_records' contract
                idx0 = np.flatnonzero(np.isfinite(finite))
                if idx0.size and idx0[0] > 0:
                    finite[: idx0[0]] = finite[idx0[0]]
                if np.isfinite(finite).all():
                    streams[name] = [round(float(v), 6) for v in finite]
        output_id = (f"{r['rollout_id'][0]}_e{r['rollout_id'][1]}"
                     if isinstance(r["rollout_id"], tuple) else r["rollout_id"])
        ep: dict[str, Any] = {"success": r["success"], "n_chunks": r["n_chunks"],
                              "rollout_id": output_id}
        for name, vals in streams.items():
            ep[f"{name}_scores"] = vals
            ep[f"mean_{name}"] = round(float(np.mean(vals)), 6)
            score_keys.add(f"{name}_scores")
        per_file.setdefault((r["group"], r["task"]), []).append(ep)

    groups = sorted({g for g, _ in per_file})
    for (group, task), eps in per_file.items():
        (out / group).mkdir(parents=True, exist_ok=True)
        write_json(out / group / f"{task}.json", {"tasks": {task: {"episodes": eps}}})
    logger.info("detector_score: %d episodes from %d rollout file(s) (%d with resample, %d with obs_emb, %d with hidden; "
                "%d held out for fit) -> %d (group,task) files under %s; score-keys=%s",
                len(per_rollout), len(idxs), n_with_div, n_with_emb, n_with_hidden, len(fit_ids), len(per_file), out,
                sorted(score_keys))
    produced = {k[: -len("_scores")] for k in score_keys}
    dropped = [d for d in dets if d.name not in produced]
    if dropped:  # a requested detector never yielded a finite score on any rollout -> name it (not silent)
        logger.warning(
            "detector_score: %d requested detector(s) produced NO usable scores and were dropped: %s. "
            "Likely a required signal was never populated on this post-hoc path (obs_emb needs the "
            "obs-embedding hook; 'hidden' needs the hidden-states capture stage + labeled fit (SAFE); "
            "fm_loss needs an injected velocity; resample/prev_resample detectors need chunk_divergence/). "
            "Required signals: %s",
            len(dropped), [d.name for d in dropped],
            {d.name: sorted(d.requires) for d in dropped})

    summary: dict = {}
    if run_harness and score_keys:
        calib_seed = int(fit_seed if cusum_calib_seed is None else cusum_calib_seed)
        summary = run_failure_detection_eval(
            [out], score_keys=sorted(score_keys), splits=groups, out_dir=out,
            cusum_calib_n=int(cusum_calib_n), cusum_calib_seed=calib_seed,
            cusum_calib_draws=int(cusum_calib_draws),
            label=label or f"{rd.run_id}: detectors {sorted(score_keys)}")
        def _json_id(v: Any) -> Any:
            return [int(v[0]), int(v[1])] if isinstance(v, tuple) else int(v)

        summary["detector_score_protocol"] = {
            "fit_seed": int(fit_seed),
            "fit_n_unsup_requested": None if fit_n_unsup is None else int(fit_n_unsup),
            "fit_n_sup_succ_requested": (
                None if fit_n_sup_succ is None else int(fit_n_sup_succ)),
            "fit_n_sup_fail_requested": (
                None if fit_n_sup_fail is None else int(fit_n_sup_fail)),
            "fit_unsup_rollout_ids": [_json_id(v) for v in sorted(fit_unsup_ids, key=str)],
            "fit_sup_rollout_ids": [_json_id(v) for v in sorted(fit_sup_ids, key=str)],
            "held_out_rollout_ids": [_json_id(v) for v in sorted(fit_ids, key=str)],
            "external_excluded_rollout_ids": [
                _json_id(v) for v in sorted(external_exclude_ids, key=str)
            ],
            "resample_k": None if resample_k is None else int(resample_k),
            "resample_seed": int(resample_seed),
            "resample_source": "deterministic subset of saved chunk_divergence candidates",
            "cusum_calib_n": int(cusum_calib_n),
            "cusum_calib_seed": calib_seed,
            "cusum_calib_draws": int(cusum_calib_draws),
        }
        # run_failure_detection_eval writes meta.json before caller-side protocol fields exist.
        # Rewrite it so every repeat is independently auditable from disk.
        write_json(out / "meta.json", summary)
    return rd, summary
