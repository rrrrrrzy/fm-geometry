"""``Straightness`` and per-episode FM-trajectory readouts (numpy only — no torch, no policy).

*Straightness* is the paper's second free geometric score: the **chord-to-arc ratio** of the
denoising path, ``||x_T - x_0|| / sum_t ||dx_t||``. 1.0 means the noise-to-action shot was a
straight line; below 1.0 means it curved or doubled back.

It reads the same bend as ``accel`` and correlates with the resample ground truth, but for plain
CFM it **saturates**: ~97% of chunks land in [0.99, 1.0], so it can barely resolve them. That
saturation is precisely why ``accel`` drops the normalizing denominator
(:mod:`fmaccel.geometry.accel`). Both are reported in the paper — the contrast is the argument.

Operates on :class:`fmaccel.recording.RolloutRecord` and reads the v3 ``dims`` block.
"""

from __future__ import annotations

import numpy as np

from fmaccel.recording.loader import FMRecording, RolloutRecord

EPS = 1e-12


def executed_view(ro: RolloutRecord, n_action_steps: int) -> dict[str, np.ndarray]:
    """Reorganize an FM rollout into per-executed-step arrays.

    Each chunk holds ``chunk_size`` future actions but only the first
    ``n_action_steps`` are executed before re-planning. FM-state arrays
    (noise/x_t/v_t) live in padded ``max_action_dim`` space; ``chunk_actions`` is
    already truncated to ``action_dim``. Returns dict keyed by env step with
    post-done entries NaN-filled (``valid`` marks the live ones)."""
    N, B, T = ro.n_chunks, ro.batch_size, ro.num_inference_steps
    K = n_action_steps
    action_dim = ro.chunk_actions.shape[-1]

    noise = ro.noise[:, :, :K, :].transpose(0, 2, 1, 3).reshape(N * K, B, -1)
    x_t = ro.x_t[:, :, :, :K, :].transpose(0, 3, 1, 2, 4).reshape(N * K, T + 1, B, -1)
    v_t = ro.v_t[:, :, :, :K, :].transpose(0, 3, 1, 2, 4).reshape(N * K, T, B, -1)
    chunk_actions = ro.chunk_actions[:, :, :K, :].transpose(0, 2, 1, 3).reshape(N * K, B, action_dim)

    noise, x_t, v_t = noise[..., :action_dim], x_t[..., :action_dim], v_t[..., :action_dim]

    n_env = ro.n_env_steps
    S = N * K if n_env <= 0 else min(N * K, n_env)
    noise, x_t, v_t, chunk_actions = noise[:S], x_t[:S], v_t[:S], chunk_actions[:S]

    valid = np.ones((S, B), dtype=bool)
    if n_env > 0:
        done = ro.terminated[:S] | ro.truncated[:S]
        for b in range(B):
            idxs = np.flatnonzero(done[:, b])
            if len(idxs):
                valid[idxs[0] + 1:, b] = False

    noise = np.where(valid[:, :, None], noise, np.float32("nan"))
    chunk_actions = np.where(valid[:, :, None], chunk_actions, np.float32("nan"))
    x_t = np.where(valid[:, None, :, None], x_t, np.float32("nan"))
    v_t = np.where(valid[:, None, :, None], v_t, np.float32("nan"))
    return {"noise": noise, "x_t": x_t, "v_t": v_t, "chunk_actions": chunk_actions, "valid": valid}


def extract_episode(ro: RolloutRecord, slot: int, n_action_steps: int) -> dict[str, np.ndarray]:
    """Pull one batch slot out of a rollout, clipped to its first-done env step."""
    ev = executed_view(ro, n_action_steps)
    n_valid = int(ev["valid"][:, slot].sum())
    return {
        "noise": ev["noise"][:n_valid, slot, :],
        "x_t": ev["x_t"][:n_valid, :, slot, :],
        "v_t": ev["v_t"][:n_valid, :, slot, :],
        "chunk_actions": ev["chunk_actions"][:n_valid, slot, :],
    }


def episode_velocity_magnitude(ep: dict[str, np.ndarray],
                               v_mu: np.ndarray | None = None,
                               v_sd: np.ndarray | None = None) -> np.ndarray:
    """Per env step, ‖v_z‖ at each denoise step. Returns (S_ep, T)."""
    v = ep["v_t"]
    v_mu = v.mean(axis=(0, 1)) if v_mu is None else v_mu
    v_sd = v.std(axis=(0, 1)) if v_sd is None else v_sd
    return np.linalg.norm((v - v_mu) / v_sd, axis=-1)


def episode_straightness(ep: dict[str, np.ndarray], x_sd: np.ndarray | None = None) -> np.ndarray:
    """Per env step, z-scored straightness ‖Δx_total‖ / Σ‖Δx_t‖ (1.0 = straight)."""
    x = ep["x_t"]
    x_sd = x.std(axis=(0, 1)) if x_sd is None else x_sd
    x_z = x / x_sd
    diffs = np.diff(x_z, axis=1)
    path = np.linalg.norm(diffs, axis=-1).sum(axis=1)
    chord = np.linalg.norm(x_z[:, -1] - x_z[:, 0], axis=-1)
    return chord / (path + EPS)


def episode_velocity_consistency(ep: dict[str, np.ndarray],
                                 v_mu: np.ndarray | None = None,
                                 v_sd: np.ndarray | None = None) -> np.ndarray:
    """Per env step, cos(v_z_t, v_z_{t+1}) for adjacent denoise steps. (S_ep, T-1)."""
    v = ep["v_t"]
    v_mu = v.mean(axis=(0, 1)) if v_mu is None else v_mu
    v_sd = v.std(axis=(0, 1)) if v_sd is None else v_sd
    v_z = (v - v_mu) / v_sd
    a, b = v_z[:, :-1, :], v_z[:, 1:, :]
    dot = (a * b).sum(axis=-1)
    norm_prod = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return dot / (norm_prod + EPS)


def n_action_steps_of(rec: FMRecording) -> int:
    return int(rec.manifest["dims"]["n_action_steps"])
