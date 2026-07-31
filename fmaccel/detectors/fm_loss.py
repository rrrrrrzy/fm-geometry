"""``fm_loss`` — Diff-DAgger's denoising loss, adapted to flow matching.

Diff-DAgger (Lee et al., ICRA 2025, arXiv:2410.14868) scores a chunk by how well the policy can
denoise its OWN generated action across noise levels: at an in-distribution state the model
reconstructs its action at every noise scale (low loss); at an OOD state the generated action lies
where the learned score field is inconsistent, so the loss spikes — a model-internal novelty signal
that (unlike sample variance) does not mistake legitimate multimodality for uncertainty.

Adapted to flow matching with pi0.5's CLEAN-target interpolant: for the chunk ``a_hat``, draw
``(t~U[0,1], x0~N(0,I))``, form ``x_t = (1-t)·x0 + t·a_hat``, evaluate the policy velocity field
``v_θ(x_t, t, o)`` and average ``‖v_θ - (a_hat - x0)‖²`` over many ``(t, x0)``. Higher = worse
self-reconstruction = **more likely failure** (native). The Diff-DAgger *decision* (0.99-quantile
demo threshold + K-consecutive window) is just an operating point on this score — left to the
harness's TPR@FPR / online-sim, so the detector returns the raw loss.

⚠️ MODEL-IN-THE-LOOP, NOT free: unlike accel/ace this needs many fresh re-noised velocity-field
forwards per decision (Diff-DAgger uses 16×32=512), so it can't be read off the single recorded
denoise path. The model's per-chunk velocity field is injected via ``rec.extra['fm_velocity']`` — a
callable ``(x_t: (M,chunk,act), t: (M,)) -> v: (M,chunk,act)`` closing over THAT chunk's conditioning
(the GPU ``detector_score`` pipeline builds it from the cached prefix / ChunkResampleSession).

⚠️ VELOCITY CONTRACT (read before wiring ``fm_velocity``): this file uses the **clean-data** flow
``x_t = (1-t)·x0 + t·â`` with ``target = â − x0``, which is the **time- and sign-reverse** of pi0.5's
*native* convention ``x_s = s·noise + (1-s)·action``, ``u = noise − action`` (``modeling_pi05.py``).
So ``fm_velocity`` must return the velocity OF THIS file's interpolant, i.e. **``−v_θ(x_t, 1−t)``**
relative to pi0.5's raw velocity field (negate the value AND evaluate at FM time ``1−t``). The in-file
math is a self-consistent rectified-flow parametrization (the conditional velocity of the straight path
``x0→â`` is exactly ``â − x0 = target``), so a correctly-built callback gives ~0 loss on an
in-distribution chunk — **add that low-loss smoke gate before trusting any number** (a raw
``v_θ(x_t, t)`` without the flip yields a large, magnitude-ranked loss, not an OOD signal). ``a_hat`` =
the recorded chunk endpoint (``x_t[-1]``) — score the SAME action source in calibration and online.
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector


class FmLossDetector(Detector):
    """MC flow-matching re-noise loss of the policy's own chunk (higher = OOD = more likely failure)."""

    name = "fm_loss"
    requires = frozenset({"x_t"})
    online = False

    def __init__(self, *, m_t: int = 16, m_noise: int = 2, n_exec: int | None = None,
                 seed: int = 0) -> None:
        self.m_t = int(m_t)
        self.m_noise = int(m_noise)
        self.n_exec = n_exec
        self.seed = int(seed)

    def score(self, rec: ChunkRecord) -> float:
        self.check(rec)
        fmv = (rec.extra or {}).get("fm_velocity")
        if fmv is None:
            raise ValueError(
                "fm_loss needs rec.extra['fm_velocity'] — a per-chunk velocity callable "
                "(x_t (M,chunk,act), t (M,)) -> v. The GPU detector_score pipeline binds it from the "
                "model (cached prefix / ChunkResampleSession); see docs/baselines.md §3.3."
            )
        a_hat = rec.chunk_actions if rec.chunk_actions is not None else np.asarray(rec.x_t)[-1]
        a_hat = np.asarray(a_hat, np.float32)[..., : rec.action_dim]  # (chunk, act)
        n_exec = rec.n_exec if rec.n_exec is not None else self.n_exec
        if n_exec is not None:
            a_hat = a_hat[: int(n_exec)]
        # fold task/episode into the seed so two chunks at the same env_step (different ep/task) don't
        # draw identical noise; still fully reproducible.
        rng = np.random.default_rng(self.seed + 1000003 * int(rec.task_id or 0) + int(rec.env_step))
        # batch all M=(m_t × m_noise) re-noisings into ONE velocity call (the GPU side batches the forward)
        t_grid = (np.arange(self.m_t, dtype=np.float32) + 0.5) / self.m_t  # (m_t,) midpoints in (0,1)
        ts = np.tile(t_grid, self.m_noise).astype(np.float32)              # (M,)
        M = ts.shape[0]
        x0 = rng.standard_normal((M, *a_hat.shape)).astype(np.float32)     # (M, chunk, act)
        tb = ts[:, None, None]
        x_t = (1.0 - tb) * x0 + tb * a_hat[None]                           # (M, chunk, act)
        target = a_hat[None] - x0                                          # FM clean-target velocity
        v = np.asarray(fmv(x_t, ts), np.float32).reshape(M, *a_hat.shape)
        return float(np.mean((v - target) ** 2))                          # MC E[‖v_θ - u_t‖²]
