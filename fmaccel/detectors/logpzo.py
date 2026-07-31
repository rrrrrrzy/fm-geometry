"""``logpzo`` — flow-matching density of the observation embedding (FAIL-Detect's best score).

FAIL-Detect (Xu et al., RSS 2025, arXiv:2503.08558) finds logpZO the most consistently effective
score: fit a conditional-flow-matching density model to the policy's observation embeddings from
SUCCESS-only rollouts (a velocity net ``v_θ(x, t)`` trained to flow ``O → N(0,I)`` with the CFM
objective), then at test take ONE forward at ``t=0`` and score ``‖O + v_θ(O, 0)‖²`` — a fast
neg-log-density surrogate. Small on in-distribution embeddings, large on novel ones: **higher = OOD
= more likely failure** (native). Observation-side; reuses flow-matching machinery we already study.

⚠️ Same obs-embedding hand-off as ``rnd_oe`` (``rec.obs_emb``; same vector at fit and score). This is
the O_t-only variant (not the A_t-concat one). CPU-trainable, no GPU.

ORIENTATION (load-bearing, matches ``CXU-TRI/FAIL-Detect`` ``UQ_baselines/logpZO/train.py``): the CFM
net is trained to flow **data → noise**, i.e. the embedding ``O`` sits at ``t=0`` (``x0=O``,
``x1~N(0,I)``, ``target = x1 − x0``, ``x_t = (1−t)·x0 + t·x1``). The score then evaluates the velocity
at the SAME end the data lives on (``t=0``): one Euler step ``z = O + v(O,0)`` maps in-distribution ``O``
near ``N(0,I)`` (small ``‖z‖²``) and OOD ``O`` far from it (large). Training the reverse orientation
(noise → data) while scoring at ``t=0`` feeds the net the data point at the *noise* time and silently
undersells the score — so keep ``x0 = data`` here.
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector, stack_signal


class LogpZoDetector(Detector):
    """logpZO flow-density surrogate of the obs embedding (higher = OOD = more likely failure)."""

    name = "logpzo"
    requires = frozenset({"obs_emb"})
    online = True

    def __init__(self, *, hidden: int = 256, epochs: int = 200, lr: float = 1e-3,
                 batch_size: int = 256, seed: int = 0, device: str = "cpu") -> None:
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.device = device
        self._net = None
        self._mu = None
        self._sd = None

    def _build(self, d: int):
        import torch.nn as nn

        # velocity net v_θ(x, t): concat the embedding x and a scalar time → R^d
        return nn.Sequential(
            nn.Linear(d + 1, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, d),
        )

    def fit(self, success_rollouts) -> None:
        import torch

        X = stack_signal(success_rollouts, "obs_emb")  # (N, d)
        self._mu = X.mean(0)
        self._sd = np.maximum(X.std(0), 1e-3)  # floor so a near-constant fit dim can't blow up the z-score
        X1 = torch.from_numpy(((X - self._mu) / self._sd).astype(np.float32)).to(self.device)
        torch.manual_seed(self.seed)
        d = X1.shape[1]
        self._net = self._build(d).to(self.device)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        n = X1.shape[0]
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                data = X1[perm[i : i + self.batch_size]]          # observation embedding O (data, sits at t=0)
                noise = torch.randn_like(data)                    # N(0, I) (sits at t=1)
                t = torch.rand(data.shape[0], 1, device=self.device)
                xt = (1.0 - t) * data + t * noise                 # CFM interpolant, data -> noise (FAIL-Detect)
                target = noise - data                             # velocity that flows data -> noise
                v = self._net(torch.cat([xt, t], dim=-1))
                loss = ((v - target) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    def score(self, rec: ChunkRecord) -> float:
        import torch

        self.check(rec)
        if self._net is None:
            raise RuntimeError("logpzo.score before fit(); call fit(success embeddings) first")
        O = (np.asarray(rec.obs_emb, np.float32).reshape(-1) - self._mu) / self._sd
        with torch.no_grad():
            o = torch.from_numpy(O.astype(np.float32))[None].to(self.device)
            t0 = torch.zeros(1, 1, device=self.device)
            z = o + self._net(torch.cat([o, t0], dim=-1))         # one-step-to-noise surrogate
            score = (z ** 2).sum(-1)
        return float(score.item())  # higher = lower density = OOD
