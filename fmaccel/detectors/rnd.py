"""``rnd_oe`` — Random Network Distillation on the observation embedding.

FIPER's OOD leg and a top FAIL-Detect score (Burda et al. 2018; Römer NeurIPS'25; Xu RSS'25). Train
a predictor MLP to match a FROZEN random-init target MLP on the policy's observation embeddings from
SUCCESS-only rollouts; at test the prediction error ``‖predictor(O) - target(O)‖₂`` is small on
in-distribution states and large where the embedding is novel — **higher = OOD = more likely
failure** (native). Observation-side, orthogonal to accel's action-side signal. One MLP forward at
score time (cheap online once the embedding is captured); the cost is the offline predictor fit.

Recipe matched to FIPER's ``rnd_oe`` (``utiasDSL/fiper`` ``rnd_models.py``):
* the **predictor is deliberately deeper/larger than the target** (asymmetry) — so it can fit the
  frozen target on in-distribution data while structurally still differing on OOD; symmetric nets
  underfit and weaken the signal;
* weights are **orthogonal-init (gain √2), biases zero**, and the **target is frozen** at init;
* the **obs embedding is NOT normalized** (FIPER sets ``normalize_tensors.obs_embeddings=False``); the
  raw conditioning vector is fed directly;
* the score is the **L2 norm** ``‖f−g‖₂`` (FIPER default ``rnd_loss='l2'`` = ``PairwiseDistance(p=2)``).

⚠️ Needs ``rec.obs_emb`` — a pooled prefix/observation conditioning vector that the recorder does NOT
capture yet (the obs-embedding hook is the model-specific hand-off; see docs/baselines.md §4.1).
Use the SAME embedding at ``fit`` and ``score``. CPU-trainable (small MLP on small vectors), no GPU.
(NOTE: FAIL-Detect's ``RND`` is a different variant — a ``ConditionalUnet1D`` on the action chunk with
the obs embedding as ``global_cond``; this is FIPER's pure obs-embedding ``RND_OE``.)
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector, stack_signal


class RndOeDetector(Detector):
    """RND prediction error on the obs embedding (higher = OOD = more likely failure)."""

    name = "rnd_oe"
    requires = frozenset({"obs_emb"})
    online = True

    def __init__(self, *, hidden: int = 256, out_dim: int = 128, epochs: int = 250,
                 lr: float = 1e-3, batch_size: int = 256, seed: int = 0, device: str = "cpu") -> None:
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.device = device
        self._predictor = None
        self._target = None

    def _ortho(self, module):
        """FIPER post-process: orthogonal weight init (gain √2), zero bias."""
        import torch.nn as nn

        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2.0 ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return module

    def _target_mlp(self, d_in: int):
        import torch.nn as nn

        return nn.Sequential(  # shallow random target
            nn.Linear(d_in, self.hidden), nn.LeakyReLU(),
            nn.Linear(self.hidden, self.out_dim),
        )

    def _predictor_mlp(self, d_in: int):
        import torch.nn as nn

        return nn.Sequential(  # DEEPER than the target (RND asymmetry, FIPER recipe)
            nn.Linear(d_in, self.hidden), nn.LeakyReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.out_dim),
        )

    def fit(self, success_rollouts) -> None:
        import torch

        X = stack_signal(success_rollouts, "obs_emb")  # (N, d) — raw obs embedding, NOT normalized (FIPER)
        Xd = torch.from_numpy(X.astype(np.float32)).to(self.device)
        torch.manual_seed(self.seed)
        d = Xd.shape[1]
        self._target = self._ortho(self._target_mlp(d)).to(self.device)
        for p in self._target.parameters():
            p.requires_grad_(False)  # frozen random target — NEVER updated
        self._predictor = self._ortho(self._predictor_mlp(d)).to(self.device)
        opt = torch.optim.AdamW(self._predictor.parameters(), lr=self.lr)
        n = Xd.shape[0]
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                xb = Xd[perm[i : i + self.batch_size]]
                loss = ((self._predictor(xb) - self._target(xb)) ** 2).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    def score(self, rec: ChunkRecord) -> float:
        import torch

        self.check(rec)
        if self._predictor is None:
            raise RuntimeError("rnd_oe.score before fit(); call fit(success embeddings) first")
        O = np.asarray(rec.obs_emb, np.float32).reshape(-1)  # raw embedding, no normalization (FIPER)
        with torch.no_grad():
            o = torch.from_numpy(O.astype(np.float32))[None].to(self.device)
            err = ((self._predictor(o) - self._target(o)) ** 2).sum(-1).sqrt()  # L2 norm = FIPER 'l2'
        return float(err.item())  # higher = OOD
