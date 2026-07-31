"""``safe`` — SAFE's supervised failure probe on the VLA's last-layer internal feature.

SAFE (Gu et al., NeurIPS 2025, arXiv:2506.09937, "SAFE: Multitask Failure Detection for
Vision-Language-Action Models"; official code ``github.com/vla-safe/SAFE``) trains a *small*
supervised probe on the VLA's internal feature — the intuition being that a VLA already "knows"
about impending failure in its hidden state, generically across tasks. This is the paper's
**category-C** intrusive control (uses FAILURE labels), the contrast that shows accel needs none.

FEATURE (flow-matching / π₀ case = ours, ``failure_prob/data/pizero.py``): the raw per-decision
feature is the action-expert last-layer hidden tensor ``(n_diff_steps, n_pred_horizon, d)``
(π₀.₅: the ``suffix_out`` fed to ``action_out_proj``). SAFE reduces it to one vector
``e_t ∈ R^d`` by aggregating the HORIZON axis first, then the FLOW-STEP axis, via
``process_tensor_idx_rel`` (``failure_prob/data/utils.py``). PizeroDatasetConfig default =
``horizon_idx_rel="mean"``, ``diff_idx_rel="mean"`` (mean over horizon AND flow steps). No PCA.
Our ``hidden`` capture stage stores ``e_t`` already reduced to ``(d,)`` (the reduction is done in
the GPU producer so this lerobot-free detector just consumes ``rec.hidden['action_expert_last']``).

TWO PROBES (``--safe-variant``), matched to the repo:
* **``mlp``** (``failure_prob/model/indep.py`` ``IndepModel``, config default, ``cumsum=True``):
  ``g`` = an MLP (2 layers / hidden 256) → scalar, then ``σ``. Per-step failure signal
  ``p_t = σ(g(e_t)) ∈ (0,1)``; SAFE's score is the RUNNING SUM ``s_t = Σ_{τ≤t} p_τ`` (0 < s_t < t,
  NOT a probability). Training loss (``forward_compute_loss``, non-threshold default): SUCCESS
  episodes minimize ``Σ_t s_t`` (push per-step sigmoids DOWN), FAILURE episodes minimize
  ``Σ_t (−s_t)`` (push UP), with inverse-class-frequency weighting + ``λ_reg`` L2. We train that
  exact objective on ``p_t`` (equivalent: success → ``+Σ p_t``, fail → ``−Σ p_t``).
* **``lstm``** (``failure_prob/model/lstm.py`` ``LstmModel``): ``nn.LSTM(d,256)`` → ``fc`` → ``σ``,
  full-history (``n_history_steps=-1``): ``s_t = σ(LSTM(e_{0:t})) ∈ [0,1]``. Trained with BCE against
  the trajectory label broadcast to every step (FAILURE = positive = ``1 − success``), inverse-class
  weighting + L2. Recurrent, so it scores a whole episode at once via :meth:`score_stream`.

SCORE CONVENTION. Both are natively **higher = more likely failure** (no flip). We return the
**per-step signal** (MLP: ``p_t = σ(g(e_t))``; LSTM: ``s_t = σ(LSTM(e_{0:t}))``) as the per-chunk
score; the shared harness then does the aggregation SAFE stacks on top — SAFE-MLP's cumulative sum
``s_t = Σ p_τ`` is exactly the harness's running statistic, and SAFE's **functional conformal
prediction** band (``failure_prob/utils/conformal/functional_predictor.py``, calibrate μ_t + band on
SUCCESS rollouts, flag ``s_t > δ_t``) is the same success-calibrated threshold layer the harness
applies uniformly (CUSUM / quantile). Keeping the raw per-step signal here makes the cross-detector
AUROC threshold-free, like every other baseline (see the README §"alarm lives in the harness").

SUPERVISION & DEVIATIONS (all flagged, per the audit):
* **Needs FAILURE labels** (``supervised = True``): unlike the success-only embedding-OOD family,
  SAFE's ``fit`` consumes ``(episode_records, success_bool)`` pairs and trains on both classes. The
  label is TRAJECTORY-level, broadcast to every step (SAFE's MIL/weak supervision — the paper does
  not need the exact failure timestep). Using true per-step labels would be stronger supervision → an
  ablation, not the headline.
* **Feature-reduction is config-driven / grid-searched in SAFE** (best of First/Last/Mean/First&Last
  per benchmark on the seen split). We fix the PizeroDatasetConfig default (mean/mean) in the GPU
  ``hidden`` producer; alternatives are a producer-side knob, documented not hidden.
* **Needs the ``hidden`` capture hook** — the action-expert last-hidden-state is not recorded in the
  hot path (same class of GPU hand-off as ``obs_emb``; see ``pipelines/hidden_states.py`` +
  ``docs/baselines.md §4.2``). Until wired, SAFE is synthetic-verified only. ``torch.compile``
  must be OFF on the sampling path or the hidden hook silently misses (compile captures the graph
  before the hook fires).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector

_HIDDEN_KEY = "action_expert_last"  # the reduced (d,) feature the hidden producer stores per chunk


class SafeDetector(Detector):
    """SAFE supervised probe on the action-expert last-hidden feature (higher = more likely failure)."""

    name = "safe"
    requires = frozenset({"hidden"})
    online = True          # one small MLP/LSTM forward at score time (cheap once the feature is captured)
    supervised = True      # fit() needs FAILURE-labeled episodes, not success-only

    def __init__(self, *, variant: str = "mlp", hidden_dim: int = 256, n_layers: int = 2,
                 epochs: int = 1000, lr: float | None = None, batch_size: int = 512,
                 weight_decay: float = 1e-2, lambda_reg: float = 1.0, seed: int = 0,
                 device: str = "cpu", hidden_key: str = _HIDDEN_KEY) -> None:
        if variant not in ("mlp", "lstm"):
            raise ValueError(f"safe variant must be mlp/lstm, got {variant!r}")
        self.variant = variant
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        # repo defaults: MLP lr 1e-3, LSTM lr 3e-4
        self.lr = float(lr) if lr is not None else (3e-4 if variant == "lstm" else 1e-3)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.lambda_reg = float(lambda_reg)
        self.seed = int(seed)
        self.device = device
        self.hidden_key = str(hidden_key)
        self._net = None
        self._d: int | None = None

    # ---- feature extraction ------------------------------------------------
    def _feat(self, rec: ChunkRecord) -> np.ndarray:
        """The reduced per-decision feature ``e_t ∈ R^d`` from ``rec.hidden``.

        The GPU ``hidden`` producer stores it already reduced (mean over horizon then flow steps,
        PizeroDatasetConfig default) under ``hidden_key``; if a raw multi-axis tensor is present we
        reduce it here (mean over all leading axes) so the detector is robust to either producer."""
        h = rec.hidden
        v = h.get(self.hidden_key) if hasattr(h, "get") else h[self.hidden_key]
        v = np.asarray(v, np.float32)
        if v.ndim > 1:                          # raw (…, d) tensor -> mean-reduce leading axes (SAFE mean/mean)
            v = v.reshape(-1, v.shape[-1]).mean(axis=0)
        return v.reshape(-1)

    # ---- models ------------------------------------------------------------
    def _build_mlp(self, d: int):
        import torch.nn as nn

        if self.n_layers <= 1:
            return nn.Linear(d, 1)              # single-layer projector (repo n_layers==1 path)
        layers: list[Any] = [nn.Linear(d, self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(self.hidden_dim, 1)]
        return nn.Sequential(*layers)          # -> (…, 1); sigmoid applied outside (SAFE final_act sigmoid)

    def _build_lstm(self, d: int):
        import torch.nn as nn

        class _Lstm(nn.Module):
            def __init__(self, d_in: int, hid: int):
                super().__init__()
                self.lstm = nn.LSTM(d_in, hid, num_layers=1, batch_first=True)
                self.fc = nn.Linear(hid, 1)

            def forward(self, x):              # x (B, T, d) -> (B, T) logits
                out, _ = self.lstm(x)
                return self.fc(out).squeeze(-1)

        return _Lstm(d, self.hidden_dim)

    # ---- fit (SUPERVISED: labeled episodes) --------------------------------
    def fit(self, labeled_episodes: Any = None) -> None:
        """Train the probe on FAILURE-labeled episodes.

        ``labeled_episodes`` is an iterable of ``(records, success)`` where ``records`` is one
        episode's ordered :class:`ChunkRecord` list and ``success`` is its trajectory outcome
        (True = success). Matches SAFE's supervised objective (success→push per-step signal down,
        failure→up), inverse-class-frequency weighted, ``λ_reg`` L2. Sequences are variable length,
        padded per batch with a validity mask.
        """
        import torch

        if labeled_episodes is None:
            raise ValueError("safe.fit needs FAILURE-labeled episodes ((records, success) pairs); "
                             "it is supervised (category-C), not success-only.")
        # Build per-episode feature sequences + labels (drop episodes with no usable feature).
        seqs: list[np.ndarray] = []
        labels: list[int] = []
        for ep in labeled_episodes:
            records, success = ep
            feats = [self._feat(r) for r in records if r.has("hidden")]
            if not feats:
                continue
            seqs.append(np.stack(feats).astype(np.float32))  # (T, d)
            labels.append(0 if bool(success) else 1)         # FAILURE = positive = 1
        if not seqs:
            raise ValueError("safe.fit: no episode carried the 'hidden' feature (run the hidden "
                             "capture stage first).")
        self._d = int(seqs[0].shape[1])
        y = np.asarray(labels, np.float32)
        # inverse-class-frequency weights (SAFE aggregate_monitor_loss balances success vs failure)
        n_fail = float(max(1.0, y.sum()))
        n_succ = float(max(1.0, len(y) - y.sum()))
        w_fail = 0.5 * len(y) / n_fail
        w_succ = 0.5 * len(y) / n_succ

        torch.manual_seed(self.seed)
        self._net = (self._build_lstm(self._d) if self.variant == "lstm"
                     else self._build_mlp(self._d)).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        n = len(seqs)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g).tolist()
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                loss = self._batch_loss(idx, seqs, y, w_succ, w_fail, bce)
                opt.zero_grad(); loss.backward(); opt.step()

    def _batch_loss(self, idx, seqs, y, w_succ, w_fail, bce):
        import torch

        # pad the variable-length feature sequences of this batch to a common T with a valid-mask
        maxT = max(seqs[j].shape[0] for j in idx)
        B = len(idx)
        X = np.zeros((B, maxT, self._d), np.float32)
        M = np.zeros((B, maxT), np.float32)
        yb = np.zeros(B, np.float32)
        for b, j in enumerate(idx):
            T = seqs[j].shape[0]
            X[b, :T] = seqs[j]
            M[b, :T] = 1.0
            yb[b] = y[j]
        Xt = torch.from_numpy(X).to(self.device)
        Mt = torch.from_numpy(M).to(self.device)
        yt = torch.from_numpy(yb).to(self.device)
        wt = torch.where(yt > 0.5, torch.as_tensor(w_fail, device=self.device),
                         torch.as_tensor(w_succ, device=self.device))

        if self.variant == "lstm":
            logits = self._net(Xt)                        # (B, T)
            per_step = bce(logits, yt[:, None].expand_as(logits))  # BCE(step, traj-label)  (B, T)
            per_ep = (per_step * Mt).sum(1) / Mt.sum(1).clamp_min(1.0)  # masked mean over steps
            data_loss = (per_ep * wt).mean()
        else:  # mlp: SAFE non-threshold objective on per-step sigmoids p_t = σ(g(e_t))
            p = torch.sigmoid(self._net(Xt).squeeze(-1))  # (B, T) in (0,1)
            ssum = (p * Mt).sum(1) / Mt.sum(1).clamp_min(1.0)  # mean per-step signal per episode
            # success (yt=0) -> minimize +ssum (push down); failure (yt=1) -> minimize −ssum (push up)
            sign = torch.where(yt > 0.5, torch.as_tensor(-1.0, device=self.device),
                               torch.as_tensor(1.0, device=self.device))
            data_loss = (sign * ssum * wt).mean()

        # λ_reg L2 on the probe weights (SAFE lambda_reg)
        l2 = sum((prm ** 2).sum() for prm in self._net.parameters())
        return data_loss + self.lambda_reg * 1e-4 * l2

    # ---- scoring -----------------------------------------------------------
    def _require_fit(self) -> None:
        if self._net is None:
            raise RuntimeError("safe.score before fit(); call fit(labeled episodes) first")

    def score(self, rec: ChunkRecord) -> float:
        """Per-decision failure signal (higher = more likely failure).

        MLP: ``p_t = σ(g(e_t))`` (memoryless — the harness cumulatively sums it into SAFE's
        ``s_t = Σ p_τ``). LSTM: a single-step forward, ``σ(LSTM([e_t]))`` — a sliding-window-1
        approximation of the recurrent score; the faithful full-history LSTM score is
        :meth:`score_stream`, which the harness uses for whole episodes."""
        import torch

        self.check(rec)
        self._require_fit()
        e = self._feat(rec)
        with torch.no_grad():
            x = torch.from_numpy(e.astype(np.float32))[None].to(self.device)  # (1, d)
            if self.variant == "lstm":
                logit = self._net(x[:, None, :])[0, 0]    # (1,1,d) -> (1,1) -> scalar logit
            else:
                logit = self._net(x).reshape(-1)[0]
            return float(torch.sigmoid(logit).item())

    def score_stream(self, recs: Sequence[ChunkRecord]) -> list[float]:
        """Per-chunk failure signal over ONE episode's ordered records.

        LSTM: the faithful recurrent score ``s_t = σ(LSTM(e_{0:t}))`` — one pass over the whole
        sequence, so step ``t`` sees the full history (not a 1-step window). MLP: per-step
        ``p_t = σ(g(e_t))`` (independent), left for the harness to cumulatively sum. Records without
        the ``hidden`` feature are skipped (their score is undefined)."""
        import torch

        self._require_fit()
        feats = [self._feat(r) for r in recs if r.has("hidden")]
        if not feats:
            return []
        X = np.stack(feats).astype(np.float32)            # (T, d)
        with torch.no_grad():
            Xt = torch.from_numpy(X)[None].to(self.device)  # (1, T, d)
            if self.variant == "lstm":
                logits = self._net(Xt)[0]                 # (T,) full-history logits
            else:
                logits = self._net(Xt).squeeze(-1)[0]     # (T,) per-step logits
            return [float(v) for v in torch.sigmoid(logits).cpu().numpy()]
