"""Classical embedding-OOD density baselines: ``pca_kmeans`` / ``knn`` / ``mahalanobis``.

The cheap, (near-)training-free density family on the observation embedding. Each fits on SUCCESS-only
embeddings and scores an embedding's atypicality — **higher = more OOD = more likely failure** (native).
Pure numpy (no sklearn dependency). They share the obs-embedding hand-off with ``rnd_oe``/``logpzo``
(``rec.obs_emb``; same vector at fit/score), so once that hook captures the embedding all of them come
for free. A documented weakness (shared by the whole embedding family): they conflate OOD-but-successful
states with failure — which is exactly the contrast accel is meant to expose, so they are worth
reporting.

Fidelity / attribution (honest):
* ``pca_kmeans`` is the **only** density detector actually in the official FAIL-Detect repo
  (``CXU-TRI/FAIL-Detect`` ``UQ_baselines/PCA_kmeans``): ``PCA(n_components=emb_dim)`` (a pure rotation,
  **no dimensionality reduction**) → ``KMeans(n_clusters=64)`` on the **RAW** embedding, score = Euclidean
  distance to the nearest centroid. Reproduced here on the raw embedding (``n_components=None`` ⇒ full rank).
* ``knn`` reproduces Sun et al. 2022 (KNN-OOD): **L2-unit-normalize** the feature, then the k-th nearest
  in-distribution Euclidean distance (≡ cosine on the unit sphere). NOT in the FAIL-Detect benchmark code.
* ``mahalanobis`` reproduces Lee et al. 2018: distance to the success Gaussian (shrinkage covariance,
  pinv). Affine-invariant, so computed on the raw embedding. NOT in the FAIL-Detect benchmark code.

Each subclass applies its OWN faithful normalization (none / L2-unit / cov-whitening); there is no shared
z-score, because the reference methods differ.
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector, stack_signal


class _EmbeddingDetector(Detector):
    """Shared obs-embedding stacking for the embedding-density family (raw; per-subclass normalization).

    Scoring is *batched per episode* via :meth:`score_stream`: the shared harness hands one episode's
    ordered records at a time, we stack their raw embeddings into ``(Q, d)`` and call the subclass's
    vectorized :meth:`_score_matrix`. This replaces the per-chunk Python broadcast (which materialized a
    ``(bank, d)`` temporary *per query* — the KNN CPU tail) with a single BLAS GEMM. Each subclass's
    ``_score_matrix`` uses the algebraic identity ``‖a−b‖² = ‖a‖²+‖b‖²−2a·b`` (or a plain batched matmul
    for Mahalanobis), which is a monotone transform of the original per-row form → identical rank
    ordering, hence identical AUROC (values match to float rounding). ``score`` is retained unchanged for
    the single-record / online path."""

    requires = frozenset({"obs_emb"})
    online = True

    def _stack(self, success_rollouts) -> np.ndarray:
        return stack_signal(success_rollouts, "obs_emb").astype(np.float32)  # (N, d) raw

    def _vec(self, rec: ChunkRecord) -> np.ndarray:
        self.check(rec)
        return np.asarray(rec.obs_emb, np.float32).reshape(-1)  # (d,) raw

    def _stack_recs(self, recs) -> np.ndarray:
        """Stack one episode's records' raw embeddings into ``(Q, d)`` (harness already filtered to
        records that carry ``obs_emb``, so no per-record ``check`` is needed here)."""
        return np.stack([np.asarray(r.obs_emb, np.float32).reshape(-1) for r in recs]).astype(np.float32)

    def _score_matrix(self, Z: np.ndarray) -> np.ndarray:
        """Vectorized score of a raw-embedding batch ``Z`` ``(Q, d)`` → ``(Q,)``. Default fallback =
        per-row :meth:`score` via a lightweight ChunkRecord shim (subclasses override with a batched path)."""
        return np.asarray([self.score(ChunkRecord(x_t=np.zeros((1, 1, 1), np.float32),
                                                   action_dim=int(z.shape[-1]), obs_emb=z)) for z in Z],
                          np.float32)

    def score_stream(self, recs):  # type: ignore[override]
        """Batched per-episode scoring (overrides the per-record default so the family runs as one
        GEMM per episode instead of a Python loop over ~14k-row broadcasts)."""
        if not recs:
            return []
        return [float(v) for v in self._score_matrix(self._stack_recs(recs))]


class PcaKmeansDetector(_EmbeddingDetector):
    """PCA → k-means on the RAW embedding; score = distance to nearest centroid (FAIL-Detect)."""

    name = "pca_kmeans"

    def __init__(self, *, n_components: int | None = None, n_clusters: int = 64, iters: int = 25,
                 seed: int = 0) -> None:
        self.n_components = n_components  # None = full-dim PCA (pure rotation), matching the repo
        self.n_clusters = int(n_clusters)
        self.iters = int(iters)
        self.seed = int(seed)

    def fit(self, success_rollouts) -> None:
        X = self._stack(success_rollouts)  # raw, no z-score (repo fits PCA/KMeans on the raw embedding)
        self._pca_mean = X.mean(0)
        Xc = X - self._pca_mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        nc = Vt.shape[0] if self.n_components is None else min(self.n_components, Vt.shape[0])
        self._components = Vt[:nc]                                  # (nc, d)
        P = Xc @ self._components.T                                 # (N, nc)
        # k-means (Lloyd) in PCA space
        rng = np.random.default_rng(self.seed)
        k = min(self.n_clusters, P.shape[0])
        cent = P[rng.choice(P.shape[0], k, replace=False)].copy()
        for _ in range(self.iters):
            d2 = ((P[:, None, :] - cent[None]) ** 2).sum(-1)        # (N, k)
            assign = d2.argmin(1)
            new = np.stack([P[assign == j].mean(0) if np.any(assign == j) else cent[j]
                            for j in range(k)])
            if np.allclose(new, cent):
                cent = new; break
            cent = new
        self._centroids = cent                                     # (k, nc)

    def score(self, rec: ChunkRecord) -> float:
        p = (self._vec(rec) - self._pca_mean) @ self._components.T
        return float(np.sqrt(((p[None] - self._centroids) ** 2).sum(-1).min()))

    def _score_matrix(self, Z: np.ndarray) -> np.ndarray:
        # project the batch into PCA space, then min distance to the centroids via one GEMM
        P = (np.asarray(Z, np.float32) - self._pca_mean) @ self._components.T   # (Q, nc)
        C = self._centroids                                                     # (k, nc)
        D2 = np.maximum((P * P).sum(1)[:, None] + (C * C).sum(1)[None, :]
                        - 2.0 * (P @ C.T), 0.0)                                 # (Q, k)  ‖p-c‖²
        return np.sqrt(D2.min(1)).astype(np.float32)


class KnnDetector(_EmbeddingDetector):
    """Distance to the k-th nearest in-distribution embedding on the unit sphere (Sun et al. 2022)."""

    name = "knn"

    def __init__(self, *, k: int = 5, max_bank: int = 20000, seed: int = 0) -> None:
        self.k = int(k)
        self.max_bank = int(max_bank)
        self.seed = int(seed)

    @staticmethod
    def _l2norm(X: np.ndarray) -> np.ndarray:
        return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)  # KNN-OOD: project to unit sphere

    def fit(self, success_rollouts) -> None:
        X = self._l2norm(self._stack(success_rollouts))
        if X.shape[0] > self.max_bank:                             # subsample the bank for speed
            rng = np.random.default_rng(self.seed)
            X = X[rng.choice(X.shape[0], self.max_bank, replace=False)]
        self._bank = X                                            # (B, d) unit-norm

    def score(self, rec: ChunkRecord) -> float:
        z = self._l2norm(self._vec(rec)[None])[0]
        d = np.sqrt(((self._bank - z[None]) ** 2).sum(-1))         # (B,)
        kth = min(self.k, d.shape[0]) - 1
        return float(np.partition(d, kth)[kth])

    def _score_matrix(self, Z: np.ndarray) -> np.ndarray:
        # unit-normalize the query batch, then k-th nearest bank distance via one (Q, B) GEMM.
        Zn = self._l2norm(np.asarray(Z, np.float32))                # (Q, d) unit-norm
        B = self._bank                                              # (B, d) unit-norm
        # ‖z-b‖² = 2 - 2 z·b on the unit sphere; keep the identity general (bank norms may drift ~1)
        D2 = np.maximum((Zn * Zn).sum(1)[:, None] + (B * B).sum(1)[None, :]
                        - 2.0 * (Zn @ B.T), 0.0)                    # (Q, B)
        kth = min(self.k, B.shape[0]) - 1
        # k-th smallest per row (partition on the squared distance = same order), then sqrt
        part = np.partition(D2, kth, axis=1)[:, kth]                # (Q,)
        return np.sqrt(part).astype(np.float32)


class MahalanobisDetector(_EmbeddingDetector):
    """Mahalanobis distance to the success-embedding Gaussian (Lee et al. 2018); affine-invariant."""

    name = "mahalanobis"

    def __init__(self, *, shrinkage: float = 1e-2) -> None:
        self.shrinkage = float(shrinkage)

    def fit(self, success_rollouts) -> None:
        X = self._stack(success_rollouts)  # raw — Mahalanobis is affine-invariant, no standardization needed
        self._mean = X.mean(0)
        cov = np.cov(X, rowvar=False)
        cov = (1 - self.shrinkage) * cov + self.shrinkage * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
        self._prec = np.linalg.pinv(cov)                           # (d, d)

    def score(self, rec: ChunkRecord) -> float:
        delta = self._vec(rec) - self._mean
        return float(np.sqrt(max(delta @ self._prec @ delta, 0.0)))

    def _score_matrix(self, Z: np.ndarray) -> np.ndarray:
        # batched quadratic form: sqrt(diag(Δ P Δᵀ)) = sqrt(sum((Δ @ P) * Δ, axis=1)), two GEMMs
        D = np.asarray(Z, np.float32) - self._mean                 # (Q, d)
        q = np.einsum("qi,qi->q", D @ self._prec, D)               # (Q,)  δ P δ per row
        return np.sqrt(np.maximum(q, 0.0)).astype(np.float32)
