"""Data generation: observations in R^n, each with a Gaussian-mixture action target.

No flow-channel posteriors here — we never use a ground-truth field. ``GMMTarget``
only *samples* (plus a marginal mean/cov for the learning-quality check). An
``ObsTask`` pairs one observation vector ``o`` with its action target; OOD tasks
carry an observation but no target (the net must extrapolate the o->field map).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from fmaccel.toy.config import ToyConfig


# =========================================================== Gaussian mixture
def _as_cov(spec, d: int, dtype, device) -> Tensor:
    """Coerce scalar / (d,) diagonal / (d,d) into a (d,d) covariance."""
    t = torch.as_tensor(spec, dtype=dtype, device=device)
    if t.ndim == 0:
        return t * torch.eye(d, dtype=dtype, device=device)
    if t.ndim == 1:
        return torch.diag(t)
    return t


@dataclass
class GMMTarget:
    """A Gaussian-mixture action distribution p(x_1 | o). Sampling only."""

    weights: Tensor          # (K,)
    means: Tensor            # (K, d)
    covs: Tensor             # (K, d, d)
    name: str = ""

    @classmethod
    def build(cls, weights, means, covs, *, d: int | None = None, name: str = "",
              dtype=torch.float64) -> "GMMTarget":
        means = torch.as_tensor(means, dtype=dtype)
        if means.ndim == 1:
            means = means[None, :]
        K, dd = means.shape
        d = d or dd
        weights = torch.as_tensor(weights, dtype=dtype).reshape(K)
        weights = weights / weights.sum()
        if not isinstance(covs, Tensor) or covs.ndim != 3:
            if isinstance(covs, (list, tuple)) and len(covs) == K:
                covs = torch.stack([_as_cov(c, d, dtype, "cpu") for c in covs])
            else:
                covs = _as_cov(covs, d, dtype, "cpu").expand(K, d, d).contiguous()
        return cls(weights=weights, means=means, covs=covs.to(dtype), name=name)

    @property
    def d(self) -> int:
        return self.means.shape[-1]

    @property
    def K(self) -> int:
        return self.means.shape[0]

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """Draw n action targets x_1 ~ p(x_1 | o)."""
        idx = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        out = torch.empty(n, self.d, dtype=self.means.dtype)
        for k in range(self.K):
            mask = idx == k
            m = int(mask.sum())
            if m == 0:
                continue
            L = torch.linalg.cholesky(self.covs[k])
            z = torch.randn(m, self.d, dtype=out.dtype, generator=generator)
            out[mask] = self.means[k] + z @ L.T
        return out

    def mean(self) -> Tensor:
        """Marginal mean E[x_1] (fidelity check only)."""
        return (self.weights[:, None] * self.means).sum(0)

    def cov(self) -> Tensor:
        """Marginal covariance Cov(x_1) (fidelity check only)."""
        mbar = self.mean()
        within = torch.einsum("k,kij->ij", self.weights, self.covs)
        dm = self.means - mbar[None]
        between = torch.einsum("k,ki,kj->ij", self.weights, dm, dm)
        return within + between


# ============================================================= observations
@dataclass
class ObsTask:
    """One observation. Training tasks carry a target + a frozen finite training set;
    OOD tasks carry only the observation vector."""

    name: str
    o: Tensor                      # (obs_dim,)
    target: GMMTarget | None = None
    n_modes: int = 0
    is_ood: bool = False
    samples: Tensor | None = None  # (n_i, d) frozen training set (the ONLY data the net sees)


# -------------------------------------------------------------- mode placement
def _sample_separated_means(K: int, d: int, box: float, min_sep: float,
                            g: torch.Generator, *, max_tries: int = 2000) -> Tensor:
    """K means in [-box, box]^d with pairwise distance >= min_sep (rejection sampling)."""
    means = torch.empty(0, d, dtype=torch.float64)
    for _ in range(max_tries):
        if means.shape[0] >= K:
            break
        cand = (torch.rand(d, generator=g, dtype=torch.float64) * 2 - 1) * box
        if means.shape[0] == 0 or torch.cdist(cand[None], means).min() >= min_sep:
            means = torch.cat([means, cand[None]], 0)
    while means.shape[0] < K:                              # rejection gave up; fill loosely
        cand = (torch.rand(d, generator=g, dtype=torch.float64) * 2 - 1) * box
        means = torch.cat([means, cand[None]], 0)
    return means


def _sample_sphere_means(K: int, d: int, r_lo: float, r_hi: float, min_sep: float,
                         g: torch.Generator, *, max_tries: int = 4000) -> Tensor:
    """K means at (radius in [r_lo, r_hi]) * (random unit direction), pairwise dist >= min_sep.

    Keeps ||mu|| ~ O(1) in any dimension (a box sampler would put ||mu|| ~ box*sqrt(d),
    making high-d modes trivially far apart and unreachable from the N(0,I) noise).
    """
    def draw() -> Tensor:
        v = torch.randn(d, generator=g, dtype=torch.float64)
        v = v / v.norm().clamp_min(1e-12)
        r = r_lo + (r_hi - r_lo) * torch.rand((), generator=g, dtype=torch.float64)
        return (r * v)[None]

    means = torch.empty(0, d, dtype=torch.float64)
    for _ in range(max_tries):
        if means.shape[0] >= K:
            break
        cand = draw()
        if means.shape[0] == 0 or torch.cdist(cand, means).min() >= min_sep:
            means = torch.cat([means, cand], 0)
    while means.shape[0] < K:
        means = torch.cat([means, draw()], 0)
    return means


def _lift_circle(thetas: Tensor, obs_dim: int, seed: int) -> Tensor:
    """Map angles to 2-D unit-circle points, then lift into R^obs_dim by fixed
    orthonormal columns (distance- and norm-preserving). The task manifold stays
    1-D, so "held-out angle in a gap" remains well-posed in any obs_dim."""
    obs2d = torch.stack([torch.cos(thetas), torch.sin(thetas)], -1).double()  # (m, 2)
    if obs_dim == 2:
        return obs2d
    if obs_dim < 2:
        raise ValueError("obs_dim must be >= 2")
    g = torch.Generator().manual_seed(seed + 7)
    Q, _ = torch.linalg.qr(torch.randn(obs_dim, 2, generator=g, dtype=torch.float64))
    return obs2d @ Q.T                                                        # (m, obs_dim)


# ----------------------------------------------------------------- builders
def _build_target(i: int, multimodal: bool, cfg: ToyConfig, g: torch.Generator) -> GMMTarget:
    d = cfg.action_dim
    lo, hi = cfg.modes_range
    K = int(torch.randint(lo, hi + 1, (1,), generator=g)) if multimodal else 1
    if cfg.mode_radius is None:
        means = _sample_separated_means(K, d, cfg.mode_box, cfg.mode_sep, g)
    else:
        means = _sample_sphere_means(K, d, cfg.mode_radius[0], cfg.mode_radius[1], cfg.mode_sep, g)
    weights = torch.softmax(cfg.weight_temp * torch.randn(K, generator=g, dtype=torch.float64), 0)
    slo, shi = cfg.sigma_range
    sig = slo + (shi - slo) * torch.rand(K, generator=g, dtype=torch.float64)
    return GMMTarget.build(weights, means, [float(s) ** 2 for s in sig], d=d, name=f"o{i}")


def build_train_tasks(cfg: ToyConfig) -> list[ObsTask]:
    """The k training observations, each with a uni/multimodal GMM action target."""
    g = torch.Generator().manual_seed(cfg.seed)
    if cfg.obs_layout == "circle":
        thetas = torch.linspace(0, 2 * math.pi, cfg.n_obs + 1)[:-1]
        obs = _lift_circle(thetas, cfg.obs_dim, cfg.seed)
    elif cfg.obs_layout == "free":
        if not cfg.train_obs:
            raise ValueError("obs_layout='free' needs cfg.train_obs (a list of observation vectors)")
        obs = torch.as_tensor(cfg.train_obs, dtype=torch.float64)
        if obs.shape[-1] != cfg.obs_dim:
            raise ValueError(f"train_obs vectors must have length obs_dim={cfg.obs_dim}")
    else:
        raise ValueError(f"unknown obs_layout {cfg.obs_layout!r}")

    n = obs.shape[0]
    multi = torch.zeros(n, dtype=torch.bool)
    multi[: min(cfg.n_multimodal, n)] = True
    multi = multi[torch.randperm(n, generator=g)]            # scatter the multimodal ones
    # Separate generator for the frozen datasets, so the target GEOMETRY is independent
    # of how many samples each observation gets.
    gd = torch.Generator().manual_seed(cfg.seed + 1234)
    counts = cfg.samples_list(n)
    tasks = []
    for i in range(n):
        tgt = _build_target(i, bool(multi[i]), cfg, g)
        samples = tgt.sample(counts[i], generator=gd)        # the ONLY data for this obs
        tasks.append(ObsTask(name=f"o{i}", o=obs[i], target=tgt, n_modes=tgt.K, samples=samples))
    return tasks


def build_ood_tasks(cfg: ToyConfig) -> list[ObsTask]:
    """OOD observations to probe (no target — these are held-out conditions)."""
    if cfg.ood_obs:
        obs = torch.as_tensor(cfg.ood_obs, dtype=torch.float64)
        if obs.ndim == 1:
            obs = obs[None, :]
        return [ObsTask(name=f"ood{i}", o=obs[i], is_ood=True) for i in range(obs.shape[0])]
    if cfg.obs_layout != "circle":
        raise ValueError("obs_layout='free' needs cfg.ood_obs (a list of OOD observation vectors)")
    # auto: midpoints of evenly-spaced gaps, spread around the circle
    step = 2 * math.pi / cfg.n_obs
    idx = torch.linspace(0, cfg.n_obs - 1, cfg.n_ood).round().long().unique()
    thetas = (idx.double() + 0.5) * step
    obs = _lift_circle(thetas, cfg.obs_dim, cfg.seed)
    return [ObsTask(name=f"ood{i}", o=obs[i], is_ood=True) for i in range(obs.shape[0])]
