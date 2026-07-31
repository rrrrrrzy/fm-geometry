"""Generation-fidelity readout for the toy velocity net.

Energy distance between the net's generated samples and its target samples, per observation.
Used by ``model._print_eval`` during training as a learning-quality gate: the geometric claim is
only meaningful once the net has actually fit its targets, so a bad fidelity number invalidates
the field reading rather than showing "uncertainty". No ground-truth flow field is ever used —
we only read what the net learned.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fmaccel.toy.config import ToyConfig
from fmaccel.toy.data import ObsTask
from fmaccel.toy.model import ObsFlowMLP, euler_sample, obs_velocity_fn


def _energy_distance(X: Tensor, Y: Tensor) -> float:
    """e = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||  (>= 0; 0 iff equal in distribution)."""
    a = torch.cdist(X, Y).mean()
    b = torch.cdist(X, X).mean()
    c = torch.cdist(Y, Y).mean()
    return float((2 * a - b - c).clamp_min(0.0))


@torch.no_grad()
def fidelity(model: ObsFlowMLP, task: ObsTask, cfg: ToyConfig, *, seed_offset: int = 7) -> dict:
    """Learning quality at a trained obs: generated samples vs the target distribution."""
    dev, dt = cfg.resolved_device(), cfg.dtype
    g = torch.Generator().manual_seed(cfg.seed + seed_offset)
    noise = torch.randn(cfg.eval_samples, model.d, generator=g, dtype=dt).to(dev)
    gen = euler_sample(obs_velocity_fn(model, task.o), cfg.eval_samples, model.d,
                       num_steps=cfg.num_steps, device=dev, dtype=dt, noise=noise).double().cpu()
    tgt = task.target.sample(cfg.eval_samples, generator=g).double()
    scale = float(task.target.cov().diagonal().sum().clamp_min(1e-12) ** 0.5)
    ed = _energy_distance(gen, tgt)
    rec = {"name": task.name, "K": task.n_modes,
           "energy_dist_norm": ed / scale,
           "mean_err": float((gen.mean(0) - task.target.mean()).norm())}
    if task.n_modes > 1:                           # mode coverage for the forks
        assign = torch.cdist(gen, task.target.means).argmin(1)
        emp_w = torch.bincount(assign, minlength=task.n_modes).double() / cfg.eval_samples
        rec["weight_tv"] = float(0.5 * (emp_w - task.target.weights).abs().sum())
        rec["min_mode_cover"] = float((emp_w / task.target.weights.clamp_min(1e-9)).min())
    learned = rec["energy_dist_norm"] < 0.12
    if task.n_modes > 1:
        learned = learned and rec["weight_tv"] < 0.12 and rec["min_mode_cover"] > 0.5
    rec["learned"] = bool(learned)
    return rec
