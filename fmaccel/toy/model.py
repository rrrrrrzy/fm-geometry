"""Conditional flow field v_theta(x, s, o), CFM training, and the Euler sampler.

The net is conditioned on a CONTINUOUS observation o (a small MLP encoder), so it
must learn a smooth o->field map — and its extrapolation off the training coverage
is exactly what the OOD analysis probes. Training and sampling use pi0.5's
s-convention (s=0 noise -> s=1 action); the sampler mirrors `sample_actions`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from fmaccel.toy.config import ToyConfig
from fmaccel.toy.data import ObsTask


class SinusoidalTimeEmbed(nn.Module):
    """Fourier features of the scalar flow-time s in [0, 1] (pi0.5-style)."""

    def __init__(self, dim: int = 64, max_period: float = 1.0e4):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)
        self.dim = dim

    def forward(self, s: Tensor) -> Tensor:
        s = s.reshape(-1, 1)
        ang = s * self.freqs[None, :] * (2.0 * math.pi)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class ObsFlowMLP(nn.Module):
    """v_theta(x, s, o): one flow field conditioned on a continuous observation o."""

    def __init__(self, d: int, obs_dim: int, hidden: int = 256, depth: int = 4,
                 time_dim: int = 64, cond_dim: int = 32):
        super().__init__()
        self.d = d
        self.obs_dim = obs_dim
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU(),
        )
        layers: list[nn.Module] = [nn.Linear(d + time_dim + cond_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, s: Tensor | float, o: Tensor) -> Tensor:
        n = x.shape[0]
        if not torch.is_tensor(s):
            s = torch.full((n,), float(s), dtype=x.dtype, device=x.device)
        s = s.to(x.dtype).reshape(-1)
        if s.shape[0] == 1 and n > 1:
            s = s.expand(n)
        o = o.to(device=x.device, dtype=x.dtype)
        if o.ndim == 1:
            o = o[None, :]
        if o.shape[0] == 1 and n > 1:
            o = o.expand(n, -1)
        h = torch.cat([x, self.time_embed(s), self.obs_enc(o)], dim=-1)
        return self.net(h)


def obs_velocity_fn(model: ObsFlowMLP, o: Tensor):
    """Slice the field at a fixed observation o -> (x, s) -> v (for euler_sample / Jacobian)."""
    o = o.detach()

    def fn(x: Tensor, s: float) -> Tensor:
        return model(x, s, o.to(x.device, x.dtype))

    return fn


def train_obs_flow(tasks: list[ObsTask], cfg: ToyConfig) -> tuple[ObsFlowMLP, list[tuple[int, float]]]:
    """CFM-train v_theta(x, s, o) on the observation clusters; returns (model, loss_log).

    The net only ever sees each task's FROZEN finite training set (``task.samples`` of
    size ``cfg.samples_per_obs`` — no infinite sampling). Each step picks a task per row
    (uniform), resamples x1 from that task's set with replacement, sets the observation
    o = nominal_o + eps * N(0, I) (the deployment jitter), x0 ~ N(0, I), and regresses
    the marginal velocity:  E || v_theta(x_s, s, o) - (x1 - x0) ||^2,  x_s = s*x1 + (1-s)*x0.

    The frozen sets are concatenated into one device tensor with per-task offsets, so the
    whole step is GPU-resident and gathered without a Python loop or host sync.
    """
    torch.manual_seed(cfg.seed)
    dev, dt = cfg.resolved_device(), cfg.dtype
    if dev.startswith("cuda"):
        torch.cuda.manual_seed(cfg.seed)
    d = tasks[0].target.d
    obs_dim = tasks[0].o.shape[0]
    model = ObsFlowMLP(d, obs_dim, cfg.hidden, cfg.depth, cfg.time_dim, cfg.cond_dim).to(dev, dt)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    nominal_dev = torch.stack([t.o for t in tasks]).to(dev, dt)        # (k, obs_dim)
    n_tasks = len(tasks)
    eps, B = cfg.obs_jitter, cfg.batch_size

    # frozen finite datasets, concatenated on device with per-task offsets
    data = torch.cat([t.samples for t in tasks]).to(dev, dt)          # (sum n_i, d)
    sizes = torch.tensor([t.samples.shape[0] for t in tasks], device=dev)         # (k,)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), sizes.cumsum(0)[:-1]])
    log: list[tuple[int, float]] = []

    for step in range(cfg.steps):
        tid = torch.randint(0, n_tasks, (B,), generator=gen).to(dev)             # which obs per row
        row_size = sizes[tid]                                                    # (B,)
        sel = (torch.rand(B, device=dev) * row_size).long().clamp_max_(row_size - 1)  # in [0, n_i)
        x1 = data[offsets[tid] + sel]                                            # gather from frozen set
        o = nominal_dev[tid] + eps * torch.randn(B, obs_dim, device=dev, dtype=dt)
        x0 = torch.randn(B, d, device=dev, dtype=dt)
        s = torch.empty(B, device=dev, dtype=dt).uniform_(cfg.s_eps, 1.0 - cfg.s_eps)
        x_s = s[:, None] * x1 + (1.0 - s[:, None]) * x0
        u = x1 - x0
        loss = ((model(x_s, s, o) - u) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            lv = float(loss.detach())
            log.append((step, lv))
            print(f"  step {step:6d}/{cfg.steps}   loss {lv:.4f}", flush=True)
        if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.steps - 1):
            _print_eval(model, tasks, cfg, step)
    model.eval()
    return model, log


def _print_eval(model: ObsFlowMLP, tasks: list[ObsTask], cfg: ToyConfig, step: int) -> None:
    """Quick in-training fidelity print: per-obs energy distance + learned count.

    Lazy-imports analysis.fidelity to avoid the model<->analysis import cycle.
    Toggles eval/train mode around the call (no dropout/BN here, but it's the right
    habit), and reports a one-line summary plus a compact per-obs detail.
    """
    from fmaccel.toy.analysis import fidelity   # lazy
    was_training = model.training
    model.eval()
    rows = [fidelity(model, t, cfg) for t in tasks]
    if was_training:
        model.train()
    learned = sum(r["learned"] for r in rows)
    mean_ed = sum(r["energy_dist_norm"] for r in rows) / len(rows)
    per = "  ".join(f"{r['name']}:{r['energy_dist_norm']:.2f}" for r in rows)
    print(f"    eval @ step {step:6d}: mean_ed_norm={mean_ed:.3f}  learned={learned}/{len(rows)}    {per}",
          flush=True)


@torch.no_grad()
def euler_sample(velocity_fn, n: int, d: int, *, num_steps: int = 50,
                 device: str = "cpu", dtype: torch.dtype = torch.float32,
                 generator: torch.Generator | None = None,
                 return_traj: bool = False, noise: Tensor | None = None):
    """Integrate dx/ds = v(x, s) from s=0 (noise) to s=1 (action), T Euler steps.

    Mirrors pi0.5's `sample_actions`: T steps of size ds = 1/T in the s-convention
    (so x += ds * v). Returns x_1 (n, d), or (x_1, traj (T+1,n,d), s_grid (T+1,)).
    """
    if noise is None:
        noise = torch.randn(n, d, device=device, dtype=dtype, generator=generator)
    x = noise.clone()
    ds = 1.0 / num_steps
    traj = [x.clone()] if return_traj else None
    for step in range(num_steps):
        x = x + ds * velocity_fn(x, step * ds)
        if return_traj:
            traj.append(x.clone())
    if return_traj:
        s_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
        return x, torch.stack(traj), s_grid
    return x
