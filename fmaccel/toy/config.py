"""The ONE place to set every knob of the toy.

``ToyConfig`` below holds every default; nothing else in the toy carries its own. The paper's
appendix specifies this configuration in full (ring size, target shapes, net widths, optimizer,
step count, seed) — each stated hyperparameter maps to one field here, so the flow-field figure
is reproducible from a clean checkout. ``experiments/flowfield_toy.py`` builds the instance.

What the toy does (no ground-truth flow field anywhere — we only ever read what the *net*
learned):

  1. data  — k observations in R^n, each carrying a Gaussian-mixture ACTION target: some
             2-mode (an aleatoric fork), the rest unimodal (confident). Held-out observations
             in the gaps between training angles give the epistemic condition.
  2. train — one conditional flow field v_theta(x, s, o) on the k observations, with optional
             observation jitter; the training loss and the generation fidelity are logged as
             learning-quality signals.
  3. read  — at each observation, draw the learned field, integrate trajectories, and measure
             ``accel`` against the resampled endpoint spread.

Conventions (linear-interpolant CFM, π₀.₅'s s-convention):
    x_s = s * x_1 + (1 - s) * x_0,   x_0 ~ N(0, I),   s=0 noise -> s=1 action.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ToyConfig:
    # =================================================================== dims
    action_dim: int = 8          # d: action/state space (2 gives clean flow-field figures)
    obs_dim: int = 32             # n: observation-vector dimension (>= 2)

    # ===================================================== data: observations
    # Where the k observations sit in R^n, and which OOD points to probe.
    obs_layout: str = "circle"   # "circle": k pts on a 2-D circle lifted into R^n
                                 # "free":   you list them in `train_obs` below
    n_obs: int = 4               # circle: number of training observations (k)

    # free layout only — explicit training observation vectors (each length obs_dim):
    train_obs: list | None = None

    # OOD observations to probe (held-out / extrapolation conditions):
    #   circle + ood_obs=None -> auto: midpoints of the `n_ood` widest angular gaps
    #   free                  -> you must list ood_obs (each length obs_dim)
    ood_obs: list | None = None
    n_ood: int = 2               # circle: how many gap-midpoint OOD angles to probe

    # ============================================= data: action target per obs
    # Each observation gets a Gaussian-mixture action target. `n_multimodal` of
    # the k are multimodal (aleatoric forks); the rest are single narrow Gaussians.
    n_multimodal: int = 2        # how many of the k observations are multimodal
    modes_range: tuple[int, int] = (2, 3)            # # modes drawn per multimodal obs
    mode_sep: float = 1.5        # min pairwise distance between a target's mode means
    mode_box: float = 2.5        # action_dim==2 path: means drawn in [-box, box]^d
    mode_radius: tuple[float, float] | None = None   # high-d path: means on a shell, ||mu|| in [lo,hi]
    sigma_range: tuple[float, float] = (0.10, 0.25)  # per-mode std (drawn uniformly in range)
    weight_temp: float = 0.4     # mixture-weight spread (softmax temp on N(0,1) logits; 0 -> uniform-ish)

    # ============================================================== training
    # Training data is a FROZEN FINITE set per observation (NO infinite sampling):
    # each observation's target is sampled exactly this many times ONCE, and training
    # only ever resamples (with replacement) from that fixed set.
    #   int        -> same count for every observation
    #   list[int]  -> per-observation counts (length must equal the number of train obs)
    # Small values (e.g. 32 / 64) are the data-scarcity regime.
    samples_per_obs: int | list[int] = 4096
    obs_jitter: float = 0.01      # eps: per-step obs jitter o += eps * N(0,I). 0 = none; ~0.02 = tiny
    steps: int = 80000
    batch_size: int = 1024
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    hidden: int = 256            # MLP width
    depth: int = 4               # MLP layers
    time_dim: int = 64           # sinusoidal flow-time embedding width
    cond_dim: int = 32           # observation-encoder width
    s_eps: float = 1.0e-3        # clamp flow-time s away from {0,1} during training
    log_every: int = 500         # training-loss log / print cadence
    eval_every: int = 5000       # in-training fidelity (energy distance) print cadence (0 = off)

    # ============================================================== analysis
    n_traj: int = 256            # # noise seeds / trajectories integrated per observation
    num_steps: int = 20          # Euler integration steps T. Raise when reading geometry: Euler
                                 # discretization error masquerades as curvature, so too few steps
                                 # inflate accel. The figure uses 40.
    eval_samples: int = 2000     # # generated samples for the per-obs fidelity (learning-quality) check

    # ============================================================ run / output
    device: str = "cuda"         # "cuda" or "cpu"
    dtype_str: str = "float32"   # "float32" (fast) or "float64"
    seed: int = 0
    threads: int = 8             # cap torch CPU threads (a many-core default thrashes a tiny MLP)
    output_dir: str = "outputs/toy"

    # ---- derived ----------------------------------------------------------
    @property
    def dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype_str == "float64" else torch.float32

    def resolved_device(self) -> str:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def samples_list(self, k: int) -> list[int]:
        """Per-observation frozen training-set sizes for k observations."""
        if isinstance(self.samples_per_obs, int):
            return [self.samples_per_obs] * k
        if len(self.samples_per_obs) != k:
            raise ValueError(
                f"samples_per_obs has {len(self.samples_per_obs)} entries but there are "
                f"{k} training observations")
        return [int(x) for x in self.samples_per_obs]
