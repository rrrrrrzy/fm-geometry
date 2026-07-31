"""The 2-D toy: a conditional flow-matching velocity net with full control of the target.

One conditional flow net ``v_theta(x, s, o)`` (``ObsFlowMLP``) conditioned on a plain
observation vector. Because we choose each observation's action target ourselves, the toy is
where the paper's geometric claim can be checked against a *known* posterior: unimodal targets
(certain), 2-mode targets (an aleatoric fork), and held-out observations the net never saw
(epistemic). Reading the learned field at those three conditions is the top row of the
flow-field figure, and the toy's ``accel`` ↔ posterior-spread correlation is the controlled
counterpart of the real-model result.

Contents: the net (``model``), its config and target generator (``config``/``data``), and the
generation-fidelity scorer (``analysis.fidelity``, used as a training-quality gate).

Top-level names are exposed LAZILY (PEP 562) so importing ``ObsFlowMLP`` never pulls a
matplotlib/analysis dependency. This package is torch+numpy only — no policy, no lerobot.

Conventions: linear-interpolant CFM, x_s = s*x_1 + (1-s)*x_0, x_0 ~ N(0,I),
s=0 noise -> s=1 action (pi0.5 internal t = 1 - s).
"""

import importlib

# attribute name -> defining submodule (imported on first access only)
_LAZY = {
    "ToyConfig": "fmaccel.toy.config",
    "GMMTarget": "fmaccel.toy.data",
    "ObsTask": "fmaccel.toy.data",
    "build_train_tasks": "fmaccel.toy.data",
    "build_ood_tasks": "fmaccel.toy.data",
    "ObsFlowMLP": "fmaccel.toy.model",
    "obs_velocity_fn": "fmaccel.toy.model",
    "train_obs_flow": "fmaccel.toy.model",
    "euler_sample": "fmaccel.toy.model",
    "fidelity": "fmaccel.toy.analysis",
}


def __getattr__(name):  # PEP 562 module-level lazy attribute access
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ToyConfig",
    "GMMTarget",
    "ObsTask",
    "build_train_tasks",
    "build_ood_tasks",
    "ObsFlowMLP",
    "obs_velocity_fn",
    "train_obs_flow",
    "euler_sample",
    "fidelity",
]
