"""Unified run-directory schema (the single output contract for every pipeline).

One self-describing directory per run lives under ``outputs/runs/<run-id>/``::

    <run-id> = <date>_<model>_<dataset>[_<tag>]          (HHMMSS injected if no tag)

    outputs/runs/<date>_pi05_libero-object_v1/
      run.json            # this module's RunMeta — model, dataset, args, git sha, lineage, stages
      eval/eval_info.json
      fm/manifest.json  fm/rollouts/rollout_0000.npz ...
      chunk_divergence/chunk_divergence_ro0.npz
      chunk_geometry/meta.json  chunk_geometry/accel_divergence_scatter.png
      videos/<task>/episode_<e>.mp4

A run is *created* by the producing eval, which writes ``run.json`` with ``stage`` = its own
name. Each downstream analysis stage resolves an existing run with :func:`resolve_run`, writes
into its own stage subdir, and calls :meth:`RunDir.record_stage` to log itself into ``run.json``
— so a partially-scored run is self-describing rather than ambiguous. Use :meth:`RunDir.child`
only for a genuinely *derived* run (new model/dataset), which sets ``parent_run`` for lineage.

This module is lerobot-free (stdlib + numpy via io), so it imports cleanly without
lerobot installed.
"""

from __future__ import annotations

import datetime
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fmaccel.core import io
from fmaccel.core.bootstrap import REPO_ROOT

RUN_JSON = "run.json"

# Canonical stage subdir names (also the keys recorded in run.json["stages"]).
STAGES = (
    "eval",                 # the closed-loop rollout that produced the run
    "fm",                   # the FM recording: Euler iterates + captured context
    "chunk_divergence",     # K resampled candidates per decision — the uncertainty GT
    "chunk_geometry",        # accel / Straightness / the pooled rho table
    "accel_profile",        # where along the denoise the bending happens
    "posterior", "resample",  # dense single-decision posterior diagnostics
    "obs_emb", "hidden_states", "fm_loss",  # GPU capture stages for the learned baselines
    "detectors",            # per-episode score streams per detector
    "flow_viz",             # velocity-field visualizations
)


def outputs_root() -> Path:
    """``$FMACCEL_OUTPUT_DIR`` or ``<repo>/outputs``."""
    return Path(os.environ.get("FMACCEL_OUTPUT_DIR", REPO_ROOT / "outputs"))


def runs_root() -> Path:
    return outputs_root() / "runs"


def _now() -> datetime.datetime:
    return datetime.datetime.now()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _conda_env() -> str | None:
    return os.environ.get("CONDA_DEFAULT_ENV")


def _slug(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-.") else "-" for c in str(s)).strip("-")


@dataclass
class RunMeta:
    run_id: str
    created: str
    git_sha: str | None
    conda_env: str | None
    stage: str                       # the producing pipeline (eval / toy_record / ...)
    parent_run: str | None = None    # set for derived runs (see RunDir.child)
    model: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)   # name -> {at, args} per analysis stage run

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunMeta":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class RunDir:
    """Handle to one ``outputs/runs/<run-id>/`` directory + its ``run.json``."""

    def __init__(self, root: Path, meta: RunMeta) -> None:
        self.root = Path(root)
        self.meta = meta

    # ---- construction -----------------------------------------------------
    @property
    def run_id(self) -> str:
        return self.meta.run_id

    @property
    def run_json_path(self) -> Path:
        return self.root / RUN_JSON

    def ensure(self) -> "RunDir":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def write_meta(self) -> "RunDir":
        self.ensure()
        io.write_json(self.run_json_path, asdict(self.meta))
        return self

    # ---- stage paths ------------------------------------------------------
    def stage_dir(self, stage: str, *, ensure: bool = True) -> Path:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; known: {STAGES}")
        d = self.root / stage
        if ensure:
            d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def eval_dir(self) -> Path: return self.stage_dir("eval")
    @property
    def fm_dir(self) -> Path: return self.stage_dir("fm")
    @property
    def fm_rollouts_dir(self) -> Path:
        d = self.fm_dir / "rollouts"
        d.mkdir(parents=True, exist_ok=True)
        return d
    @property
    def posterior_dir(self) -> Path: return self.stage_dir("posterior")
    @property
    def resample_dir(self) -> Path: return self.stage_dir("resample")
    @property
    def chunk_divergence_dir(self) -> Path: return self.stage_dir("chunk_divergence")
    @property
    def chunk_geometry_dir(self) -> Path: return self.stage_dir("chunk_geometry")
    @property
    def accel_profile_dir(self) -> Path: return self.stage_dir("accel_profile")
    @property
    def videos_dir(self) -> Path:
        d = self.root / "videos"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- lineage / bookkeeping -------------------------------------------
    def record_stage(self, stage: str, args: dict[str, Any] | None = None) -> "RunDir":
        """Log that an analysis ``stage`` ran on this run, then persist run.json."""
        self.meta.stages[stage] = {"at": _now().isoformat(timespec="seconds"), "args": args or {}}
        return self.write_meta()

    def child(self, *, model: dict, dataset: dict, stage: str, tag: str | None = None,
              args: dict | None = None) -> "RunDir":
        """Create a NEW derived run whose ``parent_run`` points back at this one."""
        return create_run(model=model, dataset=dataset, stage=stage, tag=tag,
                          args=args, parent_run=self.run_id)


def _make_run_id(model_name: str, dataset_slug: str, tag: str | None) -> str:
    now = _now()
    date = now.strftime("%Y-%m-%d")
    model_s, ds_s = _slug(model_name), _slug(dataset_slug)
    base = f"{date}_{model_s}_{ds_s}_{_slug(tag)}" if tag else f"{date}_{now.strftime('%H%M%S')}_{model_s}_{ds_s}"
    run_id, n = base, 2
    while (runs_root() / run_id).exists():
        run_id = f"{base}_{n}"
        n += 1
    return run_id


def create_run(*, model: dict[str, Any], dataset: dict[str, Any], stage: str,
               tag: str | None = None, args: dict[str, Any] | None = None,
               parent_run: str | None = None) -> RunDir:
    """Create ``outputs/runs/<run-id>/`` with a fresh ``run.json`` and return it.

    ``model``/``dataset`` are free-form metadata dicts (typically
    ``{"name", "checkpoint", "config"}`` and ``{"name", "task_group", "task_ids"}``).
    The dataset run-id slug is ``dataset["run_slug"]`` if present, else its name.
    """
    dataset_slug = dataset.get("run_slug") or dataset.get("task_group") or dataset.get("name", "data")
    run_id = _make_run_id(model.get("name", "model"), dataset_slug, tag)
    meta = RunMeta(
        run_id=run_id,
        created=_now().isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        conda_env=_conda_env(),
        stage=stage,
        parent_run=parent_run,
        model=model,
        dataset=dataset,
        args=args or {},
    )
    return RunDir(runs_root() / run_id, meta).write_meta()


def resolve_run(run_id_or_path: str | Path) -> RunDir:
    """Load an existing run from a run-id (under ``outputs/runs/``) or a path."""
    p = Path(run_id_or_path)
    root = p if p.exists() and (p / RUN_JSON).exists() else runs_root() / str(run_id_or_path)
    rj = root / RUN_JSON
    if not rj.exists():
        raise FileNotFoundError(f"no run.json at {root} (looked up {run_id_or_path!r})")
    return RunDir(root, RunMeta.from_dict(io.read_json(rj)))


def list_runs() -> list[str]:
    root = runs_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / RUN_JSON).exists())
