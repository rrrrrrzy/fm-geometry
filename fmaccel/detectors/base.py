"""The failure-detector abstraction: one carrier + one protocol.

Every failure detector in this repo — the free ``accel`` reference and every
literature *baseline* (FIPER, FAIL-Detect's score family, Diff-DAgger, …) — is a
:class:`Detector` that maps ONE re-plan decision (a :class:`ChunkRecord`) to a single
per-chunk scalar. That scalar then flows, identically for every detector, through the
shared evaluation harness (:mod:`fmaccel.detection.cusum`):
pooled / within-task / early-window AUROC, TPR@FPR, lead-time. Detectors are resolved
by name from :mod:`fmaccel.detectors.registry`, exactly like models and datasets.

Two invariants make the comparison fair and the code uniform:

* **One score convention.** ``score(rec)`` returns a scalar where **lower = more
  confident / less likely to be failing** (the native polarity of ``accel``). A
  detector whose raw signal points the other way (e.g. a density log-likelihood, where
  *higher* = in-distribution) flips its sign *inside* ``score`` so the harness never
  special-cases polarity.

* **One build path, two sources.** A :class:`ChunkRecord` is built either inline in the
  policy server from ``sample_chunks_with_traj`` (cheap online detectors that read only
  the recorded denoise path) or post-hoc from a run's ``fm/rollouts/*.npz`` (+ the
  ``_context.npz`` sidecar and resample npz) for detectors that also need the
  observation embedding, hidden states, or the MC-resample posterior. Same fields, same
  ``score`` call either way.

This module is **pure numpy** (dataclass + ABC) so it imports without lerobot
installed, like the rest of ``core``/``datasets``/``recording``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# The signals a detector may declare in ``requires``. Before calling ``score`` the
# harness guarantees the ChunkRecord carries every name a detector lists here, so a
# detector never has to defend against a missing input. This is the set of *requireable*
# signals, NOT a mirror of ChunkRecord's optional fields: it includes the always-present
# ``x_t`` (so a detector can list it explicitly), and it deliberately EXCLUDES
# ``chunk_actions`` because that is derivable from ``x_t[-1]`` (read it directly, with a
# fallback, rather than declaring it). ``extra`` is a free-form bag, also not requireable.
REQUIREABLE: frozenset[str] = frozenset(
    {"x_t", "v_t", "obs_emb", "hidden", "resample", "prev_chunk", "prev_resample",
     "resample_full", "prev_resample_full"}
)


@dataclass
class ChunkRecord:
    """Everything any failure detector could read about ONE re-plan decision.

    One *executed* candidate per decision — the failure-detection setting is ``k=1``
    observational sampling (the k-candidate *selection* lever is a separate, already-NULL
    concern). The only always-present payload is the recorded denoise trajectory ``x_t``
    of that chunk (what the free geometric proxies read); the rest are populated on
    demand by the recorder / adapter and stay ``None`` until a detector that needs them
    is run.

    Shapes (``T`` = #denoise steps, ``chunk`` = chunk_size, ``act`` = action_dim):

    * ``x_t``           ``(T+1, chunk, act)`` — every Euler iterate, noise→action (model-space).
    * ``v_t``           ``(T,   chunk, act)`` — velocity-field outputs at each step.
    * ``chunk_actions`` ``(chunk, act)``      — the executed chunk (== ``x_t[-1]`` truncated).
    * ``obs_emb``       ``(d,)``              — pooled observation/prefix conditioning embedding.
    * ``hidden``        ``{layer: (...,)}``   — transformer hidden states (e.g. action-expert last layer).
    * ``resample``      ``(N, exec, act)``    — N MC-resample endpoints, EXECUTED window only.
    * ``prev_chunk``    ``(chunk, act)``      — the previous decision's executed chunk (for temporal-consistency detectors).
    * ``prev_resample`` ``(N, exec, act)``    — the previous decision's executed-window resample set.
    * ``resample_full`` ``(N, chunk, act)``   — the SAME resample set over the FULL chunk horizon.
    * ``prev_resample_full`` ``(N, chunk, act)`` — the previous decision's full-horizon resample set.

    ⚠️ ``resample`` / ``prev_resample`` are truncated to the *executed* window (``first_actions ==
    n_exec``, what ``chunk_divergence`` writes) so the spread they report is the spread of the plan
    that actually ran — the window ``accel`` is labelled on. A **cross-time** detector cannot use
    them: consecutive executed windows cover DISJOINT absolute timesteps, so comparing them measures
    "the robot moved", not "the plan changed". Those detectors (STAC) must require the ``_full``
    pair, whose overlap ``cur[:, :chunk-n_exec]`` vs ``prev[:, n_exec:]`` is the same absolute future.
    """

    # --- always present: the free denoise-path signal + bookkeeping ---
    x_t: np.ndarray
    action_dim: int
    env_step: int = 0
    task_id: int | None = None
    task_group: str | None = None
    n_exec: int | None = None  # executed window = min(n_action_steps, chunk_size)
    time: np.ndarray | None = None  # (T,) FM schedule, descending ~1.0 → ~0.1

    # --- optional signals (None until supplied); names mirror REQUIREABLE ---
    v_t: np.ndarray | None = None
    chunk_actions: np.ndarray | None = None
    obs_emb: np.ndarray | None = None
    hidden: Mapping[str, np.ndarray] | None = None
    resample: np.ndarray | None = None       # (N, exec, act) EXECUTED-window resample set
    prev_chunk: np.ndarray | None = None
    prev_resample: np.ndarray | None = None  # (N, exec, act) previous decision's executed-window set
    # Full-horizon counterparts — the ONLY ones a cross-time detector may use (see class docstring).
    resample_full: np.ndarray | None = None       # (N, chunk, act)
    prev_resample_full: np.ndarray | None = None  # (N, chunk, act)
    extra: dict[str, Any] = field(default_factory=dict)

    def has(self, signal: str) -> bool:
        """True iff this record carries the named signal (``REQUIREABLE`` member)."""
        return getattr(self, signal, None) is not None


class Detector(ABC):
    """A per-chunk failure-detection signal.

    Subclasses set three class attributes and implement :meth:`score` (and, if learned,
    :meth:`fit`). The constructor holds hyper-parameters; ``fit`` does any offline
    training / threshold calibration on a held-out set of (success-only or labeled)
    rollouts; ``score`` reads one :class:`ChunkRecord` and returns one scalar in the
    accel convention (lower = more confident).
    """

    #: registry / score-key name (also the column key in the eval harness)
    name: str = "detector"
    #: which ChunkRecord signals this detector needs (subset of REQUIREABLE)
    requires: frozenset[str] = frozenset({"x_t"})
    #: True iff pure-numpy off the recorded denoise path, cheap enough to run inline in
    #: the server (like accel); False iff it needs a model re-forward / aux net / resample
    online: bool = False
    #: True iff ``fit`` needs FAILURE-labeled rollouts, not just success-only. The default
    #: unsupervised contract is ``fit(success_rollouts)`` (embedding-OOD family, ACE cell
    #: calibration, …); a supervised detector (e.g. SAFE, category-C) instead expects
    #: ``fit(labeled_episodes)`` where each item is ``(records: list[ChunkRecord], success:
    #: bool)`` — an episode's ordered ChunkRecords plus its trajectory-level outcome. The
    #: harness routes the two by this flag and holds the fit episodes out of scoring either way.
    supervised: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        bad = set(cls.requires) - REQUIREABLE
        if bad:
            raise TypeError(
                f"{cls.__name__}.requires has unknown signal(s) {sorted(bad)}; "
                f"allowed: {sorted(REQUIREABLE)}"
            )

    def fit(self, success_rollouts: Iterable[Any] | None = None) -> None:
        """Offline calibration / training. No-op by default (training-free detectors).

        Implementations MUST only ever see calibration / training rollouts, never the
        held-out test rollouts the harness scores — keeping the splits disjoint is the
        caller's job, but a detector should not stash test data here.
        """
        return None

    @abstractmethod
    def score(self, rec: ChunkRecord) -> float:
        """One per-chunk scalar, ACCEL CONVENTION: lower = more confident / less failing.

        Flip any opposite-polarity native signal here so the harness stays sign-agnostic.
        """
        ...

    def score_stream(self, recs: Sequence[ChunkRecord]) -> list[float]:
        """Per-chunk scores for ONE episode's ORDERED :class:`ChunkRecord` list.

        Default = independent per-chunk :meth:`score` (the memoryless contract every
        geometric / embedding / resample detector satisfies). A detector whose per-step
        score depends on the episode history (a recurrent probe like SAFE-LSTM, whose
        ``s_t = σ(LSTM(e_{0:t}))``) overrides this to consume the whole ordered sequence
        at once; the shared harness then does the cumulative-sum / conformal-band
        aggregation on top of the returned stream, identically for every detector.
        """
        return [float(self.score(r)) for r in recs]

    def check(self, rec: ChunkRecord) -> None:
        """Raise if ``rec`` is missing any signal this detector declared in ``requires``."""
        missing = [s for s in self.requires if not rec.has(s)]
        if missing:
            raise ValueError(
                f"detector {self.name!r} needs signals {sorted(self.requires)} but the "
                f"ChunkRecord is missing {missing}"
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r} online={self.online} requires={sorted(self.requires)}>"


def stack_signal(rollouts: Any, attr: str = "obs_emb") -> np.ndarray:
    """Stack a per-record signal from a ``fit`` input into ``(N, d)``.

    Learned detectors (RND-OE, logpZO, the density family) train on a matrix of success-rollout
    signals. This accepts that matrix three ways so callers stay flexible: an ``(N, d)`` ndarray
    as-is; an iterable of :class:`ChunkRecord` (the named ``attr`` is pulled from each); or an
    iterable of raw vectors / dicts. Rows missing the signal are skipped.
    """
    if isinstance(rollouts, np.ndarray):
        return rollouts.astype(np.float32).reshape(rollouts.shape[0], -1)
    rows: list[np.ndarray] = []
    for r in rollouts:
        if isinstance(r, ChunkRecord):
            v = getattr(r, attr, None)
        elif isinstance(r, dict):
            v = r.get(attr)
        else:
            v = r
        if v is None:
            continue
        rows.append(np.asarray(v, np.float32).reshape(-1))
    if not rows:
        raise ValueError(f"stack_signal: no {attr!r} found in fit() input")
    return np.stack(rows)
