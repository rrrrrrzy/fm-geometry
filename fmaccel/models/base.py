"""The model-adapter contract: exactly what FM tooling needs from a policy.

A :class:`FMModelAdapter` exposes a loaded ``policy`` (``reset`` /
``select_action`` / ``predict_action_chunk``), the FM-head ``model``, an
:class:`FMConfig` of dims, and the FM-head hook pair:

  * :meth:`attach_fm_hooks` / :meth:`detach_fm_hooks` — patch the model's own
    ``sample_actions`` / ``denoise_step`` (whatever their signature) and call back
    into the recorder's :meth:`~fmaccel.recording.recorder.FMRecorder.on_sample_actions`
    / :meth:`~...on_denoise_step`.

Adding a model = subclass this, implement ``build`` + the two hooks, and add one
line to :data:`fmaccel.registry._MODELS`. Stays torch/lerobot-free here (only
typing + ABCs), so importing the contract never pulls a backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class FMConfig:
    """Dims the recorder/resampler read off a model (model-space)."""

    num_inference_steps: int
    chunk_size: int
    max_action_dim: int       # padded model action dim (32 for π₀.₅, == action_dim for the toy)
    n_action_steps: int
    action_dim: int           # real (un-padded) action dim (7 for LIBERO)

    @classmethod
    def from_policy_config(cls, pc: Any, action_dim: int) -> "FMConfig":
        return cls(
            num_inference_steps=int(pc.num_inference_steps),
            chunk_size=int(pc.chunk_size),
            max_action_dim=int(pc.max_action_dim),
            n_action_steps=int(pc.n_action_steps),
            action_dim=int(action_dim),
        )


class FMModelAdapter(ABC):
    name: ClassVar[str] = "?"
    supports_context: ClassVar[bool] = False     # can stash exact FM-head prefix inputs
    # An adapter using the *custom-context* pattern (its own ``ctx_*`` schema +
    # ``sample_with_context``, not the π₀.₅ ``ChunkResampleSession``) sets this True and
    # implements :meth:`resample_context_chunk`, so the chunk_divergence stage can drive its
    # resample directly. Adapters on the default π₀.₅ schema leave it False (they resample
    # via ``ChunkResampleSession``); see ``pipelines/chunk_divergence._run_context_divergence``.
    supports_context_resample: ClassVar[bool] = False
    # Set True + implement :meth:`embed_context_obs` to materialize a pooled prefix
    # embedding from captured context — unlocks the embedding-OOD detectors
    # (rnd_oe / logpzo / pca_kmeans / knn / mahalanobis / fiper). See ``pipelines/obs_emb``.
    supports_obs_emb: ClassVar[bool] = False
    # Set True + implement :meth:`embed_context_hidden` to materialize the action-expert
    # last-layer hidden feature from captured context — unlocks the SAFE supervised probe.
    # See ``pipelines/hidden_states``.
    supports_hidden: ClassVar[bool] = False
    # Set True + implement :meth:`velocity_fn_for_context` to build a per-chunk velocity
    # callable from captured context — unlocks the ``fm_loss`` (Diff-DAgger) detector.
    supports_fm_velocity: ClassVar[bool] = False
    # Max acceptable resample/reproduce error for the chunk_divergence fidelity gate. An fp32-head
    # adapter (π₀.₅) reproduces ~bit-exactly → 1e-2. A bf16-autocast head has an eager-attention
    # floor ~5e-2 even when the dtype flow is matched, so such an adapter overrides this.
    # run_chunk_divergence defaults reproduce_tol from the resolved adapter.
    resample_reproduce_tol: ClassVar[float] = 1e-2
    # fm_loss velocity-contract convention. ``velocity_fn_for_context`` must return the velocity of
    # the fm_loss CLEAN-DATA interpolant ``x_τ=(1−τ)·x0+τ·â`` (target ``â−x0``). Two families:
    #   * pi05 (native ``x_s=s·noise+(1−s)·action``, ``u=noise−action``, s:1→0) → the closure is the
    #     time+sign reverse, so at a recorded FM time ``s`` the raw model velocity is ``−closure(x,1−s)``
    #     ⇒ ``fm_velocity_matches_recorded = False`` (the gate must flip+reverse-time to compare v_t).
    #   * clean-data-native heads (native ``x_τ=(1−τ)·noise+τ·action``, ``v=action−noise``, τ:0→1 ==
    #     the clean-data direction) → the closure returns ``+v_θ(x,τ)`` at the SAME time convention the
    #     recorder stored ⇒ ``fm_velocity_matches_recorded = True`` (gate compares closure(x,s) to v_t
    #     directly). Only affects the DIAGNOSTIC contract gate; the fm_loss score itself is identical.
    fm_velocity_matches_recorded: ClassVar[bool] = False

    def __init__(self, policy: Any, config: FMConfig) -> None:
        self.policy = policy
        self.model = policy.model
        self.config = config
        # The resolved checkpoint path actually loaded (even if defaulted from an
        # env var). Stored in run.json so analysis stages can rebuild the policy.
        self.checkpoint: str | None = None
        # Set by build_eval (vec-env closed-loop): the lerobot pre/post processors
        # and the resolved policy config. None for the standalone build() path.
        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self.policy_cfg: Any = None

    # ---- construction -----------------------------------------------------
    @classmethod
    @abstractmethod
    def build(cls, checkpoint: str, *, device: str = "cuda", action_dim: int | None = None,
              n_action_steps: int | None = None, num_inference_steps: int | None = None,
              disable_compile: bool = True, dataset: Any = None) -> "FMModelAdapter":
        """Load a policy STANDALONE from ``checkpoint`` (no env) and wrap it — for
        resample / serving / teacher-forced recording. ``action_dim`` (or
        ``dataset.action_dim``) fixes the real action dim; ``disable_compile`` must
        be True whenever the result will be recorded (compile breaks the hooks)."""

    @classmethod
    def build_eval(cls, checkpoint: str, *, env_cfg: Any, device: str = "cuda",
                   action_dim: int = 7, n_action_steps: int | None = None,
                   num_inference_steps: int | None = None, disable_compile: bool = True,
                   compile_mode: str = "max-autotune") -> "FMModelAdapter":
        """Build for a vec-env closed-loop eval (env-derived features + the
        pre/post processors), attaching ``preprocessor``/``postprocessor``. Adapters
        without a vec-env eval path (e.g. the toy) leave this unimplemented."""
        raise NotImplementedError(f"{cls.__name__} has no vec-env eval build (build_eval)")

    # ---- input preprocessing (model-specific) -----------------------------
    def preprocess(self, batch: Any) -> Any:
        """Make a raw obs ``batch`` (assembled by the dataset's ``chunk_inputs``)
        model-ready before ``predict_action_chunk``. Default: identity — the state
        toys consume raw ``{observation.state, task}`` directly. π₀.₅ overrides to run
        its policy preprocessor (state normalize + prompt discretize + tokenize +
        device), matching the live LIBERO eval."""
        return batch

    # ---- FM-head hook surface (model-specific) ----------------------------
    @abstractmethod
    def attach_fm_hooks(self, recorder: Any) -> None: ...

    @abstractmethod
    def detach_fm_hooks(self) -> None: ...

    # ---- custom-context resample (model-specific; only when supports_context_resample)
    def resample_context_chunk(
        self, ctx_arrays: dict, chunk_idx: int, *, k: int, seed: int,
        num_inference_steps: int | None = None,
    ) -> Any:
        """Resample ``k`` candidate chunks at one recorded chunk's context.

        For custom-context adapters (``supports_context_resample=True``): slice this
        chunk's per-chunk context out of the loaded ``ctx_arrays`` (the rollout's
        ``ctx_*`` sidecar), draw ``k`` fresh FM noises seeded by ``seed``, and return the
        model-space candidate chunks as a numpy array ``(k, chunk_size, max_action_dim)``.
        Default raises; a custom-context adapter overrides via :meth:`sample_with_context`."""
        raise NotImplementedError(
            f"{type(self).__name__} has no resample_context_chunk; either it uses the π₀.₅ "
            f"ChunkResampleSession path (supports_context_resample=False) or it is unimplemented.")

    def embed_context_obs(self, ctx_arrays: dict, chunk_idx: int) -> Any:
        """Pooled observation/prefix embedding at one recorded chunk's context → ``(d,)``.

        For adapters with ``supports_obs_emb=True``: the conditioning vector the
        embedding-OOD detectors read as ``rec.obs_emb``. Default raises; π₀.₅ overrides
        (masked-mean-pool of ``embed_prefix``). Must return the SAME embedding at fit + score."""
        raise NotImplementedError(
            f"{type(self).__name__} has no embed_context_obs (supports_obs_emb=False); the "
            f"embedding-OOD detectors (rnd_oe/logpzo/density/fiper) are unavailable for it.")

    def embed_context_hidden(self, ctx_arrays: dict, chunk_idx: int) -> Any:
        """Action-expert LAST-LAYER hidden feature at one recorded chunk's context → ``(d,)``.

        For adapters with ``supports_hidden=True``: the internal feature SAFE's supervised probe
        reads as ``rec.hidden['action_expert_last']``. SAFE (arXiv:2506.09937,
        ``failure_prob/data/pizero.py``) takes the action-expert last-layer hidden tensor
        ``(n_diff_steps, n_pred_horizon, d)`` — for π₀.₅ the ``suffix_out`` fed to
        ``action_out_proj`` at each denoise step — and reduces it to one vector by aggregating the
        HORIZON axis first, then the FLOW-STEP axis (PizeroDatasetConfig default = mean over both).
        This method must return that reduced ``(d,)`` vector, computed by re-running the FM head at
        the recorded context (torch.compile OFF so the hook fires). Default raises; π₀.₅ overrides."""
        raise NotImplementedError(
            f"{type(self).__name__} has no embed_context_hidden (supports_hidden=False); the "
            f"SAFE supervised probe is unavailable for it.")

    def velocity_fn_for_context(self, ctx_arrays: dict, chunk_idx: int) -> Any:
        """Per-chunk velocity callable for ``fm_loss`` (``supports_fm_velocity=True``).

        Returns ``fn(x_t (M,chunk,act), t (M,)) -> v`` closing over this chunk's recorded
        conditioning, in fm_loss's clean-data convention (see ``detectors/fm_loss.py`` for the
        ``−v_θ(x_τ, 1−τ)`` contract). Default raises; π₀.₅ overrides."""
        raise NotImplementedError(
            f"{type(self).__name__} has no velocity_fn_for_context (supports_fm_velocity=False); "
            f"the fm_loss detector is unavailable for it.")
