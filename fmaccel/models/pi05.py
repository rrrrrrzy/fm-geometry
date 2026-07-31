"""``Pi05Adapter`` — π₀.₅ (LeRobot) as a model adapter.

Imports lerobot, so it is resolved ONLY via ``get_model("pi05")`` (never at
package import) and only runs under the the project environment env. Loads the policy with the
canonical context-mode build (mirrors ``resample.build_resample_policy`` and
``pi05_policy_server._build``: ``PI05Policy.from_pretrained`` + compile disabled)
and patches the prefix-conditioned FM head for recording.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from fmaccel.models.base import FMConfig, FMModelAdapter

DEFAULT_CHECKPOINT_ENV = "PI05_LIBERO_CHECKPOINT"
_FALLBACK_CHECKPOINT = "data/models/pi05-libero-hf"   # repo-relative; override with --checkpoint


class Pi05Adapter(FMModelAdapter):
    name = "pi05"
    supports_context = True
    # π₀.₅'s FM head: PaliGemma prefix + Gemma action expert; embed_prefix -> denoise_step
    # ending in action_out_proj(suffix_out); CFM convention x_s = s·noise + (1-s)·action,
    # u = noise - action, t: 1->0. π₀.₅ has NO separate state vector (it folds state into the
    # discretized prompt tokens), so the default context schema is
    # {ctx_images, ctx_img_masks, ctx_tokens, ctx_masks} and embed_prefix takes
    # (images, img_masks, tokens, masks). supports_context_resample stays False: resample keeps
    # going through the generic ChunkResampleSession (the π₀.₅ path).
    supports_obs_emb = True        # pooled embed_prefix embedding (rnd_oe/logpzo/density/fiper)
    supports_hidden = True         # action-expert last-hidden = suffix_out -> action_out_proj (SAFE)
    supports_fm_velocity = True    # per-chunk velocity callable from captured context (fm_loss)

    @classmethod
    def resolve_checkpoint(cls, checkpoint: str | None = None) -> str:
        """The checkpoint path ``build``/``build_eval`` would load: explicit arg →
        ``PI05_LIBERO_CHECKPOINT`` env-var → bundled ``pi05-libero-hf`` fallback.
        Exposed so callers (e.g. the LIBERO dataset's camera auto-detection) can read
        the *same* checkpoint config this adapter will load, without loading weights."""
        return str(checkpoint or os.environ.get(DEFAULT_CHECKPOINT_ENV) or _FALLBACK_CHECKPOINT)

    @classmethod
    def build(cls, checkpoint: str | None = None, *, device: str = "cuda",
              action_dim: int | None = None, n_action_steps: int | None = None,
              num_inference_steps: int | None = None, disable_compile: bool = True,
              dataset: Any = None) -> "Pi05Adapter":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        ckpt = cls.resolve_checkpoint(checkpoint)
        policy = PI05Policy.from_pretrained(str(ckpt))
        policy.config.device = device
        if disable_compile and hasattr(policy.config, "compile_model"):
            policy.config.compile_model = False  # keep denoise_step patchable
        if n_action_steps is not None and hasattr(policy.config, "n_action_steps"):
            policy.config.n_action_steps = int(n_action_steps)
        if num_inference_steps is not None and hasattr(policy.config, "num_inference_steps"):
            policy.config.num_inference_steps = int(num_inference_steps)
        policy.to(device)
        policy.eval()

        adim = action_dim
        if adim is None and dataset is not None:
            adim = getattr(dataset, "action_dim", None)
        if adim is None:
            adim = 7  # LIBERO default
        adapter = cls.from_policy(policy, action_dim=int(adim))
        adapter.checkpoint = str(ckpt)
        return adapter

    @classmethod
    def build_eval(cls, checkpoint: str | None = None, *, env_cfg: Any, device: str = "cuda",
                   action_dim: int = 7, n_action_steps: int | None = None,
                   num_inference_steps: int | None = None, disable_compile: bool = True,
                   compile_mode: str = "max-autotune") -> "Pi05Adapter":
        """Closed-loop LIBERO build: env-derived features via ``make_policy`` +
        the pre/post processors (mirrors the original eval_pi05_libero assembly)."""
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_policy, make_pre_post_processors

        ckpt = cls.resolve_checkpoint(checkpoint)
        policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt))
        policy_cfg.pretrained_path = ckpt
        policy_cfg.device = device
        if disable_compile and hasattr(policy_cfg, "compile_model"):
            policy_cfg.compile_model = False
        if not disable_compile and hasattr(policy_cfg, "compile_mode"):
            policy_cfg.compile_mode = compile_mode
        if n_action_steps is not None and hasattr(policy_cfg, "n_action_steps"):
            chunk = getattr(policy_cfg, "chunk_size", n_action_steps)
            if n_action_steps > chunk:
                raise ValueError(f"n_action_steps={n_action_steps} exceeds ckpt chunk_size={chunk}")
            policy_cfg.n_action_steps = int(n_action_steps)
        if num_inference_steps is not None and hasattr(policy_cfg, "num_inference_steps"):
            policy_cfg.num_inference_steps = int(num_inference_steps)

        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(ckpt),
            preprocessor_overrides={
                "device_processor": {"device": str(policy.config.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        adapter = cls.from_policy(policy, action_dim=int(action_dim))
        adapter.checkpoint = str(ckpt)
        adapter.preprocessor = preprocessor
        adapter.postprocessor = postprocessor
        adapter.policy_cfg = policy_cfg
        return adapter

    @classmethod
    def from_policy(cls, policy: Any, *, action_dim: int) -> "Pi05Adapter":
        return cls(policy, FMConfig.from_policy_config(policy.config, action_dim))

    # ---- input preprocessing ---------------------------------------------
    def ensure_preprocessor(self) -> Any:
        """Build (once) the lerobot policy preprocessor straight from the checkpoint —
        for the standalone ``build()`` path (teacher-forced labeling), where
        ``build_eval`` never ran. Mirrors that env-free assembly minus the env
        feature override (a pretrained π₀.₅ checkpoint already carries the right
        input/output features), so it normalizes state, discretizes it into the
        prompt, tokenizes and moves to device exactly as the live LIBERO eval does."""
        if self.preprocessor is None:
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies import make_pre_post_processors

            cfg = PreTrainedConfig.from_pretrained(str(self.checkpoint))
            cfg.pretrained_path = self.checkpoint
            cfg.device = str(self.policy.config.device)
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=cfg, pretrained_path=str(self.checkpoint),
                preprocessor_overrides={"device_processor": {"device": str(self.policy.config.device)}},
            )
            self.policy_cfg = cfg
        return self.preprocessor

    # The two camera keys the libero_demos adapter emits (exterior, wrist). These are
    # the *base* π₀.₅ slot names; _remap_image_keys retargets them to whatever image
    # slots the LOADED checkpoint actually declares (identity for base).
    _SRC_EXTERIOR = "observation.images.base_0_rgb"
    _SRC_WRIST = "observation.images.left_wrist_0_rgb"

    def _expected_image_slots(self) -> tuple[str | None, str | None]:
        """``(exterior_key, wrist_key)`` the loaded checkpoint declares, read from its
        ``input_features``: drop the auto-masked empty / right-wrist slots, then the
        first remaining VISUAL key is the exterior camera and the second the wrist —
        matching lerobot's LIBERO mapping (agentview->image, eye_in_hand->image2) and
        the base DROID layout (base_0_rgb, left_wrist_0_rgb) alike."""
        feats = getattr(self.policy.config, "input_features", {}) or {}
        img = [k for k, v in feats.items()
               if "VISUAL" in str(getattr(v, "type", "")) and
               "empty" not in k and "right_wrist" not in k]
        return (img[0] if img else None, img[1] if len(img) > 1 else None)

    def _remap_image_keys(self, batch: Any) -> Any:
        """Rename the dataset's exterior/wrist camera keys onto the checkpoint's declared
        image slots. No-op for ``pi05-base-hf`` (slots already match); renames
        base_0_rgb->image, left_wrist_0_rgb->image2 for ``pi05-libero-hf``."""
        if not isinstance(batch, dict):
            return batch
        ext, wr = self._expected_image_slots()
        out = dict(batch)
        if ext and ext != self._SRC_EXTERIOR and self._SRC_EXTERIOR in out:
            out[ext] = out.pop(self._SRC_EXTERIOR)
        if wr and wr != self._SRC_WRIST and self._SRC_WRIST in out:
            out[wr] = out.pop(self._SRC_WRIST)
        return out

    def preprocess(self, batch: Any) -> Any:
        """Run the π₀.₅ policy preprocessor over a raw camera+state obs batch (LIBERO
        labeling). The dataset emits exterior/wrist under the base slots
        (``base_0_rgb`` / ``left_wrist_0_rgb``); :meth:`_remap_image_keys` retargets them
        to the loaded checkpoint's declared slots (``image`` / ``image2`` for
        ``pi05-libero-hf``), and the leftover empty / right-wrist slot is masked inside
        ``PI05Policy._preprocess_images``."""
        return self.ensure_preprocessor()(self._remap_image_keys(batch))

    def attach_fm_hooks(self, recorder: Any) -> None:
        model = self.model
        self._orig_sample = model.sample_actions
        self._orig_denoise = model.denoise_step

        def _sample(images, img_masks, tokens, masks, noise=None, num_steps=None, **kw):
            if noise is None:
                shape = (tokens.shape[0], recorder.chunk_size, recorder.max_action_dim)
                noise = model.sample_noise(shape, tokens.device)
            ctx = None
            if recorder.capture_context:
                # fp32 images are a lossless superset of any bf16/fp16 compute dtype,
                # so the rebuilt prefix matches to eager-attention float noise.
                ctx = {
                    "images": [im.detach().to("cpu", torch.float32) for im in images],
                    "img_masks": [m.detach().to("cpu") for m in img_masks],
                    "tokens": tokens.detach().to("cpu"),
                    "masks": masks.detach().to("cpu"),
                }
            recorder.on_sample_actions(noise, ctx)
            return self._orig_sample(images, img_masks, tokens, masks,
                                     noise=noise, num_steps=num_steps, **kw)

        def _denoise(prefix_pad_masks, past_key_values, x_t, timestep):
            v_t = self._orig_denoise(prefix_pad_masks, past_key_values, x_t, timestep)
            t_scalar = float(timestep[0].item()) if hasattr(timestep, "item") else float(timestep)
            recorder.on_denoise_step(t_scalar, x_t, v_t)
            return v_t

        model.sample_actions = _sample
        model.denoise_step = _denoise

    def detach_fm_hooks(self) -> None:
        self.model.sample_actions = self._orig_sample
        self.model.denoise_step = self._orig_denoise

    # ---- captured-context slot -------------------------------------------------
    def _ctx_slot_tensors(self, ctx_arrays: dict, chunk_idx: int, device: Any):
        """Rebuild the B=1 ``embed_prefix`` inputs ``(images, img_masks, tokens, masks)`` for one
        recorded chunk from the DEFAULT π₀.₅ context schema (``ctx_images`` (ncam,B,3,H,W),
        ``ctx_img_masks`` (ncam,B), ``ctx_tokens`` (B,L), ``ctx_masks`` (B,L)). Mirrors
        ``resample.ChunkResampleSession._load_context_slot``: images float32 (a lossless superset of
        the bf16 compute dtype), masks/tokens in their captured dtypes — so the rebuilt prefix
        reproduces the recorded chunk (the same fidelity the ChunkResampleSession reproduce gate proves)."""
        imgs = np.asarray(ctx_arrays["ctx_images"][chunk_idx])       # (ncam, B, 3, H, W)
        imsk = np.asarray(ctx_arrays["ctx_img_masks"][chunk_idx])    # (ncam, B)
        img_list = [torch.as_tensor(imgs[i]).to(device, torch.float32) for i in range(imgs.shape[0])]
        mask_list = [torch.as_tensor(imsk[i]).to(device) for i in range(imsk.shape[0])]
        tokens = torch.as_tensor(np.asarray(ctx_arrays["ctx_tokens"][chunk_idx])).to(device)
        masks = torch.as_tensor(np.asarray(ctx_arrays["ctx_masks"][chunk_idx])).to(device)
        return img_list, mask_list, tokens, masks

    # ---- obs embedding (embedding-OOD detectors: rnd_oe/logpzo/density/fiper) -----
    @torch.no_grad()
    def embed_context_obs(self, ctx_arrays: dict, chunk_idx: int) -> "np.ndarray":
        """Pooled observation/prefix conditioning embedding at one recorded chunk's context.

        The conditioning vector the embedding-OOD detectors read as ``rec.obs_emb``: run π₀.₅'s
        ``embed_prefix`` (images + language tokens — the exact prefix ``sample_actions`` builds,
        ``modeling_pi05.py:807``) at this chunk's recorded context, then **masked-mean-pool** the
        per-token prefix embeddings over the valid (``pad_masks``) sequence → one ``(d,)`` vector
        (``d`` = action-expert width, 1024 for the gemma_300m expert). There is no
        separate ``state`` input (π₀.₅ folds it into the prompt tokens). B=1 recorded context, so no
        broadcast; the SAME routine drives ``fit`` (success rollouts) and ``score``, per the contract."""
        model = self.model
        device = next(model.parameters()).device
        img_list, mask_list, tokens, masks = self._ctx_slot_tensors(ctx_arrays, chunk_idx, device)
        embs, pad_masks, _ = model.embed_prefix(img_list, mask_list, tokens, masks)
        m = pad_masks[..., None].to(embs.dtype)                      # (B, seq, 1)
        pooled = (embs * m).sum(1) / m.sum(1).clamp_min(1.0)         # (B, d)
        return pooled[0].detach().cpu().float().numpy()             # (d,)

    # ---- action-expert last hidden (SAFE supervised probe) ------------------------
    @torch.no_grad()
    def embed_context_hidden(self, ctx_arrays: dict, chunk_idx: int, *,
                             horizon_reduce: str = "mean", diff_reduce: str = "mean") -> "np.ndarray":
        """Action-expert LAST-LAYER hidden feature at one recorded chunk's context → ``(d,)`` for SAFE.

        SAFE (arXiv:2506.09937, ``failure_prob/data/pizero.py``) reads the flow-matching action
        expert's last-layer hidden ``(n_diff_steps, n_pred_horizon, d)`` and reduces it to one vector
        by aggregating the HORIZON axis first, then the FLOW-STEP axis (PizeroDatasetConfig default =
        mean over both). For π₀.₅ that hidden is exactly the ``suffix_out`` fed to ``action_out_proj``
        at each ``denoise_step`` (``modeling_pi05.py:894-897``, sliced to the last ``chunk_size``
        tokens, cast float32). We re-run the recorded prefix + the
        full denoise loop (torch.compile off on this path) and capture ``suffix_out`` at every step via
        a forward-pre-hook on ``action_out_proj``, giving a ``(n_steps, chunk, d)`` stack, then reduce
        horizon→flow-step per SAFE. ``horizon_reduce`` / ``diff_reduce`` ∈ {"mean","first","last"}
        expose SAFE's grid-searched alternatives (default mean/mean). B=1, so no broadcast."""
        model = self.model
        device = next(model.parameters()).device
        img_list, mask_list, tokens, masks = self._ctx_slot_tensors(ctx_arrays, chunk_idx, device)

        # capture the input to action_out_proj (= suffix_out, the last-layer expert hidden) each step
        captured: list[torch.Tensor] = []

        def _pre_hook(_module, inputs):
            captured.append(inputs[0].detach().to(torch.float32))    # (B=1, chunk, d)

        handle = model.action_out_proj.register_forward_pre_hook(_pre_hook)
        try:
            # deterministic noise so the captured hidden is reproducible for this recorded context
            gen = torch.Generator(device=device).manual_seed(0)
            noise = torch.normal(0.0, 1.0, size=(1, self.config.chunk_size, self.config.max_action_dim),
                                 generator=gen, dtype=torch.float32, device=device)
            model.sample_actions(img_list, mask_list, tokens, masks, noise=noise)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("embed_context_hidden: action_out_proj hook never fired — is "
                               "torch.compile disabled on the sampling path?")
        h = torch.stack(captured, dim=0)[:, 0]                       # (n_steps, chunk, d)

        def _reduce(t: torch.Tensor, axis: int, how: str) -> torch.Tensor:
            if how == "mean":
                return t.mean(dim=axis)
            if how == "first":
                return t.index_select(axis, torch.tensor([0], device=t.device)).squeeze(axis)
            if how == "last":
                return t.index_select(axis, torch.tensor([t.shape[axis] - 1], device=t.device)).squeeze(axis)
            raise ValueError(f"reduce must be mean/first/last, got {how!r}")

        h = _reduce(h, 1, horizon_reduce)                            # (n_steps, d)  horizon reduced first
        e = _reduce(h, 0, diff_reduce)                               # (d,)          then flow-step
        return e.detach().cpu().float().numpy()

    # ---- fm_loss velocity (Diff-DAgger; model-in-the-loop) ------------------------
    @torch.no_grad()
    def velocity_fn_for_context(self, ctx_arrays: dict, chunk_idx: int):
        """Build the per-chunk velocity callable ``fm_loss`` injects as ``rec.extra['fm_velocity']``.

        Returns ``fn(x_t (M,chunk,act_pad), t (M,)) -> v (M,chunk,act_pad)`` closing over THIS
        chunk's recorded prefix. The prefix KV-cache is built ONCE at B=1 (mirrors
        ``resample._compute_prefix_cache``) and expanded to ``M`` per call via ``.expand`` views
        (``denoise_step`` deep-copies the cache internally, so the expansion materializes once);
        each call batches ``M`` re-noisings through one ``denoise_step``.

        VELOCITY CONTRACT (see ``detectors/fm_loss.py``): fm_loss uses the clean-data flow
        ``x_τ=(1−τ)·x0+τ·â``, ``target=â−x0``; π₀.₅'s native convention is ``x_s=s·noise+(1−s)·action``,
        ``u=noise−action`` (``modeling_pi05.py:734-735``) — the time+sign reverse. So the callable
        returns **``−v_θ(x_τ, s=1−τ)``**: negate the model velocity AND evaluate at FM time ``1−τ``.
        fm_loss feeds ``x_t`` padded to ``max_action_dim`` (it operates on full padded chunks via
        ``a_hat=x_t[-1]``); we zero-pad the executed window/dims to the FM-head shape and slice back."""
        import copy

        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        model = self.model
        device = next(model.parameters()).device
        img_list, mask_list, tokens, masks = self._ctx_slot_tensors(ctx_arrays, chunk_idx, device)

        # Build the prefix KV-cache ONCE at B=1, exactly as resample._compute_prefix_cache does.
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            img_list, mask_list, tokens, masks)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_4d = model._prepare_attention_masks_4d(prefix_att_2d)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_kv = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_4d, position_ids=prefix_pos, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True)

        def _expanded_prefix(M: int):
            # Shallow-copy the DynamicCache shell + each layer, swap in expanded K/V views (mirrors
            # resample._expanded_prefix); denoise_step deep-copies internally so this stays allocation-cheap.
            new_layers = []
            for layer in past_kv.layers:
                nl = copy.copy(layer)
                if getattr(layer, "is_initialized", True) and layer.keys.numel() > 0:
                    nl.keys = layer.keys.expand(M, -1, -1, -1)
                    nl.values = layer.values.expand(M, -1, -1, -1)
                new_layers.append(nl)
            npkv = type(past_kv)()
            npkv.layers = new_layers
            return prefix_pad_masks.expand(M, -1), npkv

        max_adim = int(self.config.max_action_dim)
        full_chunk = int(self.config.chunk_size)

        @torch.no_grad()
        def velocity(x_t: "np.ndarray", t: "np.ndarray") -> "np.ndarray":
            xt = torch.as_tensor(np.asarray(x_t), dtype=torch.float32, device=device)  # (M, win, act)
            M, win, act = xt.shape
            # denoise_step is hardwired to (chunk_size, max_action_dim); zero-pad both axes, run, slice back.
            pad_d = max_adim - act
            pad_c = full_chunk - win
            xpad = xt
            if pad_d > 0:
                xpad = torch.cat([xpad, xpad.new_zeros(M, win, pad_d)], dim=-1)
            if pad_c > 0:
                xpad = torch.cat([xpad, xpad.new_zeros(M, pad_c, max_adim)], dim=1)
            tau = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=device).reshape(M)
            s = 1.0 - tau                                            # π₀.₅ denoises FM time s: 1->0
            pad_b, pkv_b = _expanded_prefix(M)
            v = model.denoise_step(prefix_pad_masks=pad_b, past_key_values=pkv_b, x_t=xpad, timestep=s)
            return (-v[:, :win, :act]).detach().cpu().float().numpy()  # -v_θ(x_τ,1-τ), back to (M,win,act)
        return velocity
