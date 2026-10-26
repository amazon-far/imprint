from collections.abc import Sequence
import time
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
from mme_vla_suite.shared.mem_buffer import MemoryBuffer, MemoryBufferRecurrent
from mme_vla_suite.shared.memory_caps import max_recent_tokens, max_step_tokens


def _uses_gated_delta_memory(config) -> bool:
    if config is None or getattr(config, "representation_type", None) != "cosmo_bottleneck":
        return False
    cb = getattr(config, "cosmo_bottleneck_memory", None)
    return str(getattr(cb, "memory_encoder_type", "perceiver")) == "gated_delta_hybrid"


def _load_cosmo_frontend(variant: str):
    if variant.lower().startswith("wan"):
        from mme_vla_suite.models.representation.wan_frontend import WanFrontend
        return WanFrontend(variant=variant, device="cuda")
    from mme_vla_suite.models.representation.cosmo_frontend import CosmoFrontend
    return CosmoFrontend(variant=variant, device="cuda")


def _build_cosmo_buffer(config, siglip_fn):
    from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
    from mme_vla_suite.shared.wan_alignment import wan_chunk_alignment_from_config
    cb = config.cosmo_bottleneck_memory
    action_encoding = getattr(cb, "action_encoding", "raw_delta")
    proprio_encoding = getattr(cb, "proprio_encoding", "raw_delta")
    # Raw stored action dim (pre-encoding, per-step): the buffer stores RAW actions and
    # the encoding is applied in get_history_feats. raw_delta packs action+Delta so
    # action_in_dim=2*raw=24 -> raw=12; RAW (no delta) has action_in_dim=raw=12 -> raw=12
    # (NO //2, else 6-dim actions feed a 12-dim tokenizer); geometric -> raw 12.
    if action_encoding == "raw":
        raw_action_dim = int(cb.action_in_dim)          # raw: action_in_dim already = raw width (12)
    elif action_encoding == "geometric":
        raw_action_dim = 12
    else:
        raw_action_dim = int(cb.action_in_dim) // 2     # raw_delta: action_in_dim = 2*raw
    return CosmoMemoryBuffer(
        cosmo=_load_cosmo_frontend(cb.variant), siglip_fn=siglip_fn,
        chunk_frames=int(cb.temporal_factor), downsample=int(cb.downsample),
        num_views=config.num_views, state_dim=config.memory_feature.state.input_dim,
        action_dim=raw_action_dim, latent_dim=int(cb.cosmos_latent_dim),
        action_encoding=action_encoding, proprio_encoding=proprio_encoding,
        pool_hw=(int(getattr(cb, "pool_hw", 0)) or None),
        # grid-agnostic pooling for NON-square grids (RoboDojo 12x16). Absent -> robocasa 8x8 path.
        pool_h=(int(getattr(cb, "pool_h", 0)) or None),
        pool_w=(int(getattr(cb, "pool_w", 0)) or None),
        wan_h=int(getattr(cb, "wan_h", 0)), wan_w=int(getattr(cb, "wan_w", 0)),
        # Absent -> "native" (the released stores and checkpoints). See shared/wan_alignment.py.
        wan_chunk_alignment=wan_chunk_alignment_from_config(cb))


class MME_VLA_Policy:
    def __init__(
        self,
        model: HistoryPi0,
        *,
        seed: int = 42,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantiles: bool = False,
    ):
        self._model = model
        self._seed = seed
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._vision_encode = nnx_utils.module_jit(model.vision_encode)


        self.config = model.history_config
        self._uses_gated_delta = _uses_gated_delta_memory(self.config)
        self._delta_update = (
            nnx_utils.module_jit(model.update_delta_memory_state)
            if self._uses_gated_delta
            else None
        )
        self._delta_read = (
            nnx_utils.module_jit(model.read_delta_memory_state)
            if self._uses_gated_delta
            else None
        )
        self.mem_buffer = None

        self.state_norm_stats = norm_stats['state']
        self.use_quantiles = use_quantiles

        self.reset()


    def _prepare_mem_buffer(self):
        if self.config is None or self.config.representation_type == "symbolic":
            self.mem_buffer = None
        elif self.config.representation_type == "recurrent":
            self.mem_buffer = MemoryBufferRecurrent(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                input_obs_horizon=self.config.streaming_obs_horizon,
                max_recur_steps=self.config.recurrent_memory.max_recur_steps,
                max_video_steps=self.config.recurrent_memory.max_pretraj_steps,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )
        elif self.config.representation_type == "cosmo_bottleneck":
            self.mem_buffer = _build_cosmo_buffer(self.config, siglip_fn=self._vision_encode)
        else:
            self.mem_buffer = MemoryBuffer(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                compute_token_drop_score = self.config.perceptual_memory.type == "token_dropping",
                token_drop_stride=self.config.streaming_obs_horizon // 2,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
                use_time=bool(getattr(self.config, "memory_use_time", False)),
            )

    @override
    def infer(self, obs: dict) -> dict:
        if self.config is not None and self.config.representation_type not in ("symbolic", "cosmo_bottleneck"):
            assert len(self.mem_buffer._history_feats) > 0, \
                "history feats is empty, add buffer first"
        elif self.config is not None and self.config.representation_type == "cosmo_bottleneck":
            assert self.step_idx >= 0, "add_buffer must be called before infer (cosmo buffer)"

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._prepare_history(inputs)
        inputs = self._input_transform(inputs)
        if self._uses_gated_delta:
            if self._delta_state is None or self._delta_read is None:
                raise RuntimeError("gated delta memory state was not initialized")

            def _batched(name, dtype=None):
                value = inputs[name]
                array = jnp.asarray(value, dtype=dtype)
                return array[jnp.newaxis, ...]

            language = inputs.get("cosmo_lang_emb", None)
            if language is not None:
                language = jnp.asarray(language, dtype=jnp.float32)[jnp.newaxis, ...]
            mem_tokens = self._delta_read(
                self._delta_state,
                _batched("cosmo_recent_emb", jnp.float32),
                _batched("cosmo_action_emb", jnp.float32),
                _batched("cosmo_proprio_emb", jnp.float32),
                language,
                _batched("cosmo_recent_mask", jnp.bool_),
                _batched("cosmo_action_mask", jnp.bool_),
                _batched("cosmo_proprio_mask", jnp.bool_),
            )
            # Inject after the standard transform pipeline. Existing repack transforms do
            # not need to know about this inference-only bypass.
            inputs["cosmo_mem_tokens"] = np.asarray(
                jax.device_get(mem_tokens[0]), dtype=np.float32
            )
        observation = HistAugObservation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        )
        self._rng, sample_rng = jax.random.split(self._rng)

        start_time = time.monotonic()
        outputs = {
            "state": observation.state,
            "actions": self._sample_actions(sample_rng, observation, **self._sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000

        return outputs

    @override
    def reset(
        self,
        prompt: str | None = None,
        lang_emb: np.ndarray | None = None,
    ) -> None:
        del self.mem_buffer
        self._prepare_mem_buffer()
        # Prefer the archived per-episode vector used by Stage-2. Recomputing it after
        # memory fine-tuning can use different language-model weights from the store build.
        if self.config is not None and self.config.representation_type == "cosmo_bottleneck":
            if lang_emb is None:
                lang_emb = self._pool_prompt_text_emb(prompt)
            else:
                lang_emb = np.asarray(lang_emb, dtype=np.float32).reshape(-1)
                expected_dim = int(
                    getattr(
                        self.config.cosmo_bottleneck_memory,
                        "lang_emb_dim",
                        2048,
                    )
                )
                if lang_emb.shape != (expected_dim,):
                    raise ValueError(
                        "lang_emb must have shape "
                        f"({expected_dim},), got {lang_emb.shape}"
                    )
                if not np.all(np.isfinite(lang_emb)):
                    raise ValueError("lang_emb must contain only finite values")
            self.mem_buffer.set_language(lang_emb)
        self._delta_state = (
            self._model.init_delta_memory_state(batch_size=1)
            if self._uses_gated_delta
            else None
        )
        self.step_idx = -1
        self.exec_start_idx = 0
        self._rng = jax.random.key(self._seed)


    def add_buffer(self, obs: dict) -> None:
        if self.mem_buffer is None:
            return
        images = obs["images"]
        states = obs["state"]
        if obs.get("exec_start_idx", 0) > 0: # has video
            self.exec_start_idx = obs["exec_start_idx"]

        step_idx_list = list(range(self.step_idx+1, self.step_idx + len(images) + 1))
        # time add-on: per-frame elapsed fraction. Online we don't know the final episode
        # length, so we normalize by a fixed horizon (matches the eval cap; client may pass
        # "time_horizon"). elapsed = step_idx / horizon, clipped to [0,1].
        elapsed_list = None
        if getattr(self.mem_buffer, "use_time", False):
            H = float(obs.get("time_horizon", 2048))
            elapsed_list = [min(si / max(H - 1.0, 1.0), 1.0) for si in step_idx_list]
        if self.config is not None and self.config.representation_type == "cosmo_bottleneck":
            action_dim = int(getattr(self.mem_buffer, "action_dim", int(self.config.cosmo_bottleneck_memory.action_in_dim) // 2))
            actions = np.asarray(
                obs.get("actions", np.zeros((len(images), action_dim), np.float32)))
            images_siglip = obs.get("images_siglip", None)
            self.mem_buffer.add_buffer(images, states, actions, step_idx_list, images_siglip=images_siglip)
            if self._uses_gated_delta:
                if self._delta_state is None or self._delta_update is None:
                    raise RuntimeError("gated delta memory state was not initialized")
                for update in self.mem_buffer.drain_delta_updates():
                    self._delta_state = self._delta_update(
                        self._delta_state,
                        jnp.asarray(update["cosmo_chunk"], dtype=jnp.float32)[None, ...],
                        jnp.asarray(update["action_chunk"], dtype=jnp.float32)[None, ...],
                        jnp.asarray(update["proprio_chunk"], dtype=jnp.float32)[None, ...],
                    )
        else:
            self.mem_buffer.add_buffer(images, states, step_idx_list, elapsed_list=elapsed_list)
        self.step_idx += len(images)

    def _pool_prompt_text_emb(self, prompt):
        # FIX(bug4): pool the PaliGemma text embedding for language-conditioned memory FiLM,
        # via the SAME canonical helper the store-build / Stage-1 / Stage-2 paths use, so the
        # bottleneck sees the identical language vector it was trained with. prompt=None ->
        # fall back to the model default task prompt if available, else FiLM-identity (None).
        if prompt is None:
            prompt = getattr(self, "_default_prompt", None) or (self._metadata or {}).get("prompt")
        if prompt is None:
            return None
        from mme_vla_suite.shared.data_utils import pool_prompt_text_emb
        return pool_prompt_text_emb(self._model, prompt)

    def _normalize_state_generic(self, x):
        # TODO(train): proprio norm stats for the 32-dim cosmo proprio stream (state+delta) do not
        # exist yet (sub-project A); pass through raw float32 -- MLP tokenizer learns from raw values.
        return np.asarray(x, np.float32)

    def _normalize_state(self, state):
        if self.use_quantiles:
            return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
        else:
            return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _prepare_history(self, inputs: dict) -> dict:
        if self.config is None or self.config.representation_type == "symbolic":
            return inputs

        if self.config.representation_type == "recurrent":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask = \
                self.mem_buffer.prepare_token_recurrent(
                    self.step_idx, self.exec_start_idx, history_feats_gather_fn)
            inputs["recur_image_emb"] = recur_image_emb
            inputs["recur_pos_emb"] = recur_pos_emb
            inputs["recur_state_emb"] = self._normalize_state(recur_state_emb)
            inputs["recur_mask"] = recur_mask
        elif self.config.representation_type == "perceptual":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            token_budget = self.config.budget

            if self.config.perceptual_memory.type == "token_dropping":
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_token_dropping(
                        self.step_idx, token_budget, history_feats_gather_fn)
            else:
                token_per_image = self.config.token_per_image
                fs_out = self.mem_buffer.prepare_frame_sampling(
                    self.step_idx, token_budget, token_per_image, history_feats_gather_fn)
                if getattr(self.mem_buffer, "use_time", False):
                    (static_image_emb, static_pos_emb, static_state_emb,
                     static_mask, static_time) = fs_out
                else:
                    (static_image_emb, static_pos_emb, static_state_emb, static_mask) = fs_out
                    static_time = None
            if self.config.perceptual_memory.type != "token_dropping" \
                    and getattr(self.mem_buffer, "use_time", False) and static_time is not None:
                inputs["static_time"] = np.asarray(static_time, dtype=np.float32)

            inputs["static_image_emb"] = static_image_emb
            inputs["static_pos_emb"] = static_pos_emb
            inputs["static_state_emb"] = self._normalize_state(static_state_emb)
            inputs["static_mask"] = static_mask
        elif self.config.representation_type == "cosmo_bottleneck":
            feats = self.mem_buffer.get_history_feats(self.step_idx)
            # FIX(eval speed): the buffer returns VARIABLE-length streams (latent/recent/action/
            # proprio grow every step). sample_actions is jitted -> variable shapes force XLA to
            # RECOMPILE every step (~7.5s/step). Training padded to FIXED caps + masks; mirror
            # that here so shapes are static -> compile ONCE (~47ms/step). Caps identical to
            # dataset._gather_cosmo_bottleneck.
            cbm = self.config.cosmo_bottleneck_memory
            # Use the same fixed capacities as ``_gather_cosmo_bottleneck``.
            MAX_RECENT = max_recent_tokens(cbm)
            MAX_STEPS = max_step_tokens(cbm)
            LATENT_CAP = int(getattr(cbm, "latent_cap", 0) or 2048)
            sdim = int(getattr(cbm, "siglip_dim", 2048))
            _ad = int(getattr(cbm, "action_in_dim", 12))
            _pd = int(getattr(cbm, "proprio_in_dim", 16))
            def _fixed_pad(arr, L, D):
                out = np.zeros((L, D), np.float32); m = np.zeros((L,), bool)
                a = np.asarray(arr, np.float32)
                n = min(a.shape[0], L)
                if n > 0:
                    out[:n] = a[-L:][:n]; m[:n] = True   # keep MOST-RECENT L (matches training tail-keep)
                return out, m

            if self._uses_gated_delta:
                # Finalized chunks have already been applied exactly once to the persistent
                # recurrent state. Keep only the provisional/recent trajectory here.
                finalized_steps = int(feats["cosmo_finalized_steps"])
                action_source = feats["cosmo_action_emb"][finalized_steps:]
                proprio_source = feats["cosmo_proprio_emb"][finalized_steps:]
                latent_dim = int(getattr(cbm, "cosmos_latent_dim", 48))
                lat_p = np.zeros((1, latent_dim), np.float32)
                lat_m = np.zeros((1,), bool)
            else:
                action_source = feats["cosmo_action_emb"]
                proprio_source = feats["cosmo_proprio_emb"]
                lat = np.asarray(feats["cosmo_latent_emb"], np.float32)
                lat_p = np.zeros((LATENT_CAP, lat.shape[-1] if lat.size else 48), np.float32)
                lat_m = np.zeros((LATENT_CAP,), bool)
                _lc = min(lat.shape[0], LATENT_CAP)
                if _lc > 0:
                    lat_p[:_lc] = lat[-LATENT_CAP:][:_lc]; lat_m[:_lc] = True
            rec_p, rec_m = _fixed_pad(feats["cosmo_recent_emb"], MAX_RECENT, sdim)
            act_p, act_m = _fixed_pad(action_source, MAX_STEPS, _ad)
            pro_p, pro_m = _fixed_pad(
                self._normalize_state_generic(proprio_source), MAX_STEPS, _pd
            )
            inputs["cosmo_latent_emb"] = lat_p
            inputs["cosmo_mem_mask"] = lat_m
            inputs["cosmo_recent_emb"] = rec_p
            inputs["cosmo_recent_mask"] = rec_m
            inputs["cosmo_action_emb"] = act_p
            inputs["cosmo_action_mask"] = act_m
            inputs["cosmo_proprio_emb"] = pro_p
            inputs["cosmo_proprio_mask"] = pro_m
            if feats["cosmo_lang_emb"] is not None:
                inputs["cosmo_lang_emb"] = feats["cosmo_lang_emb"]
        else:
            raise ValueError(f"Not supported representation type: {self.config.representation_type}")


        return inputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
