"""Opt-in inference optimizations for the Gated DeltaNet RMBench evaluator.

This module is imported only by ``robomme_gated_delta_optimized_tcp_policy``.
It patches the exact archived training source in that policy-server process;
the legacy evaluator and archived source remain unchanged.
"""

from __future__ import annotations

import os
import pathlib
import time
from collections.abc import Mapping, Sequence

import numpy as np


DELTA_UPDATE_BATCH_CHUNKS = 8
OPTIMIZATION_NAME = "batched_wan_fixed_delta_scan_v2"
_INSTALLED = False


class BatchedWanStreamingEncoder:
    """Wan causal streamer with camera views carried in the batch dimension."""

    def __init__(self, frontend, num_views: int = 1, alignment: str = "native"):
        from mme_vla_suite.shared.wan_alignment import resolve_wan_chunk_alignment

        self.fe = frontend
        self.alignment = resolve_wan_chunk_alignment(alignment)
        self.model = frontend._model
        self.device = frontend.device
        self.dtype = frontend.dtype
        self.latent_dim = frontend.latent_dim
        self.tf = frontend.temporal_factor
        self.num_views = int(num_views)
        if self.num_views <= 0:
            raise ValueError("num_views must be positive")
        self.reset()

    def reset(self) -> None:
        from mme_vla_suite.shared.wan_alignment import REPLICATE_FIRST_FRAME

        self._feat_map = None
        self._pending: list[np.ndarray] = []
        self._iter = 0
        self._replicate_first = self.alignment == REPLICATE_FIRST_FRAME
        self._needs_leading_copy = self._replicate_first
        try:
            self.model.clear_cache()
        except Exception:
            pass

    def _frames_to_tensor(self, frames_gv: np.ndarray):
        import torch

        # (group, view, height, width, channel) -> (view, channel, group, height, width)
        frames_gv = np.asarray(frames_gv)
        tensor = torch.from_numpy(np.ascontiguousarray(frames_gv)).to(
            self.device, self.dtype
        )
        tensor = tensor.div(255.0).mul_(2.0).sub_(1.0)
        return tensor.permute(1, 4, 0, 2, 3)

    def _encode_group(self, group_frames: np.ndarray) -> np.ndarray:
        import torch

        model = self.model
        inputs = self._frames_to_tensor(group_frames)
        if model.config.patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

            inputs = patchify(inputs, patch_size=model.config.patch_size)

        if self._feat_map is None:
            model.clear_cache()
            self._feat_map = model._enc_feat_map
        model._enc_feat_map = self._feat_map
        model._enc_conv_idx = [0]
        with torch.no_grad():
            encoded = model.encoder(
                inputs,
                feat_cache=model._enc_feat_map,
                feat_idx=model._enc_conv_idx,
            )
            self._feat_map = model._enc_feat_map
            encoded = model.quant_conv(encoded)[:, : self.latent_dim]

        # Keep the exact legacy token order: all spatial tokens for view 0,
        # followed by view 1, then view 2.
        views, channels, temporal, height, width = encoded.shape
        if views != self.num_views:
            raise RuntimeError(
                f"Wan returned {views} views, expected {self.num_views}"
            )
        latent = encoded.permute(0, 2, 3, 4, 1)[:, -1]
        return latent.reshape(
            self.num_views * height * width, channels
        ).float().cpu().numpy()

    def push(self, new_frames_v: np.ndarray) -> list[np.ndarray]:
        frames = np.asarray(new_frames_v)
        if (
            frames.ndim != 5
            or frames.shape[1] != self.num_views
            or frames.shape[-1] != 3
        ):
            raise ValueError(
                "expected Wan frames (time, views, height, width, 3), got "
                f"{frames.shape}"
            )
        self._pending.extend(np.ascontiguousarray(frame) for frame in frames)
        if self._needs_leading_copy and self._pending:
            # replicate_first_frame: the one-frame leading Wan group is a copy of frame 0,
            # so each emitted chunk holds four kept frames. Its latent is dropped below.
            self._pending.insert(0, np.array(self._pending[0], copy=True))
            self._needs_leading_copy = False

        chunks = []
        while True:
            needed = 1 if self._iter == 0 else self.tf
            if len(self._pending) < needed:
                break
            group = np.stack(self._pending[:needed], axis=0)
            del self._pending[:needed]
            leading_copy = self._replicate_first and self._iter == 0
            latent = self._encode_group(group)
            self._iter += 1
            if not leading_copy:
                chunks.append(latent)
        return chunks


def pack_delta_updates(
    updates: Sequence[Mapping[str, np.ndarray]],
    *,
    batch_chunks: int = DELTA_UPDATE_BATCH_CHUNKS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack finalized chunks into one fixed-shape update batch."""

    if not updates:
        raise ValueError("at least one DeltaNet update is required")
    if len(updates) > batch_chunks:
        raise ValueError(
            f"received {len(updates)} updates for batch size {batch_chunks}"
        )

    keys = ("cosmo_chunk", "action_chunk", "proprio_chunk")
    first = {
        key: np.asarray(updates[0][key], dtype=np.float32) for key in keys
    }
    packed = {
        key: np.zeros((batch_chunks,) + value.shape, dtype=np.float32)
        for key, value in first.items()
    }
    for index, update in enumerate(updates):
        for key in keys:
            value = np.asarray(update[key], dtype=np.float32)
            if value.shape != first[key].shape:
                raise ValueError(
                    f"inconsistent {key} shape {value.shape}; "
                    f"expected {first[key].shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains non-finite values")
            packed[key][index] = value

    valid = np.arange(batch_chunks, dtype=np.int32) < len(updates)
    return (
        packed["cosmo_chunk"][None, ...],
        packed["action_chunk"][None, ...],
        packed["proprio_chunk"][None, ...],
        valid,
    )


def _promote_delta_state(state):
    """Normalize the recurrent cache dtype before crossing the JIT boundary."""

    import jax.numpy as jnp

    matrices, conv_caches, chunk_position = state
    return (
        matrices,
        conv_caches.astype(jnp.float32),
        chunk_position,
    )


def _update_delta_memory_state_sequence(
    self,
    state,
    cosmo_chunks,
    action_chunks,
    proprio_chunks,
    valid_chunks,
):
    """Apply a fixed number of recurrent updates with exact no-op padding."""

    import jax
    import jax.numpy as jnp

    if cosmo_chunks.ndim != 4:
        raise ValueError("cosmo_chunks must have shape (batch, chunks, tokens, dim)")
    if action_chunks.ndim != 4 or proprio_chunks.ndim != 4:
        raise ValueError(
            "action/proprio chunks must have shape (batch, chunks, frames, dim)"
        )
    chunk_count = cosmo_chunks.shape[1]
    if (
        action_chunks.shape[1] != chunk_count
        or proprio_chunks.shape[1] != chunk_count
        or valid_chunks.shape != (chunk_count,)
    ):
        raise ValueError("DeltaNet update batch has inconsistent chunk dimensions")

    # The archived incremental implementation initializes its convolution
    # cache in bf16, then promotes it to the event dtype (float32 here) on the
    # first update. A Python loop permits that one-time carry dtype change;
    # lax.scan does not. Promoting the all-zero initial cache up front is
    # numerically identical to the legacy first update.
    state = _promote_delta_state(state)
    scan_inputs = (
        jnp.swapaxes(cosmo_chunks, 0, 1),
        jnp.swapaxes(action_chunks, 0, 1),
        jnp.swapaxes(proprio_chunks, 0, 1),
        valid_chunks.astype(jnp.bool_),
    )

    def update_one(current_state, values):
        cosmo_chunk, action_chunk, proprio_chunk, valid = values

        def apply_update(operands):
            current, cosmo, action, proprio = operands
            return self.update_delta_memory_state(
                current,
                cosmo,
                action,
                proprio,
            )

        def keep_state(operands):
            return operands[0]

        next_state = jax.lax.cond(
            valid,
            apply_update,
            keep_state,
            (current_state, cosmo_chunk, action_chunk, proprio_chunk),
        )
        return next_state, None

    next_state, _ = jax.lax.scan(
        update_one,
        state,
        scan_inputs,
    )
    return next_state


def _optimized_policy_add_buffer(self, obs: dict) -> None:
    if not getattr(self, "_uses_gated_delta", False):
        return self._gated_delta_optimization_original_add_buffer(obs)
    if self.mem_buffer is None:
        return

    import jax.numpy as jnp

    images = obs["images"]
    states = obs["state"]
    if obs.get("exec_start_idx", 0) > 0:
        self.exec_start_idx = obs["exec_start_idx"]

    step_idx_list = list(
        range(self.step_idx + 1, self.step_idx + len(images) + 1)
    )
    action_dim = int(
        getattr(
            self.mem_buffer,
            "action_dim",
            int(self.config.cosmo_bottleneck_memory.action_in_dim) // 2,
        )
    )
    actions = np.asarray(
        obs.get(
            "actions",
            np.zeros((len(images), action_dim), dtype=np.float32),
        )
    )
    self.mem_buffer.add_buffer(
        images,
        states,
        actions,
        step_idx_list,
        images_siglip=obs.get("images_siglip"),
    )
    if self._delta_state is None or self._delta_update_many is None:
        raise RuntimeError("optimized gated delta memory state was not initialized")

    updates = self.mem_buffer.drain_delta_updates()
    if updates:
        # Always invoke the jitted batched update with the steady-state cache
        # dtype. Otherwise the initial bf16 cache and subsequent float32 cache
        # create two expensive XLA executables.
        self._delta_state = _promote_delta_state(self._delta_state)
    for start in range(0, len(updates), self._delta_update_batch_chunks):
        update_batch = updates[start : start + self._delta_update_batch_chunks]
        cosmo, action, proprio, valid = pack_delta_updates(
            update_batch,
            batch_chunks=self._delta_update_batch_chunks,
        )
        self._delta_state = self._delta_update_many(
            self._delta_state,
            jnp.asarray(cosmo, dtype=jnp.float32),
            jnp.asarray(action, dtype=jnp.float32),
            jnp.asarray(proprio, dtype=jnp.float32),
            jnp.asarray(valid, dtype=jnp.bool_),
        )
        self._delta_update_many_calls += 1
        self._delta_update_many_chunks += len(update_batch)
    self.step_idx += len(images)


def install_optimized_runtime() -> dict[str, object]:
    """Install process-local patches before the checkpoint policy is created."""

    global _INSTALLED
    if _INSTALLED:
        return {
            "name": OPTIMIZATION_NAME,
            "delta_update_batch_chunks": DELTA_UPDATE_BATCH_CHUNKS,
            "installed": True,
        }

    from openpi.shared import nnx_utils
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
    from mme_vla_suite.models.representation.wan_frontend import WanFrontend
    from mme_vla_suite.policies.policy import MME_VLA_Policy

    if not hasattr(WanFrontend, "_gated_delta_optimization_original_make_streamer"):
        WanFrontend._gated_delta_optimization_original_make_streamer = (
            WanFrontend.make_streamer
        )

        def make_batched_streamer(self, num_views: int = 1, alignment: str = "native"):
            return BatchedWanStreamingEncoder(
                self, num_views=num_views, alignment=alignment
            )

        WanFrontend.make_streamer = make_batched_streamer

    if not hasattr(
        HistoryPi0,
        "update_delta_memory_state_sequence",
    ):
        HistoryPi0.update_delta_memory_state_sequence = (
            _update_delta_memory_state_sequence
        )

    if not hasattr(
        MME_VLA_Policy,
        "_gated_delta_optimization_original_init",
    ):
        original_init = MME_VLA_Policy.__init__
        MME_VLA_Policy._gated_delta_optimization_original_init = original_init

        def optimized_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._delta_update_batch_chunks = DELTA_UPDATE_BATCH_CHUNKS
            self._delta_update_many_calls = 0
            self._delta_update_many_chunks = 0
            self._delta_update_many = (
                nnx_utils.module_jit(
                    self._model.update_delta_memory_state_sequence
                )
                if self._uses_gated_delta
                else None
            )

        MME_VLA_Policy.__init__ = optimized_init

    if not hasattr(
        MME_VLA_Policy,
        "_gated_delta_optimization_original_add_buffer",
    ):
        MME_VLA_Policy._gated_delta_optimization_original_add_buffer = (
            MME_VLA_Policy.add_buffer
        )
        MME_VLA_Policy.add_buffer = _optimized_policy_add_buffer

    _INSTALLED = True
    return {
        "name": OPTIMIZATION_NAME,
        "delta_update_batch_chunks": DELTA_UPDATE_BATCH_CHUNKS,
        "installed": True,
    }


def optimized_warmup_runtime(server) -> dict[str, object]:
    """Warm every optimized inference path and restore a clean episode state."""

    if server._runtime_warmup_result is not None:
        return dict(server._runtime_warmup_result)

    import jax
    import jax.numpy as jnp

    policy = server.policy
    if not getattr(policy, "_uses_gated_delta", False):
        raise RuntimeError("optimized gated-delta adapter loaded a non-gated policy")
    if policy._delta_update_many is None:
        raise RuntimeError("fixed-shape DeltaNet update function is unavailable")

    result: dict[str, object] = {
        "optimization": {
            "name": OPTIMIZATION_NAME,
            "wan_streamer": "camera_views_as_batch",
            "delta_update": "fixed_eight_chunk_recurrent_scan",
            "delta_update_batch_chunks": DELTA_UPDATE_BATCH_CHUNKS,
        }
    }
    memory_buffer = policy.mem_buffer
    frontend = memory_buffer.cosmo
    wan_height = int(getattr(memory_buffer, "wan_h", 0)) or 256
    wan_width = int(getattr(memory_buffer, "wan_w", 0)) or 256
    verify_optimization = bool(
        getattr(server, "_verify_gated_delta_optimization", False)
    )
    wan_frames = np.zeros(
        (
            1 + int(frontend.temporal_factor),
            int(memory_buffer.num_views),
            wan_height,
            wan_width,
            3,
        ),
        dtype=np.uint8,
    )
    if verify_optimization:
        wan_frames = np.random.default_rng(20260907).integers(
            0,
            256,
            size=wan_frames.shape,
            dtype=np.uint8,
        )
        legacy_factory = getattr(
            frontend,
            "_gated_delta_optimization_original_make_streamer",
            None,
        )
        if legacy_factory is None:
            raise RuntimeError("legacy Wan streamer factory is unavailable")
        legacy_streamer = legacy_factory(
            memory_buffer.num_views,
            alignment=getattr(memory_buffer, "wan_chunk_alignment", "native"),
        )
        started = time.perf_counter()
        legacy_chunks = legacy_streamer.push(wan_frames)
        legacy_streamer.reset()
        result["wan_legacy_reference"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "emitted_chunks": len(legacy_chunks),
            "chunk_shapes": [list(chunk.shape) for chunk in legacy_chunks],
        }
    else:
        legacy_chunks = None

    streamer = frontend.make_streamer(
        memory_buffer.num_views,
        alignment=getattr(memory_buffer, "wan_chunk_alignment", "native"),
    )
    if not isinstance(streamer, BatchedWanStreamingEncoder):
        raise RuntimeError("optimized Wan streamer patch was not applied")
    started = time.perf_counter()
    wan_chunks = streamer.push(wan_frames)
    streamer.reset()
    result["wan_streaming"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "input_shape": list(wan_frames.shape),
        "emitted_chunks": len(wan_chunks),
        "chunk_shapes": [list(chunk.shape) for chunk in wan_chunks],
    }
    if legacy_chunks is not None:
        if len(legacy_chunks) != len(wan_chunks):
            raise RuntimeError(
                "batched Wan emitted a different number of chunks: "
                f"legacy={len(legacy_chunks)} optimized={len(wan_chunks)}"
            )
        max_abs = 0.0
        mean_abs = 0.0
        cosine = 1.0
        value_count = 0
        dot = 0.0
        legacy_norm = 0.0
        optimized_norm = 0.0
        for legacy_chunk, optimized_chunk in zip(
            legacy_chunks, wan_chunks, strict=True
        ):
            if legacy_chunk.shape != optimized_chunk.shape:
                raise RuntimeError(
                    "batched Wan chunk shape differs from legacy: "
                    f"{legacy_chunk.shape} != {optimized_chunk.shape}"
                )
            legacy64 = np.asarray(legacy_chunk, dtype=np.float64)
            optimized64 = np.asarray(optimized_chunk, dtype=np.float64)
            difference = np.abs(legacy64 - optimized64)
            max_abs = max(max_abs, float(difference.max(initial=0.0)))
            mean_abs += float(difference.sum())
            value_count += difference.size
            dot += float(np.vdot(legacy64.ravel(), optimized64.ravel()))
            legacy_norm += float(np.vdot(legacy64.ravel(), legacy64.ravel()))
            optimized_norm += float(
                np.vdot(optimized64.ravel(), optimized64.ravel())
            )
        if value_count:
            mean_abs /= value_count
        denominator = np.sqrt(legacy_norm * optimized_norm)
        if denominator:
            cosine = dot / denominator
        result["wan_equivalence"] = {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "cosine": cosine,
            "atol": 0.05,
            "rtol": 0.05,
        }
        if cosine < 0.999 or any(
            not np.allclose(legacy, optimized, atol=0.05, rtol=0.05)
            for legacy, optimized in zip(
                legacy_chunks, wan_chunks, strict=True
            )
        ):
            raise RuntimeError(
                "batched Wan failed legacy equivalence: "
                f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
                f"cosine={cosine:.9f}"
            )

    started = time.perf_counter()
    visual = memory_buffer.warmup_visual_encoder()
    result["visual"] = {
        **visual,
        "elapsed_seconds": time.perf_counter() - started,
    }

    cbm = policy.config.cosmo_bottleneck_memory
    encoder = policy._model.mem_encoder
    state = policy._model.init_delta_memory_state(batch_size=1)
    delta_rng = np.random.default_rng(20260908)
    array_factory = (
        lambda shape: delta_rng.normal(0.0, 0.1, size=shape).astype(np.float32)
        if verify_optimization
        else np.zeros(shape, dtype=np.float32)
    )
    cosmo_chunks = jnp.asarray(
        array_factory(
            (
                1,
                DELTA_UPDATE_BATCH_CHUNKS,
                int(encoder.tokens_per_chunk),
                int(cbm.cosmos_latent_dim),
            )
        )
    )
    action_chunks = jnp.asarray(
        array_factory(
            (
                1,
                DELTA_UPDATE_BATCH_CHUNKS,
                int(encoder.chunk_frames),
                int(cbm.action_in_dim),
            )
        )
    )
    proprio_chunks = jnp.asarray(
        array_factory(
            (
                1,
                DELTA_UPDATE_BATCH_CHUNKS,
                int(encoder.chunk_frames),
                int(cbm.proprio_in_dim),
            )
        )
    )
    valid_count = 3 if verify_optimization else DELTA_UPDATE_BATCH_CHUNKS
    valid_chunks = jnp.arange(DELTA_UPDATE_BATCH_CHUNKS) < valid_count
    if verify_optimization:
        sequential_state = state
        started = time.perf_counter()
        for index in range(valid_count):
            sequential_state = policy._delta_update(
                sequential_state,
                cosmo_chunks[:, index],
                action_chunks[:, index],
                proprio_chunks[:, index],
            )
        jax.tree.map(jax.block_until_ready, sequential_state)
        result["incremental_update_legacy_reference"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "valid_chunks": valid_count,
        }
    else:
        sequential_state = None
    started = time.perf_counter()
    optimized_state = _promote_delta_state(state)
    updated_state = policy._delta_update_many(
        optimized_state,
        cosmo_chunks,
        action_chunks,
        proprio_chunks,
        valid_chunks,
    )
    jax.tree.map(jax.block_until_ready, updated_state)
    result["incremental_update"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "cosmo_shape": list(cosmo_chunks.shape),
        "action_shape": list(action_chunks.shape),
        "proprio_shape": list(proprio_chunks.shape),
        "valid_shape": list(valid_chunks.shape),
    }
    if sequential_state is not None:
        sequential_leaves = jax.tree.leaves(jax.device_get(sequential_state))
        optimized_leaves = jax.tree.leaves(jax.device_get(updated_state))
        if len(sequential_leaves) != len(optimized_leaves):
            raise RuntimeError("DeltaNet state trees have different leaf counts")
        leaf_differences = [
            np.max(
                np.abs(
                    np.asarray(sequential, dtype=np.float64)
                    - np.asarray(optimized, dtype=np.float64)
                ),
                initial=0.0,
            )
            for sequential, optimized in zip(
                sequential_leaves, optimized_leaves, strict=True
            )
        ]
        max_abs = max((float(value) for value in leaf_differences), default=0.0)
        result["delta_equivalence"] = {
            "max_abs": max_abs,
            "atol": 1e-5,
            "rtol": 1e-5,
            "valid_chunks": valid_count,
        }
        if any(
            not np.allclose(sequential, optimized, atol=1e-5, rtol=1e-5)
            for sequential, optimized in zip(
                sequential_leaves, optimized_leaves, strict=True
            )
        ):
            raise RuntimeError(
                "fixed-shape DeltaNet scan failed sequential equivalence: "
                f"max_abs={max_abs:.6g}"
            )

    from mme_vla_suite.shared.memory_caps import (
        max_recent_tokens,
        max_step_tokens,
    )

    recent_len = max_recent_tokens(cbm)
    step_len = max_step_tokens(cbm)
    recent = jnp.zeros(
        (1, recent_len, int(cbm.siglip_dim)),
        dtype=jnp.float32,
    )
    action = jnp.zeros(
        (1, step_len, int(cbm.action_in_dim)),
        dtype=jnp.float32,
    )
    proprio = jnp.zeros(
        (1, step_len, int(cbm.proprio_in_dim)),
        dtype=jnp.float32,
    )
    language = jnp.zeros(
        (1, int(cbm.lang_emb_dim)),
        dtype=jnp.float32,
    )
    recent_mask = jnp.zeros((1, recent_len), dtype=jnp.bool_)
    action_mask = jnp.zeros((1, step_len), dtype=jnp.bool_)
    proprio_mask = jnp.zeros((1, step_len), dtype=jnp.bool_)
    started = time.perf_counter()
    memory_tokens = policy._delta_read(
        updated_state,
        recent,
        action,
        proprio,
        language,
        recent_mask,
        action_mask,
        proprio_mask,
    )
    jax.block_until_ready(memory_tokens)
    result["incremental_read"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "recent_shape": list(recent.shape),
        "action_shape": list(action.shape),
        "proprio_shape": list(proprio.shape),
        "memory_token_shape": list(memory_tokens.shape),
    }

    warmup_prompt = os.environ.get(
        "MEMFT_RUNTIME_WARMUP_PROMPT",
        "cover the blocks with the correct covers",
    )
    zero_images = np.zeros(
        (1, int(memory_buffer.num_views), 256, 256, 3),
        dtype=np.uint8,
    )
    zero_state = np.zeros((1, 16), dtype=np.float32)
    zero_actions = np.zeros(
        (1, int(memory_buffer.action_dim)),
        dtype=np.float32,
    )
    try:
        policy.reset(warmup_prompt)
        policy.add_buffer(
            {
                "images": zero_images,
                "images_siglip": zero_images,
                "state": zero_state,
                "actions": zero_actions,
            }
        )
        # Match the evaluator's real startup sequence. A synthetic direct call
        # can still miss a JAX cache key used by the state created through
        # Policy.reset(), so compile two consolidated 50-frame updates here
        # before any measured simulation begins.
        consolidated_images = np.repeat(zero_images, 50, axis=0)
        consolidated_state = np.repeat(zero_state, 50, axis=0)
        consolidated_actions = np.repeat(zero_actions, 50, axis=0)
        buffer_update_warmup = []
        for _ in range(2):
            started = time.perf_counter()
            policy.add_buffer(
                {
                    "images": consolidated_images,
                    "images_siglip": consolidated_images,
                    "state": consolidated_state,
                    "actions": consolidated_actions,
                }
            )
            jax.tree.map(jax.block_until_ready, policy._delta_state)
            buffer_update_warmup.append(time.perf_counter() - started)
        result["runtime_buffer_update_warmup"] = {
            "elapsed_seconds": sum(buffer_update_warmup),
            "calls": len(buffer_update_warmup),
            "per_call_seconds": buffer_update_warmup,
            "frames_per_call": 50,
        }
        started = time.perf_counter()
        inference = policy.infer(
            {
                "observation/image": zero_images[0, 0],
                "observation/wrist_image": zero_images[0, 1],
                "observation/right_image": zero_images[0, 2],
                "observation/state": zero_state[0],
                "prompt": warmup_prompt,
            }
        )
        actions = np.asarray(inference["actions"])
        result["action_inference"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "action_shape": list(actions.shape),
        }
    finally:
        policy.reset()

    # Warmup uses the same profiled methods as evaluation. Remove those rows so
    # benchmark files contain only real trials.
    profile_paths = [
        getattr(server, "_detailed_profile_path", None),
        pathlib.Path(os.environ["MME_TIMING_LOG"])
        if os.environ.get("MME_TIMING_LOG")
        else None,
    ]
    removed = []
    for path in profile_paths:
        if path is not None and path.exists():
            path.unlink()
            removed.append(str(path))
    server._profile_call_index = 0
    result["profile_cleanup"] = {"removed": removed}
    result["total_elapsed_seconds"] = sum(
        float(value.get("elapsed_seconds", 0.0))
        for value in result.values()
        if isinstance(value, dict)
    )
    server._runtime_warmup_result = result
    return dict(result)
