"""Raw-TCP RMBench adapter for the direct Robomme policy.

This module is loaded in two processes by RMBench's official
``policy_model_server.py`` transport:

* the model-server process creates ``RoboMMETCPServerModel`` via ``get_model``;
* the simulator process imports ``eval`` and executes returned action chunks.

Keeping the simulator-side action loop here preserves the official evaluator
while avoiding a protocol mismatch between its raw TCP client and WebSocket.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from collections.abc import Mapping, Sequence

import numpy as np


CAMERA_ORDER = ("head_camera", "left_camera", "right_camera")
MEMORY_CAMERA_ORDER = CAMERA_ORDER
MEMORY_CAMERA_INDICES = tuple(CAMERA_ORDER.index(name) for name in MEMORY_CAMERA_ORDER)
POLICY_INPUT_COLOR_ORDER = os.environ.get(
    "RMBENCH_POLICY_INPUT_COLOR_ORDER", "rgb"
).strip().lower()
if POLICY_INPUT_COLOR_ORDER not in ("rgb", "bgr"):
    raise ValueError(
        "RMBENCH_POLICY_INPUT_COLOR_ORDER must be rgb or bgr, got "
        f"{POLICY_INPUT_COLOR_ORDER!r}"
    )
ACTION_WIDTH = 16
NATIVE_ACTION_WIDTH = 14
PREDICTION_HORIZON = 50
IMAGE_SIZE = 256
MEMORY_LANGUAGE_SOURCES = ("live", "stored")


def memory_language_prompt_key(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def load_memory_language_lookup() -> tuple[str, str | None, dict[str, np.ndarray]]:
    source = os.environ.get("MEMORY_LANGUAGE_SOURCE", "live").strip().lower()
    if source not in MEMORY_LANGUAGE_SOURCES:
        raise ValueError(
            f"MEMORY_LANGUAGE_SOURCE must be one of {MEMORY_LANGUAGE_SOURCES}, got {source!r}"
        )
    if source == "live":
        return source, None, {}

    path_value = os.environ.get("MEMORY_LANGUAGE_LOOKUP_PATH")
    if not path_value:
        raise ValueError("stored memory language requires MEMORY_LANGUAGE_LOOKUP_PATH")
    path = pathlib.Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"memory language lookup is missing: {path}")
    with np.load(path, allow_pickle=False) as lookup:
        prompts = [str(prompt) for prompt in lookup["prompts"].tolist()]
        embeddings = np.asarray(lookup["embeddings"], dtype=np.float32)
        lookup_kind = str(lookup["lookup_kind"].item())
    if not prompts or len(prompts) != len(set(prompts)):
        raise ValueError("memory language lookup must have unique non-empty prompts")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(prompts):
        raise ValueError(
            "memory language lookup embeddings must have one row per prompt"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("memory language lookup contains non-finite values")
    return source, lookup_kind, {
        prompt: np.ascontiguousarray(embedding) for prompt, embedding in zip(prompts, embeddings)
    }


def model_to_native(model_action: np.ndarray) -> np.ndarray:
    model_action = np.asarray(model_action, dtype=np.float32)
    if model_action.shape != (ACTION_WIDTH,):
        raise ValueError(
            f"expected one {ACTION_WIDTH}-D model action, got {model_action.shape}"
        )
    return np.concatenate(
        (
            model_action[:6],
            model_action[14:15],
            model_action[7:13],
            model_action[15:16],
        )
    )


def native_to_model(native_state: np.ndarray) -> np.ndarray:
    native_state = np.asarray(native_state, dtype=np.float32)
    if native_state.shape != (NATIVE_ACTION_WIDTH,):
        raise ValueError(
            f"expected one {NATIVE_ACTION_WIDTH}-D native state, got "
            f"{native_state.shape}"
        )
    state = np.zeros((ACTION_WIDTH,), dtype=np.float32)
    state[:6] = native_state[:6]
    state[7:13] = native_state[7:13]
    state[14] = native_state[6]
    state[15] = native_state[13]
    return state


def resize_with_pad(
    image: np.ndarray,
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> np.ndarray:
    import cv2

    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected uint8 HWC image, got {image.shape} {image.dtype}")

    source_height, source_width = image.shape[:2]
    ratio = max(source_width / width, source_height / height)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    output = np.zeros((height, width, 3), dtype=np.uint8)
    pad_top, remainder_height = divmod(height - resized_height, 2)
    pad_left, remainder_width = divmod(width - resized_width, 2)
    pad_bottom = pad_top + remainder_height
    pad_right = pad_left + remainder_width
    output[pad_top : height - pad_bottom, pad_left : width - pad_right] = resized
    return output


def extract_observation(observation: Mapping) -> tuple[np.ndarray, np.ndarray]:
    cameras = observation["observation"]
    frames = []
    for name in CAMERA_ORDER:
        image = np.asarray(cameras[name]["rgb"])
        image = resize_with_pad(image)
        if POLICY_INPUT_COLOR_ORDER == "bgr":
            image = image[..., ::-1]
        frames.append(np.ascontiguousarray(image))
    state = native_to_model(observation["joint_action"]["vector"])
    return np.stack(frames, axis=0), state


def configure_deterministic_inference() -> None:
    if os.environ.get("MEMFT_DETERMINISTIC_INFERENCE", "0") != "1":
        return

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


class RoboMMETCPServerModel:
    """Stateful direct Robomme policy exposed through RMBench's TCP server."""

    def __init__(self, policy, *, provenance: Mapping[str, object]):
        self.policy = policy
        self._provenance = dict(provenance)
        (
            self._memory_language_source,
            self._memory_language_lookup_kind,
            self._memory_language_lookup,
        ) = load_memory_language_lookup()
        self._memory_language_lookup_by_key: dict[str, np.ndarray] = {}
        for prompt, embedding in self._memory_language_lookup.items():
            key = memory_language_prompt_key(prompt)
            if key in self._memory_language_lookup_by_key:
                raise ValueError(
                    "stored memory language lookup has colliding canonical prompts"
                )
            self._memory_language_lookup_by_key[key] = embedding
        self._provenance["memory_language_source"] = self._memory_language_source
        self._provenance["memory_language_lookup_kind"] = self._memory_language_lookup_kind
        lookup_path = os.environ.get("MEMORY_LANGUAGE_LOOKUP_PATH")
        self._provenance["memory_language_lookup_sha256"] = (
            hashlib.sha256(pathlib.Path(lookup_path).read_bytes()).hexdigest()
            if lookup_path and self._memory_language_source == "stored"
            else None
        )
        self._episode_active = False
        self._prompt = None
        trace_dir = os.environ.get("RMBENCH_TRACE_DIR")
        self._trace_dir = pathlib.Path(trace_dir) if trace_dir else None
        self._trace_written = False
        self._trace_history_written = False
        full_trace_dir = os.environ.get("RMBENCH_FULL_TRACE_DIR")
        self._full_trace_dir = (
            pathlib.Path(full_trace_dir) if full_trace_dir else None
        )
        self._trace_call_index = 0
        self._pending_trace_call_dir = None
        detailed_profile_path = os.environ.get("RMBENCH_DETAILED_PROFILE_PATH")
        self._detailed_profile_path = (
            pathlib.Path(detailed_profile_path) if detailed_profile_path else None
        )
        self._profile_call_index = 0
        self._runtime_warmup_result = None
        if self._detailed_profile_path is not None:
            self._install_policy_profilers()
        if self._trace_dir is not None:
            self._install_history_trace()

    def _install_history_trace(self) -> None:
        original_prepare_history = self.policy._prepare_history

        def traced_prepare_history(inputs):
            result = original_prepare_history(inputs)
            if not self._trace_history_written:
                arrays = {
                    key: np.asarray(value)
                    for key, value in result.items()
                    if key.startswith("cosmo_") and value is not None
                }
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                np.savez(self._trace_dir / "first_prepared_history.npz", **arrays)
                self._trace_history_written = True
            return result

        self.policy._prepare_history = traced_prepare_history

    def _record_detail(
        self,
        phase: str,
        elapsed_seconds: float,
        *,
        call_index: int | None = None,
        **metadata,
    ) -> None:
        if self._detailed_profile_path is None:
            return
        row = {
            "phase": phase,
            "elapsed_ms": elapsed_seconds * 1_000.0,
            "call_index": call_index,
            **metadata,
        }
        self._detailed_profile_path.parent.mkdir(parents=True, exist_ok=True)
        with self._detailed_profile_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")

    def _install_policy_profilers(self) -> None:
        import jax

        original_prepare_history = self.policy._prepare_history
        original_input_transform = self.policy._input_transform
        original_sample_actions = self.policy._sample_actions
        original_output_transform = self.policy._output_transform

        def timed_prepare_history(inputs):
            started = time.perf_counter()
            result = original_prepare_history(inputs)
            self._record_detail(
                "prepare_history",
                time.perf_counter() - started,
                call_index=self._profile_call_index,
            )
            return result

        def timed_input_transform(inputs):
            started = time.perf_counter()
            result = original_input_transform(inputs)
            self._record_detail(
                "input_transform",
                time.perf_counter() - started,
                call_index=self._profile_call_index,
            )
            return result

        def timed_sample_actions(*args, **kwargs):
            started = time.perf_counter()
            result = original_sample_actions(*args, **kwargs)
            jax.block_until_ready(result)
            self._record_detail(
                "sample_actions",
                time.perf_counter() - started,
                call_index=self._profile_call_index,
            )
            return result

        def timed_output_transform(outputs):
            started = time.perf_counter()
            result = original_output_transform(outputs)
            self._record_detail(
                "output_transform",
                time.perf_counter() - started,
                call_index=self._profile_call_index,
            )
            return result

        self.policy._prepare_history = timed_prepare_history
        self.policy._input_transform = timed_input_transform
        self.policy._sample_actions = timed_sample_actions
        self.policy._output_transform = timed_output_transform

    def get_runtime_provenance(self) -> dict[str, object]:
        """Return the checkpoint identity loaded by this policy server."""
        return dict(self._provenance)

    def warmup_runtime(self) -> dict[str, object]:
        """Compile stateful memory kernels before the simulator RPC deadline."""
        if self._runtime_warmup_result is not None:
            return dict(self._runtime_warmup_result)

        import jax
        import jax.numpy as jnp

        result: dict[str, object] = {}
        memory_buffer = getattr(self.policy, "mem_buffer", None)
        if memory_buffer is not None and hasattr(
            memory_buffer, "warmup_visual_encoder"
        ):
            started = time.perf_counter()
            visual = memory_buffer.warmup_visual_encoder()
            result["visual"] = {
                **visual,
                "elapsed_seconds": time.perf_counter() - started,
            }

        if getattr(self.policy, "_uses_gated_delta", False):
            from mme_vla_suite.shared.memory_caps import (
                max_recent_tokens,
                max_step_tokens,
            )

            cbm = self.policy.config.cosmo_bottleneck_memory
            encoder = self.policy._model.mem_encoder
            state = self.policy._model.init_delta_memory_state(batch_size=1)
            cosmo_chunk = jnp.zeros(
                (
                    1,
                    int(encoder.tokens_per_chunk),
                    int(cbm.cosmos_latent_dim),
                ),
                dtype=jnp.float32,
            )
            action_chunk = jnp.zeros(
                (1, int(encoder.chunk_frames), int(cbm.action_in_dim)),
                dtype=jnp.float32,
            )
            proprio_chunk = jnp.zeros(
                (1, int(encoder.chunk_frames), int(cbm.proprio_in_dim)),
                dtype=jnp.float32,
            )

            started = time.perf_counter()
            updated_state = self.policy._delta_update(
                state,
                cosmo_chunk,
                action_chunk,
                proprio_chunk,
            )
            jax.tree.map(jax.block_until_ready, updated_state)
            result["incremental_update"] = {
                "elapsed_seconds": time.perf_counter() - started,
                "cosmo_shape": list(cosmo_chunk.shape),
                "action_shape": list(action_chunk.shape),
                "proprio_shape": list(proprio_chunk.shape),
            }

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
            memory_tokens = self.policy._delta_read(
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

        result["total_elapsed_seconds"] = sum(
            float(value.get("elapsed_seconds", 0.0))
            for value in result.values()
            if isinstance(value, dict)
        )
        self._runtime_warmup_result = result
        return dict(result)

    def reset_model(self) -> None:
        self.policy.reset()
        self._episode_active = False
        self._prompt = None

    def _reset_policy_for_episode(self, prompt: str) -> None:
        self.policy.reset(prompt)

    @staticmethod
    def _buffer_payload(
        frames: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
    ) -> dict:
        frames = np.asarray(frames)
        if frames.ndim != 5 or frames.shape[1] != len(CAMERA_ORDER):
            raise ValueError(
                "memory frames must have shape (T,3,H,W,3), got "
                f"{frames.shape}"
            )
        memory_frames = np.ascontiguousarray(
            frames[:, MEMORY_CAMERA_INDICES, ...]
        )
        return {
            "images": memory_frames,
            "images_siglip": memory_frames,
            "state": state,
            "actions": actions,
        }

    def _start_episode(self, prompt: str, observation: Mapping) -> None:
        call_index = self._profile_call_index
        started = time.perf_counter()
        frames, state = extract_observation(observation)
        self._record_detail(
            "start_extract_observation",
            time.perf_counter() - started,
            call_index=call_index,
        )
        started = time.perf_counter()
        self._reset_policy_for_episode(prompt)
        self._record_detail(
            "start_policy_reset",
            time.perf_counter() - started,
            call_index=call_index,
        )
        if self._memory_language_source == "stored":
            memory_lang_emb = self._memory_language_lookup.get(prompt)
            if memory_lang_emb is None:
                memory_lang_emb = self._memory_language_lookup_by_key.get(
                    memory_language_prompt_key(prompt)
                )
            if memory_lang_emb is None:
                raise KeyError(
                    f"no stored memory language embedding for evaluator prompt {prompt!r}"
                )
            memory_buffer = getattr(self.policy, "mem_buffer", None)
            if memory_buffer is None or not hasattr(memory_buffer, "set_language"):
                raise RuntimeError(
                    "stored memory language requires a cosmo memory buffer with set_language"
                )
            memory_buffer.set_language(memory_lang_emb)
        started = time.perf_counter()
        self.policy.add_buffer(
            self._buffer_payload(
                frames[None],
                state[None],
                np.zeros((1, ACTION_WIDTH), dtype=np.float32),
            )
        )
        self._record_detail(
            "start_add_buffer",
            time.perf_counter() - started,
            call_index=call_index,
        )
        self._episode_active = True
        self._prompt = prompt

    def _infer(self, observation: Mapping) -> np.ndarray:
        call_index = self._profile_call_index
        started = time.perf_counter()
        frames, state = extract_observation(observation)
        self._record_detail(
            "extract_observation",
            time.perf_counter() - started,
            call_index=call_index,
        )
        trace_call_dir = None
        if self._full_trace_dir is not None:
            trace_call_dir = (
                self._full_trace_dir / f"call_{self._trace_call_index:03d}"
            )
            trace_call_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                trace_call_dir / "input.npz",
                frames=frames,
                state=state,
                prompt=np.asarray(self._prompt),
            )
        started = time.perf_counter()
        result = self.policy.infer(
            {
                "observation/image": frames[0],
                "observation/wrist_image": frames[1],
                "observation/right_image": frames[2],
                "observation/state": state,
                "prompt": self._prompt,
            }
        )
        self._record_detail(
            "policy_infer_total",
            time.perf_counter() - started,
            call_index=call_index,
        )
        started = time.perf_counter()
        actions = np.asarray(result["actions"], dtype=np.float32)
        expected_shape = (PREDICTION_HORIZON, ACTION_WIDTH)
        if actions.shape != expected_shape:
            raise ValueError(
                f"expected predicted actions {expected_shape}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy returned non-finite actions")
        self._record_detail(
            "action_validation",
            time.perf_counter() - started,
            call_index=call_index,
        )
        if trace_call_dir is not None:
            np.save(trace_call_dir / "actions.npy", actions)
            self._pending_trace_call_dir = trace_call_dir
            self._trace_call_index += 1
        if self._trace_dir is not None and not self._trace_written:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                self._trace_dir / "first_policy_input.npz",
                frames=frames,
                state=state,
            )
            np.save(self._trace_dir / "first_actions.npy", actions)
            self._trace_written = True
        self._profile_call_index += 1
        return actions

    def eval(self, payload: Mapping) -> np.ndarray:
        prompt = payload["prompt"]
        observation = payload["observation"]
        if not self._episode_active:
            self._start_episode(prompt, observation)
        elif prompt != self._prompt:
            raise ValueError("episode prompt changed without reset_model")
        return self._infer(observation)

    def add_buffer(self, payload: Mapping) -> dict:
        total_started = time.perf_counter()
        observations = payload["observations"]
        actions = np.asarray(payload["actions"], dtype=np.float32)
        if not isinstance(observations, Sequence) or not observations:
            raise ValueError("add_buffer requires one or more observations")
        if actions.shape != (len(observations), ACTION_WIDTH):
            raise ValueError(
                "expected actions with shape "
                f"({len(observations)}, {ACTION_WIDTH}), got {actions.shape}"
            )
        if not self._episode_active:
            raise RuntimeError("add_buffer called before episode start")

        started = time.perf_counter()
        extracted = [extract_observation(observation) for observation in observations]
        self._record_detail(
            "buffer_extract_observations",
            time.perf_counter() - started,
            call_index=self._profile_call_index - 1,
            frame_count=len(observations),
        )
        frames = np.stack([sample[0] for sample in extracted])
        state = np.stack([sample[1] for sample in extracted])
        if self._pending_trace_call_dir is not None:
            np.savez(
                self._pending_trace_call_dir / "buffer.npz",
                frames=frames,
                state=state,
                actions=actions,
            )
            self._pending_trace_call_dir = None
        started = time.perf_counter()
        self.policy.add_buffer(self._buffer_payload(frames, state, actions))
        self._record_detail(
            "buffer_policy_add",
            time.perf_counter() - started,
            call_index=self._profile_call_index - 1,
            frame_count=len(observations),
        )
        self._record_detail(
            "buffer_total",
            time.perf_counter() - total_started,
            call_index=self._profile_call_index - 1,
            frame_count=len(observations),
        )
        return {"buffered": len(observations)}


def get_model(usr_args):
    configure_deterministic_inference()

    from rmbench_policy import create_policy
    from mme_vla_suite.shared.cosmo_buffer import MEMORY_ACTION_ALIGNMENTS
    from mme_vla_suite.training import config as training_config

    required = ("checkpoint", "config_name", "history_config", "prediction_horizon")
    missing = [key for key in required if key not in usr_args]
    if missing:
        raise KeyError(f"missing direct Robomme server settings: {missing}")
    checkpoint = pathlib.Path(usr_args["checkpoint"]).resolve()
    metadata = checkpoint / "_CHECKPOINT_METADATA"
    if not metadata.is_file():
        raise FileNotFoundError(f"checkpoint metadata is missing: {metadata}")
    resolved_config = training_config.get_config(str(usr_args["config_name"]))
    model_type = resolved_config.model.model_type.value
    pi05 = bool(getattr(resolved_config.model, "pi05", False))
    use_history = bool(getattr(resolved_config.model, "use_history", False))
    history_config = usr_args["history_config"]
    if use_history != (history_config is not None):
        raise ValueError(
            "evaluation history setting does not match the training config: "
            f"use_history={use_history} history_config={history_config!r}"
        )
    expected_model_type = os.environ.get("MEMFT_EXPECTED_MODEL_TYPE", "").strip()
    if expected_model_type and model_type != expected_model_type:
        raise ValueError(
            f"expected model type {expected_model_type!r}, resolved {model_type!r}"
        )
    if expected_model_type == "pi0" and pi05:
        raise ValueError("native PI0 evaluation resolved a pi05=True model")
    memory_action_alignment = str(
        usr_args.get("memory_action_alignment", "causal_raw_previous")
    )
    if memory_action_alignment not in MEMORY_ACTION_ALIGNMENTS:
        raise ValueError(
            f"unknown memory_action_alignment {memory_action_alignment!r}; "
            f"expected one of {MEMORY_ACTION_ALIGNMENTS}"
        )
    policy = create_policy(
        checkpoint,
        config_name=usr_args["config_name"],
        history_config=history_config,
        prediction_horizon=int(usr_args["prediction_horizon"]),
        seed=int(usr_args.get("seed", 42)),
    )
    # policy.reset() builds the memory buffer of each episode with this mode.
    policy.memory_action_alignment = memory_action_alignment
    return RoboMMETCPServerModel(
        policy,
        provenance={
            "checkpoint": str(checkpoint),
            "checkpoint_metadata_sha256": hashlib.sha256(
                metadata.read_bytes()
            ).hexdigest(),
            "config_name": str(usr_args["config_name"]),
            "history_config": history_config,
            "prediction_horizon": int(usr_args["prediction_horizon"]),
            "model_type": model_type,
            "pi05": pi05,
            "use_history": use_history,
            "memory_action_alignment": memory_action_alignment,
        },
    )


def eval(task_env, model, observation) -> None:
    profile_path_setting = os.environ.get("RMBENCH_CLIENT_PROFILE_PATH")
    profile_path = pathlib.Path(profile_path_setting) if profile_path_setting else None
    chunk_started = time.perf_counter()
    started = time.perf_counter()
    actions = np.asarray(
        model.call(
            func_name="eval",
            obs={
                "prompt": task_env.get_instruction(),
                "observation": observation,
            },
        ),
        dtype=np.float32,
    )
    eval_rpc_seconds = time.perf_counter() - started
    expected_shape = (PREDICTION_HORIZON, ACTION_WIDTH)
    if actions.shape != expected_shape:
        raise ValueError(
            f"expected predicted actions {expected_shape}, got {actions.shape}"
        )

    inter_chunk_delay_seconds = float(
        os.environ.get("RMBENCH_INTER_CHUNK_DELAY_SECONDS", "0")
    )
    if inter_chunk_delay_seconds < 0:
        raise ValueError("RMBENCH_INTER_CHUNK_DELAY_SECONDS must be non-negative")
    if inter_chunk_delay_seconds:
        time.sleep(inter_chunk_delay_seconds)

    buffered_observations = []
    buffered_actions = []
    take_action_seconds = 0.0
    get_obs_seconds = 0.0
    for model_action in actions:
        if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
            break
        started = time.perf_counter()
        task_env.take_action(model_to_native(model_action))
        take_action_seconds += time.perf_counter() - started
        started = time.perf_counter()
        buffered_observations.append(task_env.get_obs())
        get_obs_seconds += time.perf_counter() - started
        buffered_actions.append(model_action)

    add_buffer_rpc_seconds = 0.0
    if buffered_observations:
        started = time.perf_counter()
        response = model.call(
            func_name="add_buffer",
            obs={
                "observations": buffered_observations,
                "actions": np.stack(buffered_actions),
            },
        )
        add_buffer_rpc_seconds = time.perf_counter() - started
        if response.get("buffered") != len(buffered_observations):
            raise RuntimeError(f"policy server did not buffer the action chunk: {response}")
    if profile_path is not None:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "phase": "action_chunk",
            "eval_rpc_ms": eval_rpc_seconds * 1_000.0,
            "take_action_ms": take_action_seconds * 1_000.0,
            "get_obs_ms": get_obs_seconds * 1_000.0,
            "add_buffer_rpc_ms": add_buffer_rpc_seconds * 1_000.0,
            "total_ms": (time.perf_counter() - chunk_started) * 1_000.0,
            "executed_actions": len(buffered_actions),
        }
        with profile_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def reset_model(model) -> None:
    """Compatibility helper for direct callers outside eval_policy_client.py."""
    model.call(func_name="reset_model")
