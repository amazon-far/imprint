"""The online memory action stream can match the offline variant files.

The RMBench adapter buffers the first observation with a zero action, then pairs every new
observation with the action that was executed just before it
(benchmarks/rmbench/eval/runtime_overlay/adapter/robomme_tcp_policy.py). The offline variant
files pair kept frame f_j with the action executed at the previous kept frame f_{j-1}
(paper Eq. A.1, shared/variant_actions.py, action_causal_v*.npy).
"""
import importlib
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from mme_vla_suite.policies import policy as policy_module
from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
from mme_vla_suite.shared.variant_actions import load_variant_causal_actions

ADAPTER_DIR = Path(__file__).resolve().parents[4] / "benchmarks/rmbench/eval/runtime_overlay/adapter"
ACTION_DIM = 16
NEVER_FINALIZE = 1_000_000  # no Wan chunk is ever complete, so no encoder is called


class _UnusedFrontend:
    temporal_factor = NEVER_FINALIZE


def _zero_siglip(frames):  # (t, v, 224, 224, 3) -> (t, v, 64, 2048)
    return np.zeros((frames.shape[0], frames.shape[1], 64, 2048), np.float32)


def _buffer(monkeypatch, downsample=2, **kwargs):
    monkeypatch.setenv("EVAL_VISUAL_PROFILE", "siglip")
    return CosmoMemoryBuffer(
        cosmo=_UnusedFrontend(), siglip_fn=_zero_siglip, chunk_frames=NEVER_FINALIZE,
        downsample=downsample, num_views=1, state_dim=4, action_dim=ACTION_DIM,
        action_encoding="raw", proprio_encoding="raw", **kwargs,
    )


def _raw_actions(num_steps):
    """raw[t] is the action executed from raw step t; every row is unique."""
    return (np.arange(num_steps, dtype=np.float32)[:, None] + 1.0) * np.ones(
        (1, ACTION_DIM), np.float32
    )


def _memory_actions(buf):
    return np.stack(buf._actions)  # noqa: SLF001


def _feed_like_rmbench_adapter(buf, raw, chunk):
    num_steps = len(raw)
    images = np.zeros((num_steps, 1, 4, 4, 3), np.uint8)
    states = np.zeros((num_steps, 4), np.float32)
    buf.add_buffer(images[:1], states[:1], np.zeros((1, ACTION_DIM), np.float32), [0])
    step = 0
    while step < num_steps - 1:
        count = min(chunk, num_steps - 1 - step)
        new_steps = list(range(step + 1, step + count + 1))
        buf.add_buffer(images[new_steps], states[new_steps], raw[step:step + count], new_steps)
        step += count


@pytest.mark.parametrize("chunk", [50, 7, 1])
def test_previous_kept_frame_matches_variant_files(monkeypatch, tmp_path, chunk):
    raw = _raw_actions(149)
    buf = _buffer(monkeypatch, memory_action_alignment="previous_kept_frame")
    _feed_like_rmbench_adapter(buf, raw, chunk)

    frame_idx = np.arange(0, len(raw), 2)
    expected = load_variant_causal_actions(tmp_path, 0, frame_idx, raw)
    online = _memory_actions(buf)
    np.testing.assert_array_equal(online, expected)
    np.testing.assert_array_equal(buf.get_history_feats(len(raw) - 1)["cosmo_action_emb"], expected)


def test_default_alignment_keeps_previous_raw_step_rule(monkeypatch):
    raw = _raw_actions(101)
    buf = _buffer(monkeypatch)
    _feed_like_rmbench_adapter(buf, raw, 50)

    online = _memory_actions(buf)
    kept = np.arange(0, len(raw), 2)
    np.testing.assert_array_equal(online[0], np.zeros(ACTION_DIM, np.float32))
    np.testing.assert_array_equal(online[1:], raw[kept[1:] - 1])


def test_buffer_ignores_the_shell_environment(monkeypatch):
    # RoboCasa shares this class and copies the full shell environment into its server.
    monkeypatch.setenv("EVAL_MEMORY_ACTION_ALIGNMENT", "previous_kept_frame")
    raw = _raw_actions(21)
    buf = _buffer(monkeypatch)
    _feed_like_rmbench_adapter(buf, raw, 50)
    assert buf.memory_action_alignment == "causal_raw_previous"
    np.testing.assert_array_equal(_memory_actions(buf)[1:], raw[np.arange(2, 21, 2) - 1])


def test_alignments_agree_without_downsampling(monkeypatch):
    raw = _raw_actions(40)
    streams = []
    for alignment in ("causal_raw_previous", "previous_kept_frame"):
        buf = _buffer(monkeypatch, downsample=1, memory_action_alignment=alignment)
        _feed_like_rmbench_adapter(buf, raw, 50)
        streams.append(_memory_actions(buf))
    np.testing.assert_array_equal(streams[0], streams[1])


def test_previous_kept_frame_rejects_a_missing_raw_step(monkeypatch):
    buf = _buffer(monkeypatch, memory_action_alignment="previous_kept_frame")
    image = np.zeros((1, 1, 4, 4, 3), np.uint8)
    state = np.zeros((1, 4), np.float32)
    action = np.zeros((1, ACTION_DIM), np.float32)
    buf.add_buffer(image, state, action, [0])
    with pytest.raises(RuntimeError, match="never buffered"):
        buf.add_buffer(image, state, action, [2])


def test_unknown_alignment_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="unknown memory action alignment"):
        _buffer(monkeypatch, memory_action_alignment="stage2_selected_previous")


def test_policy_builds_each_buffer_with_its_alignment(monkeypatch):
    seen = []
    monkeypatch.setattr(
        policy_module,
        "_build_cosmo_buffer",
        lambda config, siglip_fn, memory_action_alignment: seen.append(memory_action_alignment),
    )
    policy = policy_module.MME_VLA_Policy.__new__(policy_module.MME_VLA_Policy)
    policy.config = SimpleNamespace(representation_type="cosmo_bottleneck")
    policy._vision_encode = None  # noqa: SLF001
    policy._prepare_mem_buffer()  # noqa: SLF001
    policy.memory_action_alignment = "previous_kept_frame"
    policy._prepare_mem_buffer()  # noqa: SLF001
    assert seen == ["causal_raw_previous", "previous_kept_frame"]


def _get_model(monkeypatch, tmp_path, **settings):
    """Run the RMBench adapter get_model with a stand-in rmbench_policy."""
    from mme_vla_suite.training import config as training_config

    # serve_robomme_tcp.py runs from the adapter folder and imports the adapter by name.
    monkeypatch.syspath_prepend(str(ADAPTER_DIR))
    robomme_tcp_policy = importlib.import_module("robomme_tcp_policy")
    for name in (
        "MEMFT_DETERMINISTIC_INFERENCE",
        "MEMFT_EXPECTED_MODEL_TYPE",
        "RMBENCH_TRACE_DIR",
        "RMBENCH_FULL_TRACE_DIR",
        "RMBENCH_DETAILED_PROFILE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMORY_LANGUAGE_SOURCE", "live")
    model = SimpleNamespace(model_type=SimpleNamespace(value="pi05"), pi05=True, use_history=True)
    monkeypatch.setattr(training_config, "get_config", lambda name: SimpleNamespace(model=model))
    fake_policy_module = ModuleType("rmbench_policy")
    fake_policy_module.create_policy = lambda *args, **kwargs: SimpleNamespace(
        memory_action_alignment="causal_raw_previous"
    )
    monkeypatch.setitem(sys.modules, "rmbench_policy", fake_policy_module)
    (tmp_path / "_CHECKPOINT_METADATA").write_text("{}")
    usr_args = {
        "checkpoint": str(tmp_path),
        "config_name": "imprint_rmbench_recurrent_memft",
        "history_config": "history.yaml",
        "prediction_horizon": 50,
        **settings,
    }
    return robomme_tcp_policy.get_model(usr_args)


def test_get_model_applies_and_records_memory_action_alignment(monkeypatch, tmp_path):
    server = _get_model(monkeypatch, tmp_path, memory_action_alignment="previous_kept_frame")

    assert server.policy.memory_action_alignment == "previous_kept_frame"
    assert server.get_runtime_provenance()["memory_action_alignment"] == "previous_kept_frame"


def test_get_model_defaults_to_released_alignment(monkeypatch, tmp_path):
    server = _get_model(monkeypatch, tmp_path)

    assert server.policy.memory_action_alignment == "causal_raw_previous"


def test_get_model_rejects_unknown_alignment(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="unknown memory_action_alignment"):
        _get_model(monkeypatch, tmp_path, memory_action_alignment="stage2_selected_previous")
