from types import SimpleNamespace

import pytest

from benchmarks.rmbench.eval import run_local


def test_simulator_preflight_checks_cuda_planner_and_renderer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="RMBENCH_SIMULATOR_READY\n")

    monkeypatch.setattr(run_local.subprocess, "run", fake_run)
    rmbench_root = tmp_path / "rmbench"
    rmbench_root.mkdir()

    run_local.verify_simulator_runtime(
        "/sim/python",
        {"CUDA_VISIBLE_DEVICES": "1"},
        rmbench_root,
        tmp_path,
    )

    script = captured["command"][2]
    assert "torch.ones" in script
    assert "from envs.robot.planner import CuroboPlanner" in script
    assert "sapien.SapienRenderer()" in script
    assert captured["kwargs"]["cwd"] == rmbench_root
    assert (tmp_path / "simulator_preflight.log").read_text() == (
        "RMBENCH_SIMULATOR_READY\n"
    )


def test_simulator_preflight_reports_diagnostic_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        run_local.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="CUDA failed\n"),
    )
    rmbench_root = tmp_path / "rmbench"
    rmbench_root.mkdir()

    with pytest.raises(RuntimeError, match="simulator_preflight.log"):
        run_local.verify_simulator_runtime(
            "/sim/python",
            {"CUDA_VISIBLE_DEVICES": "1"},
            rmbench_root,
            tmp_path,
        )

    assert (tmp_path / "simulator_preflight.log").read_text() == "CUDA failed\n"


def test_deterministic_policy_sets_cublas_workspace() -> None:
    environment = {}

    run_local.configure_deterministic_cuda(environment, "1")

    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_nondeterministic_policy_does_not_set_cublas_workspace() -> None:
    environment = {}

    run_local.configure_deterministic_cuda(environment, "0")

    assert "CUBLAS_WORKSPACE_CONFIG" not in environment


def test_policy_server_settings_record_memory_action_alignment(tmp_path) -> None:
    profile = run_local.PROFILES["recurrent"]

    settings = run_local.policy_server_settings(
        profile, tmp_path / "checkpoint", "previous_kept_frame"
    )

    assert settings["memory_action_alignment"] == "previous_kept_frame"
    assert settings["adapter_module"] == profile["adapter"]
