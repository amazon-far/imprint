#!/usr/bin/env python3
"""Run one RoboCasa task group with isolated policy and simulator environments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import time

from postprocess_results import write_paper_metrics

TASK_GROUPS = (
    "retrieve_fruit",
    "retrieve_oil",
    "wash_and_return",
    "wash_and_return_same_location",
    "sweeten_hot_chocolate",
    "pan_transfer",
    "rinse_cutting_board",
    "organize_condiments",
    "pick_place_counter_to_sink",
    "place_equal_ice_cubes",
    "scrub_cutting_board",
    "frying_pan_adjustment",
)

PROFILES = {
    "full_attention": {
        "config": "imprint_robocasa_full_attention_memft",
        "history": ("cosmo-wan-bottleneck-modul-robocasa-combined600-dinov3-memft.yaml"),
        "deterministic": "1",
    },
    "recurrent": {
        "config": "imprint_robocasa_recurrent_memft",
        "history": ("cosmo-wan-gated-delta-robocasa-combined600-dinov3-memft-optimized.yaml"),
        "deterministic": "0",
    },
}

REPLAY_ASSET_LAYOUT = {
    "wash-scene-root": Path("wash"),
    "retrieve-lineage-manifest": Path(
        "retrieve/retrieve_combined350_lineage.json"
    ),
    "retrieve-source-root": Path("retrieve/sources"),
    "sweeten-recovered-root": Path("sweeten"),
    "pan-transfer-dataset": Path("pan_transfer/lerobot"),
    "rinse-cutting-board-dataset": Path("rinse_cutting_board/lerobot"),
    "new5-scene-root": Path("robocasa365"),
}


def environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robocasa-root",
        type=Path,
        required=True,
        help="Extracted customized RoboCasa source root.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dinov3-model",
        type=Path,
        required=True,
        help="Local Hugging Face DINOv3 model directory used by the memory stream.",
    )
    parser.add_argument(
        "--dinov3-runtime-overlay",
        type=Path,
        help="Optional Python package overlay that provides DINOv3 support.",
    )
    parser.add_argument("--architecture", choices=tuple(PROFILES), required=True)
    parser.add_argument("--task-group", choices=TASK_GROUPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-trials", type=int, default=1)
    parser.add_argument("--policy-python", default=str(root / ".venv/bin/python"))
    parser.add_argument(
        "--sim-python",
        help="Python from the customized RoboCasa environment; defaults to <root>/.venv-mem/bin/python.",
    )
    parser.add_argument("--policy-gpu", default="0")
    parser.add_argument("--sim-gpu", default="1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument(
        "--replay-assets-root",
        type=Path,
        default=environment_path("ROBOCASA_REPLAY_ASSETS"),
        help=(
            "Extracted IMPRINT RoboCasa replay bundle. Task-specific paths "
            "are derived from this root and may be overridden below."
        ),
    )
    parser.add_argument(
        "--memory-data-root",
        type=Path,
        default=environment_path("ROBOCASA_MEMORY_DATA_ROOT"),
    )
    parser.add_argument(
        "--memory-store",
        type=Path,
        default=environment_path("ROBOCASA_MEMORY_STORE"),
    )
    parser.add_argument(
        "--memory-replay-helper",
        type=Path,
        help=(
            "Recorded-scene replay helper; defaults to "
            "<robocasa-root>/scripts/replay/replay_memory_demos.py."
        ),
    )
    parser.add_argument(
        "--wash-scene-root",
        type=Path,
        default=environment_path("ROBOCASA_WASH_SCENE_ROOT"),
    )
    parser.add_argument(
        "--retrieve-lineage-manifest",
        type=Path,
        default=environment_path("ROBOCASA_RETRIEVE_LINEAGE_MANIFEST"),
    )
    parser.add_argument(
        "--retrieve-source-root",
        type=Path,
        default=environment_path("ROBOCASA_RETRIEVE_SOURCE_ROOT"),
    )
    parser.add_argument(
        "--sweeten-recovered-root",
        type=Path,
        default=environment_path("ROBOCASA_SWEETEN_RECOVERED_ROOT"),
    )
    parser.add_argument(
        "--pan-transfer-dataset",
        type=Path,
        default=environment_path("ROBOCASA_PAN_TRANSFER_DATASET"),
    )
    parser.add_argument(
        "--rinse-cutting-board-dataset",
        type=Path,
        default=environment_path("ROBOCASA_RINSE_CUTTING_BOARD_DATASET"),
    )
    parser.add_argument(
        "--new5-scene-root",
        type=Path,
        default=environment_path("ROBOCASA_NEW5_SCENE_ROOT"),
    )
    return parser.parse_args()


def wait_for_port(
    host: str,
    port: int,
    process: subprocess.Popen,
    timeout: int,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited with status {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"policy server did not listen on {host}:{port}")


def resolved(path: Path | None) -> Path | None:
    return None if path is None else path.expanduser().resolve()


def executable_path(path: str | Path) -> Path:
    """Make an interpreter path absolute without resolving its virtualenv symlink."""
    return Path(os.path.abspath(Path(path).expanduser()))


def simulator_pythonpath_entries(root: Path) -> tuple[Path, ...]:
    """Find source roots in either the release archive or its legacy layout."""
    candidates = [root, root / "src"]
    for base in (root, root / "src"):
        candidates.extend(
            base / name
            for name in (
                "openpi_rc",
                "robocasa",
                "robosuite-master",
                "robomimic",
            )
        )
    entries = []
    for path in candidates:
        if path.exists() and path not in entries:
            entries.append(path)
    return tuple(entries)


def replay_asset_paths(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    return {
        name: root / relative
        for name, relative in REPLAY_ASSET_LAYOUT.items()
    }


def task_asset_paths(
    args: argparse.Namespace,
    eval_root: Path,
) -> dict[str, Path]:
    values = (
        {}
        if args.replay_assets_root is None
        else replay_asset_paths(args.replay_assets_root)
    )
    values.setdefault(
        "retrieve-lineage-manifest",
        eval_root / "retrieve_combined350_lineage.json",
    )
    overrides = {
        "memory-data-root": resolved(args.memory_data_root),
        "memory-store": resolved(args.memory_store),
        "wash-scene-root": resolved(args.wash_scene_root),
        "retrieve-lineage-manifest": resolved(args.retrieve_lineage_manifest),
        "retrieve-source-root": resolved(args.retrieve_source_root),
        "sweeten-recovered-root": resolved(args.sweeten_recovered_root),
        "pan-transfer-dataset": resolved(args.pan_transfer_dataset),
        "rinse-cutting-board-dataset": resolved(args.rinse_cutting_board_dataset),
        "new5-scene-root": resolved(args.new5_scene_root),
    }
    values.update(
        (name, path)
        for name, path in overrides.items()
        if path is not None
    )
    return {name: path for name, path in values.items() if path is not None}


def required_task_assets(task_group: str) -> tuple[str, ...]:
    if task_group in ("retrieve_fruit", "retrieve_oil"):
        return ("retrieve-lineage-manifest", "retrieve-source-root")
    if task_group in ("wash_and_return", "wash_and_return_same_location"):
        return ("wash-scene-root",)
    if task_group == "sweeten_hot_chocolate":
        return ("sweeten-recovered-root",)
    if task_group == "pan_transfer":
        return ("pan-transfer-dataset",)
    if task_group == "rinse_cutting_board":
        return ("rinse-cutting-board-dataset",)
    return ("new5-scene-root",)


def main() -> None:
    args = parse_args()
    if args.pilot_trials < 0:
        raise ValueError("--pilot-trials must be nonnegative")

    root = Path(__file__).resolve().parents[3]
    eval_root = root / "benchmarks/robocasa/eval"
    robocasa_root = args.robocasa_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    dinov3_model = args.dinov3_model.expanduser().resolve()
    dinov3_runtime_overlay = resolved(args.dinov3_runtime_overlay)
    memory_replay_helper = resolved(args.memory_replay_helper)
    if memory_replay_helper is None:
        memory_replay_helper = (
            robocasa_root / "scripts/replay/replay_memory_demos.py"
        )
    output = args.output.expanduser().resolve()
    sim_python = executable_path(args.sim_python or robocasa_root / ".venv-mem/bin/python")
    policy_python = executable_path(args.policy_python)
    profile = PROFILES[args.architecture]
    asset_paths = task_asset_paths(args, eval_root)
    simulator_paths = simulator_pythonpath_entries(robocasa_root)

    required = (
        robocasa_root,
        policy_python,
        sim_python,
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "params/_METADATA",
        dinov3_model,
        dinov3_model / "config.json",
        memory_replay_helper,
        eval_root / "eval_standard_3view.py",
        eval_root / "run_eval_compat.py",
        eval_root / "serve_policy_compat.py",
    )
    missing = [str(path) for path in required if not path.exists()]
    if (
        dinov3_runtime_overlay is not None
        and not (dinov3_runtime_overlay / "transformers/__init__.py").is_file()
    ):
        missing.append(str(dinov3_runtime_overlay / "transformers/__init__.py"))
    missing.extend(f"--{name}" for name in required_task_assets(args.task_group) if name not in asset_paths)
    missing.extend(str(path) for path in asset_paths.values() if not path.exists())
    if missing:
        raise FileNotFoundError(f"missing evaluation inputs: {missing}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    policy_pythonpath = [
        str(root / "src"),
        str(root / "packages/openpi-client/src"),
    ]
    if dinov3_runtime_overlay is not None:
        policy_pythonpath.insert(0, str(dinov3_runtime_overlay))
    common_pythonpath = os.pathsep.join(policy_pythonpath)
    policy_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.policy_gpu,
        "PYTHONPATH": common_pythonpath,
        "PYTHONUNBUFFERED": "1",
        "MME_SERVER_SCRIPT": str(root / "scripts/serve_imprint_policy.py"),
        "EVAL_CONSOLIDATE_ALL": "1",
        "EVAL_VISUAL_PROFILE": "dinov3",
        "DINOV3_MODEL_PATH": str(dinov3_model),
        "MEMFT_LEGACY_EMPTY_RECENT": "0",
        "MEMFT_DETERMINISTIC_INFERENCE": profile["deterministic"],
        "MEMORY_LANGUAGE_SOURCE": "live",
        "IMPRINT_PREFER_REQUESTED_HISTORY_CONFIG": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    if args.architecture == "recurrent":
        policy_env.update(
            {
                "MEMFT_EXPECTED_MEMORY_ARCHITECTURE": "gated_delta_hybrid_v1",
                "MEMFT_EXPECTED_MEMORY_TRAIN_SCAN": "chunkwise_wy",
                "MEMFT_EXPECTED_MEMORY_INFERENCE": "incremental_recurrent",
            }
        )

    policy_log = (output / "policy.log").open("w")
    policy = subprocess.Popen(
        (
            str(policy_python),
            "-u",
            str(eval_root / "serve_policy_compat.py"),
            "--config",
            profile["config"],
            "--checkpoint",
            str(checkpoint),
            "--history-config",
            profile["history"],
            "--host",
            args.host,
            "--port",
            str(args.port),
        ),
        cwd=root,
        env=policy_env,
        stdout=policy_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_for_port(args.host, args.port, policy, args.startup_timeout)
        sim_env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": args.sim_gpu,
            "PYTHONPATH": os.pathsep.join(
                tuple(str(path) for path in simulator_paths)
                + (str(root / "packages/openpi-client/src"),)
            ),
            "PYTHONUNBUFFERED": "1",
            "MME_EVALUATOR_SCRIPT": str(eval_root / "eval_standard_3view.py"),
            "MME_EGL_LOCAL_DEVICE_ID": "0",
            "ROBOCASA_MEMORY_REPLAY_HELPER": str(memory_replay_helper),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": args.sim_gpu,
        }
        command = [
            str(sim_python),
            "-u",
            str(eval_root / "run_eval_compat.py"),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--task-groups",
            args.task_group,
            "--log-dir",
            str(output),
            "--pilot-trials",
            str(args.pilot_trials),
            "--retrieve-exploration-mode",
            "recorded_state_replay",
            "--wash-exploration-mode",
            "none",
            "--memory-action-alignment",
            "causal_raw_previous",
        ]
        for name, path in asset_paths.items():
            command.extend((f"--{name}", str(path)))
        with (output / "evaluator.log").open("w") as evaluator_log:
            subprocess.run(
                command,
                cwd=robocasa_root,
                env=sim_env,
                stdout=evaluator_log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        group_stats = output / args.task_group / "group_stats.json"
        metrics = write_paper_metrics(
            group_stats,
            output / "paper_metrics.json",
        )
        print(
            f"paper metric: {metrics['paper_num_successes']}/"
            f"{metrics['num_episodes']} "
            f"({metrics['paper_success_rate']:.3f}); "
            f"raw: {metrics['raw_num_successes']}/"
            f"{metrics['num_episodes']} "
            f"({metrics['raw_success_rate']:.3f})"
        )
    finally:
        policy.terminate()
        try:
            policy.wait(timeout=30)
        except subprocess.TimeoutExpired:
            policy.kill()
            policy.wait()
        policy_log.close()


if __name__ == "__main__":
    main()
