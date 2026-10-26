#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import yaml

TASKS = (
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "observe_and_pickup",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "swap_T",
    "swap_blocks",
)

# Same names as mme_vla_suite.shared.wan_alignment.WAN_CHUNK_ALIGNMENTS (this launcher does not
# import mme_vla_suite).
WAN_CHUNK_ALIGNMENTS = ("native", "replicate_first_frame")

PROFILES = {
    "full_attention": {
        "config": "imprint_rmbench_full_attention_memft",
        "history": "cosmo-wan-bottleneck-modul-rmbench-all9-dinov3-memft.yaml",
        "adapter": "robomme_legacy_reference_tcp_policy",
        "deterministic": "1",
    },
    "recurrent": {
        "config": "imprint_rmbench_recurrent_memft",
        "history": "cosmo-wan-gated-delta-rmbench-all9-dinov3-memft-optimized.yaml",
        "adapter": "robomme_gated_delta_optimized_tcp_policy",
        "deterministic": "0",
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dinov3-model",
        type=Path,
        required=True,
        help="Local Hugging Face DINOv3 model directory used by the memory stream.",
    )
    parser.add_argument("--architecture", choices=tuple(PROFILES), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-python", default=str(root / ".venv/bin/python"))
    parser.add_argument("--sim-python", default=sys.executable)
    parser.add_argument("--policy-gpu", default="0")
    parser.add_argument("--sim-gpu", default="1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument(
        "--wan-chunk-alignment",
        choices=WAN_CHUNK_ALIGNMENTS,
        default="native",
        help=(
            "Wan chunk alignment of the memory latents. native: the released checkpoints. "
            "replicate_first_frame: a checkpoint trained on a re-encoded store. The policy "
            "refuses a value that differs from the alignment recorded with the checkpoint."
        ),
    )
    return parser.parse_args()


def wait_for_port(host: str, port: int, process: subprocess.Popen, timeout: int) -> None:
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


def verify_simulator_runtime(
    sim_python: str,
    sim_env: dict[str, str],
    rmbench_root: Path,
    output: Path,
) -> None:
    preflight = "\n".join(
        (
            "import torch",
            "if not torch.cuda.is_available():",
            "    raise RuntimeError('RMBench simulator GPU is not visible')",
            "gpu_name = torch.cuda.get_device_name(0)",
            "capability = torch.cuda.get_device_capability(0)",
            "torch.ones(1, device='cuda').cpu()",
            "from envs.robot.planner import CuroboPlanner",
            "import sapien",
            "sapien.SapienRenderer()",
            "print(",
            "    'RMBENCH_SIMULATOR_READY'",
            "    f' gpu={gpu_name!r} capability={capability[0]}.{capability[1]}'",
            ")",
        )
    )
    result = subprocess.run(
        (
            sim_python,
            "-c",
            preflight,
        ),
        cwd=rmbench_root,
        env=sim_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (output / "simulator_preflight.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "RMBench simulator preflight failed. Verify the simulator PyTorch and "
            "CuRobo builds support the selected GPU and that the container exposes "
            "the NVIDIA Vulkan/GLX driver; see simulator_preflight.log."
        )


def policy_server_settings(
    profile: dict[str, str], checkpoint: Path, wan_chunk_alignment: str
) -> dict[str, object]:
    """Settings for serve_robomme_tcp.py, kept in the run output as policy_server.yaml."""
    return {
        "adapter_module": profile["adapter"],
        "policy_name": profile["adapter"],
        "checkpoint": str(checkpoint),
        "config_name": profile["config"],
        "history_config": profile["history"],
        "prediction_horizon": 50,
        "seed": 42,
        "wan_chunk_alignment": wan_chunk_alignment,
    }


def configure_deterministic_cuda(
    environment: dict[str, str], deterministic: str
) -> None:
    if deterministic == "1":
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    root = Path(__file__).resolve().parents[3]
    eval_root = root / "benchmarks/rmbench/eval"
    adapter_root = eval_root / "runtime_overlay/adapter"
    overlay = eval_root / "src/rmbench_l40_eval/runtime/eval_policy_four_view.py"
    canonical = args.rmbench_root.resolve() / "script/eval_policy_client.py"
    checkpoint = args.checkpoint.expanduser().resolve()
    dinov3_model = args.dinov3_model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    profile = PROFILES[args.architecture]

    required = (
        Path(args.policy_python),
        Path(args.sim_python),
        canonical,
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "params/_METADATA",
        dinov3_model,
        dinov3_model / "config.json",
        overlay,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing evaluation inputs: {missing}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    server_config = output / "policy_server.yaml"
    server_config.write_text(
        yaml.safe_dump(
            policy_server_settings(profile, checkpoint, args.wan_chunk_alignment),
            sort_keys=False,
        )
    )
    evaluator_args = output / "evaluator_args.json"
    evaluator_args.write_text(
        json.dumps(
            {
                "task_name": args.task,
                "task_config": "demo_clean",
                "ckpt_setting": checkpoint.parent.name,
                "policy_name": profile["adapter"],
                "instruction_type": "unseen",
                "seed": 0,
                "port": args.port,
                "execution_horizon": 50,
                "start_seed": args.start_seed,
                "end_seed": None,
                "test_num": args.trials,
                "result_dir": str(output),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    common_pythonpath = os.pathsep.join(
        (
            str(adapter_root),
            str(eval_root / "src"),
            str(root / "src"),
            str(root / "packages/openpi-client/src"),
        )
    )
    policy_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.policy_gpu,
        "PYTHONPATH": common_pythonpath,
        "PYTHONUNBUFFERED": "1",
        "EVAL_CONSOLIDATE_ALL": "1",
        "EVAL_VISUAL_PROFILE": "dinov3",
        "DINOV3_MODEL_PATH": str(dinov3_model),
        "MEMFT_CAUSAL_BUFFER_REQUIRED": "1",
        "MEMFT_LEGACY_EMPTY_RECENT": "0",
        "MEMFT_MEMORY_CAMERA_ORDER": "head_camera,left_camera,right_camera",
        "MEMFT_EXPECTED_MODEL_TYPE": "pi05",
        "MEMFT_DETERMINISTIC_INFERENCE": profile["deterministic"],
        "MEMORY_LANGUAGE_SOURCE": "live",
        "IMPRINT_PREFER_REQUESTED_HISTORY_CONFIG": "1",
        "RMBENCH_POLICY_INPUT_COLOR_ORDER": "bgr",
        "RMBENCH_PREDICTION_HORIZON": "50",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    configure_deterministic_cuda(policy_env, profile["deterministic"])
    if args.architecture == "recurrent":
        policy_env.update(
            {
                "MEMFT_EXPECTED_MEMORY_ARCHITECTURE": "gated_delta_hybrid_v1",
                "MEMFT_EXPECTED_MEMORY_TRAIN_SCAN": "chunkwise_wy",
                "MEMFT_EXPECTED_MEMORY_INFERENCE": "incremental_recurrent",
            }
        )

    sim_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.sim_gpu,
        "PYTHONPATH": os.pathsep.join(
            (str(adapter_root), str(eval_root / "src"), str(args.rmbench_root / "script"))
        ),
        "PYTHONUNBUFFERED": "1",
        "RMBENCH_RENDER_DEVICE": "cuda:0",
        "RMBENCH_POLICY_INPUT_COLOR_ORDER": "bgr",
        "RMBENCH_MODEL_CLIENT_TIMEOUT": str(args.startup_timeout),
    }
    verify_simulator_runtime(
        args.sim_python,
        sim_env,
        args.rmbench_root.resolve(),
        output,
    )

    policy_log = (output / "policy.log").open("w")
    policy = subprocess.Popen(
        (
            args.policy_python,
            "-u",
            str(adapter_root / "serve_robomme_tcp.py"),
            "--config",
            str(server_config),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ),
        env=policy_env,
        stdout=policy_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_for_port(args.host, args.port, policy, args.startup_timeout)
        command = (
            args.sim_python,
            "-u",
            str(overlay),
            "--canonical-eval-policy",
            str(canonical),
            "--canonical-working-directory",
            str(args.rmbench_root.resolve()),
            "--user-args-json",
            str(evaluator_args),
            "--video-output",
            str(output),
            "--task_name",
            args.task,
            "--execution_horizon",
            "50",
            "--start_seed",
            str(args.start_seed),
        )
        with (output / "evaluator.log").open("w") as evaluator_log:
            subprocess.run(
                command,
                cwd=args.rmbench_root,
                env=sim_env,
                stdout=evaluator_log,
                stderr=subprocess.STDOUT,
                check=True,
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
