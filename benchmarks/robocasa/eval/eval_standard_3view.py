"""Evaluate a three-view RoboMME policy on the combined-350 RoboCasa tasks."""

import argparse
import collections
import contextlib
import copy
import dataclasses
import gzip
import hashlib
import io
import importlib.util
import json
import logging
import math
import os
import pathlib
import re
import stat
import types
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np


_EVALUATOR_ROOT = pathlib.Path(__file__).resolve().parent
_RETRIEVE_LINEAGE_MANIFEST = (
    _EVALUATOR_ROOT / "retrieve_combined350_lineage.json"
)
_RETRIEVE_SOURCE_ROOT = _EVALUATOR_ROOT / ".lineage_inputs" / "sources"
_RETRIEVE_LINEAGE_MANIFEST_SHA256 = (
    "1bc4b06030712c15c717d9e0f8036d950994d358854713f5bac727b43f7e4061"
)

TASK_GROUPS = {
    "organize_condiments": (
        ("OrganizeCondiments", 50),
    ),
    "pick_place_counter_to_sink": (
        ("PickPlaceCounterToSink", 50),
    ),
    "place_equal_ice_cubes": (
        ("PlaceEqualIceCubes", 50),
    ),
    "scrub_cutting_board": (
        ("ScrubCuttingBoard", 50),
    ),
    "frying_pan_adjustment": (
        ("FryingPanAdjustment", 50),
    ),
    "retrieve_fruit": (
        ("MemFruitInSinkLeftFar", 25),
        ("MemFruitInSinkRightFar", 25),
    ),
    "retrieve_oil": (
        ("MemRetrieveOilsFromCounterLL", 13),
        ("MemRetrieveOilsFromCounterLR", 13),
        ("MemRetrieveOilsFromCounterRL", 12),
        ("MemRetrieveOilsFromCounterRR", 12),
    ),
    "wash_and_return": (
        ("MemWashAndReturnLeft", 25),
        ("MemWashAndReturnRight", 25),
    ),
    "wash_and_return_same_location": (
        ("MemWashAndReturnSameLocation", 50),
    ),
    "sweeten_hot_chocolate": (
        ("SweetenHotChocolate", 50),
    ),
    "rinse_cutting_board": (
        ("RinseCuttingBoard", 50),
    ),
    "pan_transfer": (
        ("PanTransfer", 50),
    ),
}

PILOT_TASK_GROUPS = {
    "retrieve_fruit_left_causal_pilot": (
        ("MemFruitInSinkLeftFar", 5),
    ),
    "retrieve_fruit_right_causal_pilot": (
        ("MemFruitInSinkRightFar", 5),
    ),
    "wash_and_return_left_pilot": (
        ("MemWashAndReturnLeft", 2),
    ),
    "wash_and_return_right_pilot": (
        ("MemWashAndReturnRight", 2),
    ),
    "wash_and_return_same_location_pilot": (
        ("MemWashAndReturnSameLocation", 2),
    ),
}

RECORDED_SOURCE_EPISODES = {
    "RinseCuttingBoard": (
        0, 3, 4, 11, 12, 13, 14, 16, 17, 19,
        20, 22, 25, 27, 28, 29, 31, 35, 38, 43,
        48, 51, 53, 54, 57, 62, 64, 69, 71, 72,
        75, 76, 77, 81, 82, 86, 87, 88, 89, 91,
        92, 93, 94, 95, 97, 98, 100, 102, 103, 104,
    ),
    "PanTransfer": (
        7, 52, 56, 78, 92, 105, 109, 125, 127, 148,
        163, 174, 187, 192, 205, 211, 224, 227, 228, 230,
        232, 236, 239, 252, 259, 275, 280, 281, 287, 307,
        311, 320, 340, 349, 351, 352, 356, 372, 374, 383,
        395, 399, 417, 432, 442, 443, 453, 457, 468, 469,
    ),
}
NEW5_RECORDED_SCENE_SOURCES = {
    "OrganizeCondiments": (
        "v1.0/pretrain/composite/OrganizeCondiments/20250731/lerobot",
        (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18,
            19, 20, 22, 23, 25, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39,
            40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 59,
        ),
    ),
    "PickPlaceCounterToSink": (
        "v1.0/pretrain/atomic/PickPlaceCounterToSink/20250819/lerobot",
        (
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18,
            19, 21, 22, 23, 24, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37,
            38, 40, 41, 42, 43, 44, 46, 47, 48, 49, 55, 59, 60, 61, 62,
            63, 64, 65,
        ),
    ),
    "PlaceEqualIceCubes": (
        "v1.0/pretrain/composite/PlaceEqualIceCubes/20250801/lerobot",
        (
            0, 1, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22,
            23, 24, 25, 28, 29, 30, 32, 35, 37, 38, 40, 41, 42, 44, 45,
            46, 48, 50, 52, 53, 54, 55, 57, 59, 62, 63, 65, 67, 68, 69,
            70, 71, 72,
        ),
    ),
    "ScrubCuttingBoard": (
        "v1.0/pretrain/composite/ScrubCuttingBoard/20250805/lerobot",
        (
            0, 2, 3, 4, 8, 9, 10, 12, 14, 16, 17, 18, 19, 20, 22, 23, 24,
            25, 27, 28, 31, 32, 33, 35, 36, 37, 38, 39, 40, 43, 44, 45,
            46, 47, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64,
            65, 66, 67,
        ),
    ),
    "FryingPanAdjustment": (
        "v1.0/pretrain/composite/FryingPanAdjustment/20250714/lerobot",
        (
            0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52,
        ),
    ),
}
_NEW5_COMBINED_EPISODE_START = {
    "OrganizeCondiments": 350,
    "PickPlaceCounterToSink": 400,
    "PlaceEqualIceCubes": 450,
    "ScrubCuttingBoard": 500,
    "FryingPanAdjustment": 550,
}
_RINSE_POSTFIX_EPISODES = (0, 3)

GPU_TASK_GROUPS = {
    0: ("retrieve_fruit", "sweeten_hot_chocolate"),
    1: ("retrieve_oil", "rinse_cutting_board"),
    2: ("wash_and_return", "pan_transfer"),
    3: ("wash_and_return_same_location",),
}

MEMORY_ENV_NAMES = tuple(
    sorted(
        {
            env_name
            for variants in TASK_GROUPS.values()
            for env_name, _ in variants
            if env_name.startswith("Mem")
        }
    )
)
_WASH_ENV_NAMES = frozenset({
    "MemWashAndReturnLeft",
    "MemWashAndReturnRight",
    "MemWashAndReturnSameLocation",
})
_WASH_SOURCE_EPISODE_COUNTS = {
    "MemWashAndReturnLeft": 25,
    "MemWashAndReturnRight": 25,
    "MemWashAndReturnSameLocation": 51,
}
_RECORDED_SCENE_FILES = (
    "ep_meta.json",
    "model.xml.gz",
    "states.npz",
)

_DEBUG_SITE_RE = re.compile(
    r"(^reset_region)|(^success_region)|(_(ext|int)_p[xyz0]?$)"
)
_MOBILE_JOINTS = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
)
_MOBILE_ACTION_SLICE = slice(7, 10)
_TORSO_JOINT = "mobilebase0_joint_torso_height"
_VIDEO_VIEW_KEYS = {
    "agentview_left": "video.robot0_agentview_left",
    "agentview_right": "video.robot0_agentview_right",
    "eye_in_hand": "video.robot0_eye_in_hand",
}


def register_memory_gym_envs(registry, registrar) -> None:
    for env_name in MEMORY_ENV_NAMES:
        if f"robocasa/{env_name}" not in registry:
            registrar(env_name)


def make_client(host: str, port: int):
    from openpi_client import msgpack_numpy
    from openpi_client import websocket_client_policy

    class MMEClient(websocket_client_policy.WebsocketClientPolicy):
        def reset(
            self,
            prompt: str | None = None,
            lang_emb: np.ndarray | None = None,
        ) -> dict:
            request = {"reset": True, "prompt": prompt}
            if lang_emb is not None:
                request["lang_emb"] = np.asarray(lang_emb, np.float32)
            self._ws.send(self._packer.pack(request))
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"server error on reset:\n{response}")
            return msgpack_numpy.unpackb(response)

        def add_buffer(self, observation: dict) -> dict:
            request = dict(observation)
            request["add_buffer"] = True
            self._ws.send(self._packer.pack(request))
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"server error on add_buffer:\n{response}")
            return msgpack_numpy.unpackb(response)

    return MMEClient(host, port)


def capture_video_views(observation: dict) -> dict[str, np.ndarray]:
    return {
        view_name: np.ascontiguousarray(observation[key]).copy()
        for view_name, key in _VIDEO_VIEW_KEYS.items()
    }


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    task_groups: str = ""
    split: str = "target"
    log_dir: str = ""
    seed: int = 7
    resize_size: int = 224
    replan_steps: int = 5
    default_horizon: int = 1600
    pilot_trials: int = 0
    save_video: bool = True
    save_all_camera_videos: bool = False
    use_memory_buffer: bool = True
    memory_data_root: str = os.environ.get("ROBOCASA_MEMORY_DATA_ROOT", "")
    memory_store: str = ""
    wash_scene_root: str = os.environ.get("ROBOCASA_WASH_SCENE_ROOT", "")
    retrieve_lineage_manifest: str = str(_RETRIEVE_LINEAGE_MANIFEST)
    retrieve_source_root: str = str(_RETRIEVE_SOURCE_ROOT)
    sweeten_recovered_root: str = os.environ.get(
        "ROBOCASA_SWEETEN_RECOVERED_ROOT", ""
    )
    sweeten_scene_profile: str = "full50"
    sweeten_episode_ids: str = ""
    episode_offset: int = 0
    memory_action_alignment: str = "causal_raw_previous"
    retrieve_exploration_mode: str = "scripted_policy"
    wash_exploration_mode: str = "recorded_state_replay"
    pan_transfer_dataset: str = os.environ.get(
        "ROBOCASA_PAN_TRANSFER_DATASET", ""
    )
    rinse_cutting_board_dataset: str = os.environ.get(
        "ROBOCASA_RINSE_CUTTING_BOARD_DATASET", ""
    )
    new5_scene_root: str = os.environ.get("ROBOCASA_NEW5_SCENE_ROOT", "")
    rinse_scene_profile: str = "canonical_full50"
    rinse_episode_ids: str = ""


def parse_args(argv=None) -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=Args.host)
    parser.add_argument("--port", type=int, default=Args.port)
    parser.add_argument("--task-groups", default=Args.task_groups)
    parser.add_argument("--split", default=Args.split)
    parser.add_argument("--log-dir", default=Args.log_dir)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--resize-size", type=int, default=Args.resize_size)
    parser.add_argument("--replan-steps", type=int, default=Args.replan_steps)
    parser.add_argument("--default-horizon", type=int, default=Args.default_horizon)
    parser.add_argument("--pilot-trials", type=int, default=Args.pilot_trials)
    parser.add_argument("--memory-data-root", default=Args.memory_data_root)
    parser.add_argument("--memory-store", default=Args.memory_store)
    parser.add_argument("--wash-scene-root", default=Args.wash_scene_root)
    parser.add_argument(
        "--retrieve-lineage-manifest",
        default=Args.retrieve_lineage_manifest,
    )
    parser.add_argument(
        "--retrieve-source-root",
        default=Args.retrieve_source_root,
    )
    parser.add_argument(
        "--sweeten-recovered-root",
        default=Args.sweeten_recovered_root,
    )
    parser.add_argument(
        "--sweeten-scene-profile",
        choices=("full50", "full25", "part_b_210_217"),
        default=Args.sweeten_scene_profile,
    )
    parser.add_argument(
        "--sweeten-episode-ids",
        default=Args.sweeten_episode_ids,
    )
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=Args.episode_offset,
    )
    parser.add_argument(
        "--memory-action-alignment",
        choices=("causal_raw_previous", "stage2_selected_previous"),
        default=Args.memory_action_alignment,
    )
    parser.add_argument(
        "--retrieve-exploration-mode",
        choices=("none", "scripted_policy", "recorded_state_replay"),
        default=Args.retrieve_exploration_mode,
    )
    parser.add_argument(
        "--wash-exploration-mode",
        choices=("none", "recorded_state_replay"),
        default=Args.wash_exploration_mode,
    )
    parser.add_argument(
        "--pan-transfer-dataset",
        default=Args.pan_transfer_dataset,
    )
    parser.add_argument(
        "--rinse-cutting-board-dataset",
        default=Args.rinse_cutting_board_dataset,
    )
    parser.add_argument(
        "--new5-scene-root",
        default=Args.new5_scene_root,
    )
    parser.add_argument(
        "--rinse-scene-profile",
        choices=("canonical_full50", "postfix_0_3"),
        default=Args.rinse_scene_profile,
    )
    parser.add_argument(
        "--rinse-episode-ids",
        default=Args.rinse_episode_ids,
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=Args.save_video,
    )
    parser.add_argument(
        "--save-all-camera-videos",
        action="store_true",
        default=Args.save_all_camera_videos,
    )
    parser.add_argument(
        "--use-memory-buffer",
        action=argparse.BooleanOptionalAction,
        default=Args.use_memory_buffer,
    )
    return Args(**vars(parser.parse_args(argv)))


def group_variants(
    group_name: str,
    pilot_trials: int = 0,
) -> tuple[tuple[str, int], ...]:
    if group_name in TASK_GROUPS:
        variants = TASK_GROUPS[group_name]
    else:
        variants = PILOT_TASK_GROUPS[group_name]
    if pilot_trials <= 0:
        return variants

    remaining = min(pilot_trials, sum(count for _, count in variants))
    counts = [0] * len(variants)
    while remaining:
        progressed = False
        for index, (_, maximum) in enumerate(variants):
            if remaining == 0:
                break
            if counts[index] < maximum:
                counts[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return tuple(
        (env_name, count)
        for (env_name, _), count in zip(variants, counts)
        if count
    )


def recovered_demo_for_episode(
    recovered_demos: tuple,
    episode_index: int,
    episode_offset: int,
):
    if not recovered_demos:
        raise ValueError("recovered demo inventory must not be empty")
    if episode_index < 0 or episode_offset < 0:
        raise ValueError("recovered demo indices must be nonnegative")
    return recovered_demos[
        (episode_offset + episode_index) % len(recovered_demos)
    ]


def recorded_reference_for_episode(
    recorded_references: tuple,
    episode_index: int,
    episode_offset: int,
):
    return recovered_demo_for_episode(
        recorded_references,
        episode_index,
        episode_offset,
    )


@dataclasses.dataclass(frozen=True)
class DemoReference:
    dataset_path: pathlib.Path
    demo_key: str
    action_length: int


@dataclasses.dataclass(frozen=True)
class _ValidatedRecoveryBundle:
    ep_meta: object
    metadata: object
    model_xml: bytes
    initial_state: np.ndarray
    input_hashes: object
    output_hashes: object
    payload_hashes: object
    physical_augmentation: tuple


@dataclasses.dataclass(frozen=True)
class _ValidatedRecordedScenePayload:
    ep_meta_json: bytes
    model_xml: bytes
    states_npz: bytes
    file_hashes: object


@dataclasses.dataclass(frozen=True)
class RecordedSceneReference:
    dataset_path: pathlib.Path
    episode_index: int
    combined_episode: int | None = None
    source_episode: int | None = None
    source_root: pathlib.Path | None = None
    source_demo_key: str | None = None
    environment_variant: str | None = None
    source_hashes: object | None = None
    source_file_sha256: str | None = None
    source_layout_id: int | None = None
    source_style_id: int | None = None
    goal_count: int | None = None
    recovery_metadata: object | None = None
    recovery_path: pathlib.Path | None = None
    recovery_report_entry: object | None = None
    validated_payload: _ValidatedRecoveryBundle | None = None
    validated_scene_payload: _ValidatedRecordedScenePayload | None = None


@dataclasses.dataclass(frozen=True)
class DemoInitialization:
    dataset_path: pathlib.Path
    demo_key: str
    ep_meta: dict
    model_file: str
    initial_state: np.ndarray
    action_length: int
    combined_episode: int | None = None
    source_episode: int | None = None
    source_root: pathlib.Path | None = None
    source_demo_key: str | None = None
    environment_variant: str | None = None
    source_hashes: object | None = None
    source_file_sha256: str | None = None
    source_layout_id: int | None = None
    source_style_id: int | None = None
    goal_count: int | None = None
    recovery_metadata: object | None = None
    recovery_path: pathlib.Path | None = None
    recovery_report_entry: object | None = None
    validated_payload: _ValidatedRecoveryBundle | None = None
    recorded_states: np.ndarray | None = None
    recorded_actions: np.ndarray | None = None
    recorded_transition: int | None = None


_SWEETEN_GENERATION_COMMIT = (
    "5e02c58b3cd373d32531c619442571372310c03e"
)
_SWEETEN_GLOBAL_SEED = 20260801
_SOURCE31_DERIVED_SEED = 11955166025309133556
_SOURCE31_ORIGINAL_CUBE_NAMES = ("sugar0", "sugar1", "sugar2")
_PINNED_RECOVERY_ATTEMPTS = {
    12: (15, 22),
    44: (0, 0),
    53: (0, 0),
    65: (0, 0),
    71: (4, 11),
    79: (0, 0),
    80: (2, 2),
    81: (0, 0),
    83: (0, 0),
    85: (0, 0),
    89: (0, 1),
    90: (0, 3),
    93: (0, 0),
    97: (2, 2),
}
_RECOVERY_MAX_CODEC_NOISE_ENVELOPE = 1.25
_RECOVERY_CODEC_NOISE_SAFETY_FACTOR = 1.25
_RECOVERY_AUGMENTATION_MANIFEST_SHA256 = (
    "f88f5ee17c1a3c5ba531ad714674f44cc9e5b8e6c172368fdd93e9a64da4d6b4"
)
_RECOVERY_AUGMENTATION_LINEAGE = {
    "schema": "sweeten-production-archive/v1",
    "snapshot_kind": "dirty_worktree_overlay",
    "base_commit": _SWEETEN_GENERATION_COMMIT,
    "base_tree": "0a2d8612a544408830b7447243880a369f7a676c",
    "payload_root": "payload/scripts/sweeten_augmentation",
    "ordered_set_digest": (
        "f43ef047adfa720edb5f649bd86e97669e5db0c0916cdf3880c2963ba5c390d8"
    ),
}
_RECOVERY_ATTEMPTS = tuple(range(32))
_RECOVERY_CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
_RECOVERY_BUNDLE_FILES = {
    "ep_meta.json",
    "model.xml.gz",
    "states.npz",
    "recovery.json",
}
_RECOVERY_INPUT_FILES = {
    "source/ep_meta.json",
    "source/model.xml.gz",
    "source/states.npz",
    *(f"training/{camera}.mp4" for camera in _RECOVERY_CAMERA_NAMES),
}
_RECOVERY_RUNTIME_KEYS = {
    "python",
    "numpy",
    "pillow",
    "platform",
    "mujoco",
    "robosuite",
    "robocasa",
    "augmentation_archive_root",
    "augmentation_manifest_sha256",
}
SWEETEN_RECOVERED_MAPPING = (
    (200, 4, 1),
    (201, 4, 2),
    (202, 12, 1),
    (203, 12, 2),
    (204, 12, 3),
    (205, 27, 1),
    (206, 27, 2),
    (207, 31, 1),
    (208, 31, 2),
    (209, 31, 3),
    (210, 44, 1),
    (211, 44, 2),
    (212, 44, 3),
    (213, 48, 1),
    (214, 48, 2),
    (215, 51, 1),
    (216, 51, 2),
    (217, 51, 3),
    (218, 53, 1),
    (219, 53, 2),
    (220, 53, 3),
    (221, 62, 1),
    (222, 62, 2),
    (223, 65, 1),
    (224, 65, 2),
    (225, 65, 3),
    (226, 71, 1),
    (227, 71, 2),
    (228, 79, 1),
    (229, 79, 2),
    (230, 79, 3),
    (231, 80, 1),
    (232, 80, 2),
    (233, 81, 1),
    (234, 81, 2),
    (235, 81, 3),
    (236, 83, 1),
    (237, 83, 2),
    (238, 83, 3),
    (239, 85, 1),
    (240, 85, 2),
    (241, 89, 1),
    (242, 89, 2),
    (243, 89, 3),
    (244, 90, 1),
    (245, 90, 2),
    (246, 93, 1),
    (247, 93, 2),
    (248, 97, 1),
    (249, 97, 2),
)
_SWEETEN_RECOVERED_MAPPING_SHA256 = (
    "72d4a489ac67add1ccdd1a97bf0b64fedfaa4c88bce270655aaacbc5ad732c1d"
)
_SWEETEN_FULL25_MAPPING = SWEETEN_RECOVERED_MAPPING[:25]
_SWEETEN_FULL25_MAPPING_SHA256 = (
    "c5145960f035ce36d2be01e3dfb942ea1370b86a9798179ed0980bcbd8688c2d"
)
_SWEETEN_PART_B_MAPPING = SWEETEN_RECOVERED_MAPPING[10:18]
_SWEETEN_PART_B_MAPPING_SHA256 = (
    "75cf806d587ec6944fd9041824f88861e8fe59613e98bed3d6e0c0035f0441fe"
)
_SWEETEN_PART_B_SELECTED_EPISODES = (210, 211)


@dataclasses.dataclass(frozen=True)
class _SweetenMapping:
    combined_episode: int
    source_episode: int
    goal_count: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_recovery_root_locators(value) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and all(
        isinstance(item, str) for item in value
    ):
        values = value
    else:
        raise ValueError("Sweeten recovery root provenance is invalid")
    if not values or any(not item for item in values):
        raise ValueError("Sweeten recovery root provenance is invalid")
    locators = tuple(sorted(set(values)))
    if list(locators) != values and not isinstance(value, str):
        raise ValueError("Sweeten recovery root provenance is not canonical")
    return locators


def _required_open_flag(name: str) -> int:
    value = getattr(os, name, None)
    if value is None:
        raise RuntimeError(f"{name} is required for Sweeten validation")
    return value


@contextlib.contextmanager
def _open_directory_fd(
    path,
    label: str,
    *,
    dir_fd: int | None = None,
):
    flags = (
        os.O_RDONLY
        | _required_open_flag("O_DIRECTORY")
        | _required_open_flag("O_NOFOLLOW")
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags, dir_fd=dir_fd)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    except OSError as error:
        raise ValueError(
            f"{label} is not a no-follow directory: {path}"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a directory: {path}")
        yield descriptor
    finally:
        os.close(descriptor)


def _read_regular_file_bytes_at(
    directory_fd: int,
    name: str,
    label: str,
) -> bytes:
    if pathlib.PurePath(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} has an invalid basename")
    flags = (
        os.O_RDONLY
        | _required_open_flag("O_NOFOLLOW")
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {name}") from error
    except OSError as error:
        raise ValueError(
            f"{label} is not a readable no-follow regular file: {name}"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular file: {name}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _parse_json_object(data: bytes, path: pathlib.Path, label: str) -> dict:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _freeze_json(value):
    if isinstance(value, dict):
        return types.MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value):
    if isinstance(value, collections.abc.Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _recovery_number(value, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _validate_hash_manifest(
    value,
    expected_names: set[str],
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != expected_names:
        raise ValueError(f"{label} has an invalid file inventory")
    for name, digest in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"{label} has an invalid SHA256 digest")
    return value


def _validate_runtime_versions(value, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _RECOVERY_RUNTIME_KEYS:
        raise ValueError(f"{label} has an invalid inventory")
    if any(
        not isinstance(version, str) or not version
        for version in value.values()
    ):
        raise ValueError(f"{label} contains an invalid version")
    archive_root = pathlib.PurePath(value["augmentation_archive_root"])
    if not archive_root.is_absolute():
        raise ValueError(f"{label} has a non-absolute archive root")
    if (
        value["augmentation_manifest_sha256"]
        != _RECOVERY_AUGMENTATION_MANIFEST_SHA256
    ):
        raise ValueError(f"{label} has an invalid archive manifest digest")
    return value


def _validate_augmentation_archive(
    value,
    runtime_versions: dict[str, str],
) -> dict:
    expected = {
        "mode": "verified_archive",
        "root": runtime_versions["augmentation_archive_root"],
        "manifest_sha256": runtime_versions[
            "augmentation_manifest_sha256"
        ],
        **_RECOVERY_AUGMENTATION_LINEAGE,
    }
    if value != expected:
        raise ValueError("Sweeten augmentation archive evidence is invalid")
    return value


def _score_comparison_tolerance(*values: float) -> float:
    float32_ulp = max(
        abs(float(np.spacing(np.float32(abs(float(value))))))
        for value in values
    )
    return max(1e-12, 2.0 * float32_ulp)


def _same_score(first: float, second: float) -> bool:
    if type(first) not in (int, float) or type(second) not in (int, float):
        return False
    first = float(first)
    second = float(second)
    if not math.isfinite(first) or not math.isfinite(second):
        return False
    return math.isclose(
        first,
        second,
        rel_tol=0.0,
        abs_tol=_score_comparison_tolerance(first, second),
    )


def _validate_per_camera_scores(value, label: str) -> dict[str, float]:
    if (
        not isinstance(value, dict)
        or set(value) != set(_RECOVERY_CAMERA_NAMES)
    ):
        raise ValueError(f"{label} has an invalid camera inventory")
    return {
        camera: _recovery_number(value[camera], f"{label} for {camera}")
        for camera in _RECOVERY_CAMERA_NAMES
    }


def _validate_calibration(value, report_envelope: float) -> dict:
    explicit_keys = {"method", "codec_noise_envelope"}
    if isinstance(value, dict) and set(value) == explicit_keys:
        explicit_envelope = _recovery_number(
            value["codec_noise_envelope"],
            "Sweeten explicit codec noise envelope",
        )
        if (
            value["method"] != "explicit_override"
            or explicit_envelope > _RECOVERY_MAX_CODEC_NOISE_ENVELOPE
            or not _same_score(explicit_envelope, report_envelope)
        ):
            raise ValueError(
                "Sweeten explicit codec calibration is invalid"
            )
        return value

    expected_keys = {
        "calibration_source_episode",
        "selected_attempt",
        "layout_attempt",
        "original_cube_names",
        "winning_aggregate_score",
        "winning_per_camera",
        "independent_codec_aggregate_score",
        "independent_codec_per_camera",
        "zero_envelope_unique_winner",
        "safety_factor",
        "maximum_codec_noise_envelope",
        "codec_noise_envelope",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("Sweeten codec calibration evidence is invalid")
    if (
        type(value["calibration_source_episode"]) is not int
        or value["calibration_source_episode"] != 31
        or type(value["selected_attempt"]) is not int
        or value["selected_attempt"] != 0
        or type(value["layout_attempt"]) is not int
        or value["layout_attempt"] != 0
        or value["original_cube_names"]
        != list(_SOURCE31_ORIGINAL_CUBE_NAMES)
        or value["zero_envelope_unique_winner"] is not True
        or not _same_score(
            _recovery_number(
                value["safety_factor"],
                "Sweeten calibration safety factor",
            ),
            _RECOVERY_CODEC_NOISE_SAFETY_FACTOR,
        )
        or not _same_score(
            _recovery_number(
                value["maximum_codec_noise_envelope"],
                "Sweeten calibration maximum codec noise envelope",
            ),
            _RECOVERY_MAX_CODEC_NOISE_ENVELOPE,
        )
        or not _same_score(
            _recovery_number(
                value["codec_noise_envelope"],
                "Sweeten calibration codec noise envelope",
            ),
            report_envelope,
        )
    ):
        raise ValueError("Sweeten codec calibration constants are invalid")
    winning = _validate_per_camera_scores(
        value["winning_per_camera"],
        "Sweeten calibration winning score",
    )
    winning_aggregate = _recovery_number(
        value["winning_aggregate_score"],
        "Sweeten calibration winning aggregate",
    )
    if not _same_score(
        winning_aggregate,
        sum(winning.values()) / len(winning),
    ):
        raise ValueError("Sweeten calibration winning scores are inconsistent")
    independent = _validate_per_camera_scores(
        value["independent_codec_per_camera"],
        "Sweeten independent codec score",
    )
    independent_aggregate = _recovery_number(
        value["independent_codec_aggregate_score"],
        "Sweeten independent codec aggregate",
    )
    if not _same_score(
        independent_aggregate,
        sum(independent.values()) / len(independent),
    ):
        raise ValueError(
            "Sweeten independent codec scores are inconsistent"
        )
    scaled = (
        independent_aggregate * _RECOVERY_CODEC_NOISE_SAFETY_FACTOR
    )
    expected_envelope = max(
        math.nextafter(scaled, math.inf),
        independent_aggregate
        + _score_comparison_tolerance(independent_aggregate, scaled),
    )
    if (
        report_envelope > _RECOVERY_MAX_CODEC_NOISE_ENVELOPE
        or not _same_score(report_envelope, expected_envelope)
    ):
        raise ValueError("Sweeten codec calibration envelope is inconsistent")
    return value


def _validate_rejected_attempts(value, allowed_attempts: set[int]) -> list:
    if not isinstance(value, list):
        raise ValueError("recovery rejected_attempts is invalid")
    attempts = set()
    for row in value:
        if (
            not isinstance(row, dict)
            or set(row) != {"attempt", "type", "message"}
            or type(row["attempt"]) is not int
            or row["attempt"] not in allowed_attempts
            or row["attempt"] in attempts
            or not isinstance(row["type"], str)
            or not row["type"]
            or not isinstance(row["message"], str)
            or not row["message"]
        ):
            raise ValueError("recovery rejected attempt record is invalid")
        attempts.add(row["attempt"])
    return value


def _validate_recovery_scores(
    recovery: dict,
    report_envelope: float,
    allowed_attempts: set[int],
    source_episode: int | None = None,
) -> None:
    # The retained selected_attempt schema key is a candidate ordinal.
    selected_candidate_ordinal = recovery.get("selected_attempt")
    if (
        type(selected_candidate_ordinal) is not int
        or selected_candidate_ordinal < 0
    ):
        raise ValueError("recovery selected_attempt is invalid")
    if type(recovery.get("selected_canonical_source")) is not bool:
        raise ValueError("recovery selected_canonical_source is invalid")
    if selected_candidate_ordinal not in allowed_attempts:
        raise ValueError("recovery selected_attempt was not attempted")
    selected_per_camera = _validate_per_camera_scores(
        recovery.get("selected_per_camera"),
        "recovery selected score",
    )
    envelope = _recovery_number(
        recovery.get("codec_noise_envelope"),
        "recovery codec noise envelope",
    )
    if (
        envelope > _RECOVERY_MAX_CODEC_NOISE_ENVELOPE
        or not _same_score(envelope, report_envelope)
    ):
        raise ValueError("recovery codec noise envelope is inconsistent")

    candidate_scores = recovery.get("candidate_scores")
    if not isinstance(candidate_scores, list) or not candidate_scores:
        raise ValueError("recovery candidate_scores is invalid")
    candidates = []
    attempts = set()
    for row in candidate_scores:
        if not isinstance(row, dict) or set(row) != {
            "attempt",
            "canonical_source",
            "per_camera",
            "aggregate",
        }:
            raise ValueError("recovery candidate score row is invalid")
        attempt = row.get("attempt")
        canonical = row.get("canonical_source")
        per_camera = row.get("per_camera")
        if (
            type(attempt) is not int
            or attempt < 0
            or attempt not in allowed_attempts
            or attempt in attempts
            or type(canonical) is not bool
            or not isinstance(per_camera, dict)
            or set(per_camera) != set(_RECOVERY_CAMERA_NAMES)
        ):
            raise ValueError("recovery candidate score row is invalid")
        attempts.add(attempt)
        camera_scores = _validate_per_camera_scores(
            per_camera,
            "recovery candidate score",
        )
        aggregate = _recovery_number(
            row.get("aggregate"),
            "candidate aggregate score",
        )
        canonical_aggregate = float(
            np.mean(
                [
                    camera_scores[camera]
                    for camera in _RECOVERY_CAMERA_NAMES
                ]
            )
        )
        if not _same_score(
            aggregate,
            canonical_aggregate,
        ):
            raise ValueError("candidate aggregate score is inconsistent")
        candidates.append(
            (canonical_aggregate, attempt, canonical, camera_scores)
        )
    canonical_candidates = [
        candidate for candidate in candidates if candidate[2]
    ]
    if (
        len(canonical_candidates) > 1
        or any(candidate[1] != 0 for candidate in canonical_candidates)
    ):
        raise ValueError("recovery canonical source score is invalid")

    ranked = sorted(candidates)
    selected = next(
        (candidate for candidate in candidates
         if candidate[1] == selected_candidate_ordinal),
        None,
    )
    if selected is None or selected != ranked[0]:
        raise ValueError("recovery selected candidate is not the winner")
    if recovery["selected_canonical_source"] != selected[2]:
        raise ValueError("recovery selected canonical flag is inconsistent")
    if any(
        not _same_score(selected[3][camera], selected_per_camera[camera])
        for camera in _RECOVERY_CAMERA_NAMES
    ):
        raise ValueError("recovery selected per-camera scores are inconsistent")
    if not _same_score(
        recovery.get("best_aggregate_score"),
        ranked[0][0],
    ):
        raise ValueError("recovery best aggregate score is inconsistent")
    if ranked[0][0] > envelope and not _same_score(ranked[0][0], envelope):
        raise ValueError("recovery winner exceeds the codec noise envelope")

    if len(ranked) == 1:
        if (
            recovery.get("second_aggregate_score") is not None
            or recovery.get("aggregate_margin") is not None
        ):
            raise ValueError("singleton recovery score evidence is invalid")
    else:
        second = ranked[1][0]
        margin = second - ranked[0][0]
        recorded_second = _recovery_number(
            recovery.get("second_aggregate_score"),
            "recovery second aggregate score",
        )
        recorded_margin = _recovery_number(
            recovery.get("aggregate_margin"),
            "recovery aggregate margin",
        )
        if (
            not _same_score(recorded_second, second)
            or not _same_score(recorded_margin, margin)
            or margin <= 0
            or _same_score(ranked[0][0], second)
        ):
            raise ValueError("recovery score margin is inconsistent")
        for camera in _RECOVERY_CAMERA_NAMES:
            if any(
                selected[3][camera] > candidate[3][camera]
                and not _same_score(
                    selected[3][camera],
                    candidate[3][camera],
                )
                for candidate in candidates
                if candidate[1] != selected_candidate_ordinal
            ):
                raise ValueError(
                    "recovery per-camera winner is inconsistent"
                )
    rejected = _validate_rejected_attempts(
        recovery.get("rejected_attempts"),
        allowed_attempts,
    )
    rejected_attempts = {row["attempt"] for row in rejected}
    if attempts & rejected_attempts:
        raise ValueError("candidate and rejected attempts overlap")
    if recovery["selected_canonical_source"]:
        if (
            attempts != {0}
            or rejected_attempts
            or len(candidates) != 1
            or not selected[2]
            or selected_candidate_ordinal != 0
        ):
            raise ValueError(
                "canonical recovery attempt accounting is invalid"
            )
    else:
        pinned = _PINNED_RECOVERY_ATTEMPTS.get(source_episode)
        pinned_singleton = (
            pinned is not None
            and attempts == {pinned[0]}
            and not rejected_attempts
            and len(candidates) == 1
            and selected_candidate_ordinal == pinned[0]
            and recovery.get("layout_attempt") == pinned[1]
        )
        if (
            canonical_candidates
            or (
                attempts | rejected_attempts != allowed_attempts
                and not pinned_singleton
            )
        ):
            raise ValueError(
                "noncanonical recovery attempt accounting is incomplete"
            )


def _validate_recovery_augmentation(
    value,
    *,
    combined_episode: int,
    source_episode: int,
    goal_count: int,
) -> dict:
    expected_keys = {
        "combined_episode",
        "source_episode",
        "goal_count",
        "global_seed",
        "derived_seed",
        "original_cube_count",
        "total_cube_count",
        "added_distractor_count",
        "original_cube_names",
        "distractor_names",
        "layout_attempt",
        "generation_commit",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("recovery augmentation has an invalid inventory")
    expected_lineage = {
        "combined_episode": combined_episode,
        "source_episode": source_episode,
        "goal_count": goal_count,
    }
    if any(
        type(value.get(name)) is not int or value[name] != expected
        for name, expected in expected_lineage.items()
    ):
        raise ValueError("recovery augmentation lineage is invalid")
    integer_fields = (
        "derived_seed",
        "original_cube_count",
        "total_cube_count",
        "added_distractor_count",
    )
    if (
        type(value["global_seed"]) is not int
        or value["global_seed"] != _SWEETEN_GLOBAL_SEED
        or any(
            type(value[name]) is not int or value[name] < 0
            for name in integer_fields
        )
        or value["original_cube_count"] < 1
        or type(value["layout_attempt"]) is not int
        or value["layout_attempt"] < 0
        or value["generation_commit"] != _SWEETEN_GENERATION_COMMIT
    ):
        raise ValueError("recovery augmentation constants are invalid")
    original_names = value["original_cube_names"]
    distractor_names = value["distractor_names"]
    if (
        not isinstance(original_names, list)
        or not isinstance(distractor_names, list)
        or any(
            not isinstance(name, str) or not name
            for name in original_names + distractor_names
        )
        or len(set(original_names)) != len(original_names)
        or len(set(distractor_names)) != len(distractor_names)
        or set(original_names) & set(distractor_names)
        or len(original_names) != value["original_cube_count"]
        or len(distractor_names) != value["added_distractor_count"]
        or value["total_cube_count"]
        != value["original_cube_count"] + value["added_distractor_count"]
        or goal_count > value["original_cube_count"]
    ):
        raise ValueError("recovery augmentation counts or names are invalid")
    return value


def _validate_recovery_bundle(
    episode_fd: int,
    *,
    episode_name: str,
    combined_episode: int,
    source_episode: int,
    goal_count: int,
    report_entry: dict,
    source_entry: dict,
    report_envelope: float,
    report_attempts: set[int],
    report_runtime_versions: dict[str, str],
) -> _ValidatedRecoveryBundle:
    if episode_name != f"episode_{combined_episode:06d}":
        raise ValueError("Sweeten episode directory name is inconsistent")
    inventory = set(os.listdir(episode_fd))
    if inventory != _RECOVERY_BUNDLE_FILES:
        raise ValueError(f"{episode_name}: unexpected bundle inventory")
    bundle_data = {
        name: _read_regular_file_bytes_at(
            episode_fd,
            name,
            f"{episode_name} {name}",
        )
        for name in sorted(_RECOVERY_BUNDLE_FILES)
    }

    recovery_path = pathlib.Path(episode_name) / "recovery.json"
    recovery = _parse_json_object(
        bundle_data["recovery.json"],
        recovery_path,
        f"{episode_name} recovery metadata",
    )
    expected_lineage = {
        "combined_episode": combined_episode,
        "source_episode": source_episode,
        "goal_count": goal_count,
    }
    expected_recovery_keys = {
        "schema_version",
        "augmentation",
        "selected_attempt",
        "layout_attempt",
        "selected_canonical_source",
        "selected_per_camera",
        "candidate_scores",
        "rejected_attempts",
        "best_aggregate_score",
        "second_aggregate_score",
        "aggregate_margin",
        "codec_noise_envelope",
        "combined_episode",
        "source_episode",
        "goal_count",
        "input_hashes",
        "runtime_versions",
        "generation_commit",
        "output_hashes",
    }
    if set(recovery) != expected_recovery_keys:
        raise ValueError(f"{episode_name}: recovery schema is invalid")
    if (
        type(recovery.get("schema_version")) is not int
        or recovery["schema_version"] != 1
        or any(
            type(recovery.get(name)) is not int
            or recovery[name] != value
            for name, value in expected_lineage.items()
        )
    ):
        raise ValueError(f"{episode_name}: recovery lineage is invalid")
    if recovery.get("generation_commit") != _SWEETEN_GENERATION_COMMIT:
        raise ValueError(f"{episode_name}: generation commit is invalid")
    if (
        type(recovery.get("layout_attempt")) is not int
        or recovery["layout_attempt"] < 0
    ):
        raise ValueError(f"{episode_name}: layout attempt is invalid")
    _validate_recovery_scores(
        recovery,
        report_envelope,
        report_attempts,
        source_episode=source_episode,
    )
    input_hashes = _validate_hash_manifest(
        recovery.get("input_hashes"),
        _RECOVERY_INPUT_FILES,
        f"{episode_name} recovery input hashes",
    )
    runtime_versions = _validate_runtime_versions(
        recovery.get("runtime_versions"),
        f"{episode_name} runtime versions",
    )
    if runtime_versions != report_runtime_versions:
        raise ValueError(
            f"{episode_name}: runtime versions differ from the report"
        )

    augmentation = _validate_recovery_augmentation(
        recovery.get("augmentation"),
        combined_episode=combined_episode,
        source_episode=source_episode,
        goal_count=goal_count,
    )
    if recovery["layout_attempt"] != augmentation["layout_attempt"]:
        raise ValueError(f"{episode_name}: layout attempt is inconsistent")
    canonical_source = recovery["selected_canonical_source"]
    if (
        canonical_source
        != (augmentation["added_distractor_count"] == 0)
        or canonical_source
        and (
            recovery["selected_attempt"] != 0
            or augmentation["layout_attempt"] != 0
            or augmentation["distractor_names"]
        )
    ):
        raise ValueError(
            f"{episode_name}: canonical recovery distractor evidence "
            "is inconsistent"
        )

    actual_payload_hashes = {
        name: _sha256_bytes(bundle_data[name])
        for name in ("ep_meta.json", "model.xml.gz", "states.npz")
    }
    recovery_output_hashes = _validate_hash_manifest(
        recovery.get("output_hashes"),
        {"ep_meta.json", "model.xml.gz", "states.npz"},
        f"{episode_name} recovery output hashes",
    )
    if recovery_output_hashes != actual_payload_hashes:
        raise ValueError(f"{episode_name}: recovery output hashes do not match")

    ep_meta = _parse_json_object(
        bundle_data["ep_meta.json"],
        pathlib.Path(episode_name) / "ep_meta.json",
        f"{episode_name} episode metadata",
    )
    if ep_meta.get("augmentation") != augmentation:
        raise ValueError(f"{episode_name}: episode augmentation is inconsistent")
    expected_instruction = sweeten_instruction(goal_count)
    if ep_meta.get("lang") != expected_instruction:
        raise ValueError(f"{episode_name}: goal language is inconsistent")
    ep_recovery = ep_meta.get("recovery")
    if (
        not isinstance(ep_recovery, dict)
        or set(ep_recovery) != expected_recovery_keys - {
            "schema_version",
            "augmentation",
        }
    ):
        raise ValueError(f"{episode_name}: episode recovery evidence is missing")
    evidence_keys = set(recovery) - {
        "schema_version",
        "augmentation",
        "output_hashes",
    }
    if any(ep_recovery.get(key) != recovery.get(key) for key in evidence_keys):
        raise ValueError(f"{episode_name}: episode recovery evidence differs")
    ep_payload_hashes = _validate_hash_manifest(
        ep_recovery.get("output_hashes"),
        {"model.xml.gz", "states.npz"},
        f"{episode_name} episode payload hashes",
    )
    if ep_payload_hashes != {
        name: actual_payload_hashes[name]
        for name in ("model.xml.gz", "states.npz")
    }:
        raise ValueError(f"{episode_name}: episode payload hashes do not match")

    expected_report_path = f"extras/{episode_name}"
    if (
        type(report_entry.get("combined_episode")) is not int
        or report_entry["combined_episode"] != combined_episode
        or type(report_entry.get("source_episode")) is not int
        or report_entry["source_episode"] != source_episode
        or type(report_entry.get("goal_count")) is not int
        or report_entry["goal_count"] != goal_count
        or report_entry.get("path") != expected_report_path
        or report_entry.get("input_hashes") != input_hashes
    ):
        raise ValueError(f"{episode_name}: report bundle row is inconsistent")
    report_hashes = _validate_hash_manifest(
        report_entry.get("output_hashes"),
        _RECOVERY_BUNDLE_FILES,
        f"{episode_name} report output hashes",
    )
    actual_bundle_hashes = {
        name: _sha256_bytes(bundle_data[name])
        for name in _RECOVERY_BUNDLE_FILES
    }
    if report_hashes != actual_bundle_hashes:
        raise ValueError(f"{episode_name}: report output hashes do not match")
    if source_entry.get("selected_attempt") != recovery["selected_attempt"]:
        raise ValueError(f"{episode_name}: source attempt is inconsistent")
    if source_entry.get("layout_attempt") != recovery["layout_attempt"]:
        raise ValueError(f"{episode_name}: source layout attempt is inconsistent")
    if source_entry.get("rejected_attempts") != recovery["rejected_attempts"]:
        raise ValueError(f"{episode_name}: source rejections are inconsistent")

    try:
        model_xml = gzip.decompress(bundle_data["model.xml.gz"])
        if not model_xml:
            raise ValueError(f"{episode_name}: recovered model is empty")
        ET.fromstring(model_xml)
        with np.load(
            io.BytesIO(bundle_data["states.npz"]),
            allow_pickle=False,
        ) as archive:
            if set(archive.files) != {"states"}:
                raise ValueError(
                    f"{episode_name}: states archive inventory is invalid"
                )
            states = np.asarray(archive["states"])
            if (
                states.dtype != np.dtype(np.float64)
                or not np.isrealobj(states)
                or states.ndim != 2
                or states.shape[0] != 1
                or states.shape[1] < 1
                or not np.all(np.isfinite(states))
            ):
                raise ValueError(
                    f"{episode_name}: recovered states are invalid"
                )
            state_bytes = states.tobytes(order="C")
            initial_state = np.frombuffer(
                state_bytes,
                dtype=np.float64,
            ).reshape(states.shape)
    except (
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        ET.ParseError,
    ) as error:
        if isinstance(error, ValueError) and str(error).startswith(episode_name):
            raise
        raise ValueError(
            f"{episode_name}: recovered scene payload is invalid"
        ) from error
    physical_fields = (
        "global_seed",
        "derived_seed",
        "original_cube_count",
        "total_cube_count",
        "added_distractor_count",
        "layout_attempt",
        "generation_commit",
    )
    physical_augmentation = (
        *(augmentation[name] for name in physical_fields),
        tuple(augmentation["original_cube_names"]),
        tuple(augmentation["distractor_names"]),
        recovery["selected_attempt"],
    )
    return _ValidatedRecoveryBundle(
        ep_meta=_freeze_json(copy.deepcopy(ep_meta)),
        metadata=_freeze_json(copy.deepcopy(recovery)),
        model_xml=model_xml,
        initial_state=initial_state,
        input_hashes=_freeze_json(copy.deepcopy(input_hashes)),
        output_hashes=_freeze_json(copy.deepcopy(actual_bundle_hashes)),
        payload_hashes=_freeze_json(
            {
                name: actual_payload_hashes[name]
                for name in ("model.xml.gz", "states.npz")
            }
        ),
        physical_augmentation=physical_augmentation,
    )


def _validate_comparison_png(data: bytes, label: str) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to validate Sweeten comparison PNGs"
        ) from error

    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("unexpected image format")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("unexpected image format")
            image.load()
            width, height = image.size
            pixel_data = image.tobytes()
            if (
                image.mode != "RGB"
                or width <= 0
                or height <= 0
                or len(pixel_data) != width * height * 3
            ):
                raise ValueError("invalid RGB image data")
    except (OSError, SyntaxError, ValueError) as error:
        raise ValueError(f"{label}: invalid comparison PNG") from error


def _validate_recovered_sweeten_descriptors(
    root: pathlib.Path,
    root_fd: int,
    extras_fd: int,
    comparisons_fd: int,
    *,
    mapping_rows=None,
    mapping_sha256: str | None = None,
) -> tuple[RecordedSceneReference, ...]:
    extras = root / "extras"
    report_path = root / "recovery_report.json"
    completion_path = root / "RECOVERY_COMPLETE.json"
    report_bytes = _read_regular_file_bytes_at(
        root_fd,
        "recovery_report.json",
        "Sweeten recovery report",
    )
    report = _parse_json_object(
        report_bytes,
        report_path,
        "Sweeten recovery report",
    )
    completion_bytes = _read_regular_file_bytes_at(
        root_fd,
        "RECOVERY_COMPLETE.json",
        "Sweeten recovery completion marker",
    )
    completion = _parse_json_object(
        completion_bytes,
        completion_path,
        "Sweeten recovery completion marker",
    )

    if mapping_rows is None:
        mapping_rows = SWEETEN_RECOVERED_MAPPING
    if mapping_sha256 is None:
        mapping_sha256 = _SWEETEN_RECOVERED_MAPPING_SHA256
    mapping_rows = tuple(tuple(row) for row in mapping_rows)
    mapping_bytes = json.dumps(
        mapping_rows,
        separators=(",", ":"),
    ).encode("ascii")
    if _sha256_bytes(mapping_bytes) != mapping_sha256:
        raise RuntimeError("evaluator Sweeten mapping digest is inconsistent")
    mappings = tuple(
        _SweetenMapping(*row) for row in mapping_rows
    )
    combined_ids = [item.combined_episode for item in mappings]
    source_ids = {item.source_episode for item in mappings}
    expected_completion = {
        "schema_version": 1,
        "status": "complete",
        "bundle_count": len(mappings),
        "source_count": len(source_ids),
        "recovery_report_sha256": _sha256_bytes(report_bytes),
    }
    if (
        type(completion.get("schema_version")) is not int
        or completion != expected_completion
    ):
        raise ValueError("Sweeten recovery completion marker is inconsistent")
    expected_report_keys = {
        "schema_version",
        "generation_commit",
        "global_seed",
        "source_root",
        "combined_training_root",
        "output_root",
        "combined_episode_ids",
        "attempts",
        "runtime_versions",
        "augmentation_archive",
        "calibration",
        "codec_noise_envelope",
        "sources",
        "bundles",
    }
    _canonical_recovery_root_locators(report.get("source_root"))
    _canonical_recovery_root_locators(report.get("combined_training_root"))
    # output_root records where the immutable recovery bundle was generated.
    # It is provenance, not a required extraction path for evaluation.
    _canonical_recovery_root_locators(report.get("output_root"))
    if (
        set(report) != expected_report_keys
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("generation_commit") != _SWEETEN_GENERATION_COMMIT
        or type(report.get("global_seed")) is not int
        or report["global_seed"] != _SWEETEN_GLOBAL_SEED
        or not isinstance(report.get("combined_episode_ids"), list)
        or any(
            type(value) is not int
            for value in report["combined_episode_ids"]
        )
        or report.get("combined_episode_ids") != combined_ids
        or not isinstance(report.get("attempts"), list)
        or any(type(value) is not int for value in report["attempts"])
        or report.get("attempts") != list(_RECOVERY_ATTEMPTS)
    ):
        raise ValueError("Sweeten recovery report metadata is inconsistent")
    report_attempts = set(report["attempts"])
    report_runtime_versions = _validate_runtime_versions(
        report.get("runtime_versions"),
        "Sweeten report runtime versions",
    )
    _validate_augmentation_archive(
        report.get("augmentation_archive"),
        report_runtime_versions,
    )
    report_envelope = _recovery_number(
        report.get("codec_noise_envelope"),
        "Sweeten report codec noise envelope",
    )
    if report_envelope > _RECOVERY_MAX_CODEC_NOISE_ENVELOPE:
        raise ValueError("Sweeten report codec noise envelope is invalid")
    calibration = _validate_calibration(
        report.get("calibration"),
        report_envelope,
    )

    bundle_rows = report.get("bundles")
    source_rows = report.get("sources")
    if (
        not isinstance(bundle_rows, list)
        or len(bundle_rows) != len(mappings)
        or not isinstance(source_rows, list)
        or len(source_rows) != len(source_ids)
    ):
        raise ValueError("Sweeten recovery report counts are inconsistent")
    if any(not isinstance(row, dict) for row in bundle_rows + source_rows):
        raise ValueError("Sweeten recovery report rows are invalid")
    if any(
        set(row) != {
            "combined_episode",
            "source_episode",
            "goal_count",
            "path",
            "input_hashes",
            "output_hashes",
        }
        for row in bundle_rows
    ) or any(
        set(row) != {
            "source_episode",
            "selected_attempt",
            "layout_attempt",
            "combined_episodes",
            "rejected_attempts",
            "comparison",
            "comparison_sha256",
        }
        for row in source_rows
    ):
        raise ValueError("Sweeten recovery report row schemas are invalid")
    if any(
        type(row.get("combined_episode")) is not int
        or type(row.get("source_episode")) is not int
        or type(row.get("goal_count")) is not int
        for row in bundle_rows
    ) or any(
        type(row.get("source_episode")) is not int
        for row in source_rows
    ):
        raise ValueError("Sweeten recovery report row identifiers are invalid")
    rows_by_episode = {
        row.get("combined_episode"): row for row in bundle_rows
    }
    sources_by_episode = {
        row.get("source_episode"): row for row in source_rows
    }
    if (
        set(rows_by_episode) != set(combined_ids)
        or len(rows_by_episode) != len(bundle_rows)
        or set(sources_by_episode) != source_ids
        or len(sources_by_episode) != len(source_rows)
    ):
        raise ValueError("Sweeten recovery report mappings are inconsistent")

    expected_bundle_names = {
        f"episode_{episode:06d}" for episode in combined_ids
    }
    if set(os.listdir(extras_fd)) != expected_bundle_names:
        raise ValueError("Sweeten recovered extras inventory is inconsistent")
    expected_comparison_names = {
        f"source_{source:06d}.png" for source in source_ids
    }
    if (
        set(os.listdir(comparisons_fd)) != expected_comparison_names
    ):
        raise ValueError("Sweeten comparison inventory is inconsistent")

    combined_by_source = {
        source: [
            item.combined_episode
            for item in mappings
            if item.source_episode == source
        ]
        for source in source_ids
    }
    for source_episode, row in sources_by_episode.items():
        comparison_name = f"source_{source_episode:06d}.png"
        comparison_bytes = _read_regular_file_bytes_at(
            comparisons_fd,
            comparison_name,
            f"Sweeten source {source_episode} comparison",
        )
        if (
            row.get("combined_episodes")
            != combined_by_source[source_episode]
            or any(
                type(value) is not int
                for value in row.get("combined_episodes", ())
            )
            or row.get("comparison")
            != f"comparisons/{comparison_name}"
            or row.get("comparison_sha256")
            != _sha256_bytes(comparison_bytes)
            or type(row.get("selected_attempt")) is not int
            or row["selected_attempt"] not in report_attempts
            or type(row.get("layout_attempt")) is not int
            or row["layout_attempt"] < 0
        ):
            raise ValueError(
                f"Sweeten source {source_episode} report row is inconsistent"
            )
        _validate_comparison_png(
            comparison_bytes,
            f"Sweeten source {source_episode}",
        )
        _validate_rejected_attempts(
            row.get("rejected_attempts"),
            report_attempts,
        )

    references = []
    source_input_hashes = {}
    physical_scenes = {}
    recoveries_by_episode = {}
    for mapping in mappings:
        episode_name = f"episode_{mapping.combined_episode:06d}"
        report_entry = rows_by_episode[mapping.combined_episode]
        source_entry = sources_by_episode[mapping.source_episode]
        recovery_path = extras / episode_name / "recovery.json"
        with _open_directory_fd(
            episode_name,
            f"{episode_name} bundle",
            dir_fd=extras_fd,
        ) as episode_fd:
            validated_bundle = _validate_recovery_bundle(
                episode_fd,
                episode_name=episode_name,
                combined_episode=mapping.combined_episode,
                source_episode=mapping.source_episode,
                goal_count=mapping.goal_count,
                report_entry=report_entry,
                source_entry=source_entry,
                report_envelope=report_envelope,
                report_attempts=report_attempts,
                report_runtime_versions=report_runtime_versions,
            )
        recovery_metadata = _thaw_json(validated_bundle.metadata)
        recoveries_by_episode[mapping.combined_episode] = recovery_metadata
        source_hashes = {
            name: recovery_metadata["input_hashes"][name]
            for name in (
                "source/ep_meta.json",
                "source/model.xml.gz",
                "source/states.npz",
            )
        }
        previous_source_hashes = source_input_hashes.setdefault(
            mapping.source_episode,
            source_hashes,
        )
        if previous_source_hashes != source_hashes:
            raise ValueError(
                f"source input hashes differ for source episode "
                f"{mapping.source_episode}"
            )
        previous_physical = physical_scenes.setdefault(
            mapping.source_episode,
            validated_bundle,
        )
        if (
            previous_physical.model_xml != validated_bundle.model_xml
            or previous_physical.initial_state.dtype
            != validated_bundle.initial_state.dtype
            or previous_physical.initial_state.shape
            != validated_bundle.initial_state.shape
            or not np.array_equal(
                previous_physical.initial_state,
                validated_bundle.initial_state,
            )
            or previous_physical.payload_hashes
            != validated_bundle.payload_hashes
            or previous_physical.physical_augmentation
            != validated_bundle.physical_augmentation
        ):
            raise ValueError(
                f"physical scene differs for source episode "
                f"{mapping.source_episode}"
            )
        references.append(
            RecordedSceneReference(
                dataset_path=root,
                episode_index=mapping.combined_episode,
                combined_episode=mapping.combined_episode,
                source_episode=mapping.source_episode,
                goal_count=mapping.goal_count,
                recovery_metadata=validated_bundle.metadata,
                recovery_path=recovery_path,
                recovery_report_entry=_freeze_json(
                    copy.deepcopy(report_entry)
                ),
                validated_payload=validated_bundle,
            )
        )
    if 207 in recoveries_by_episode:
        source31 = recoveries_by_episode[207]
        source31_augmentation = source31["augmentation"]
        if (
            source31_augmentation["global_seed"] != _SWEETEN_GLOBAL_SEED
            or source31_augmentation["derived_seed"] != _SOURCE31_DERIVED_SEED
            or source31_augmentation["original_cube_count"] != 3
            or source31_augmentation["original_cube_names"]
            != list(_SOURCE31_ORIGINAL_CUBE_NAMES)
            or source31_augmentation["total_cube_count"] != 4
            or source31_augmentation["added_distractor_count"] != 1
            or source31_augmentation["distractor_names"] != ["sugar_aug0"]
            or source31_augmentation["layout_attempt"] != 0
            or source31["layout_attempt"] != 0
            or source31["selected_attempt"] != 0
            or source31["selected_canonical_source"] is not False
            or source31["generation_commit"] != _SWEETEN_GENERATION_COMMIT
        ):
            raise ValueError("Sweeten source 31 signature is invalid")
        if calibration.get("method") != "explicit_override":
            if (
                calibration["selected_attempt"]
                != source31["selected_attempt"]
                or calibration["layout_attempt"] != source31["layout_attempt"]
                or calibration["original_cube_names"]
                != source31_augmentation["original_cube_names"]
                or not _same_score(
                    calibration["winning_aggregate_score"],
                    source31["best_aggregate_score"],
                )
                or any(
                    not _same_score(
                        calibration["winning_per_camera"][camera],
                        source31["selected_per_camera"][camera],
                    )
                    for camera in _RECOVERY_CAMERA_NAMES
                )
            ):
                raise ValueError(
                    "Sweeten calibration evidence differs from source 31 "
                    "recovery"
                )
    return tuple(references)


def _validate_recovered_sweeten_root(
    root: pathlib.Path,
    *,
    mapping_rows=None,
    mapping_sha256: str | None = None,
) -> tuple[RecordedSceneReference, ...]:
    expected_top_level = {
        "extras",
        "comparisons",
        "recovery_report.json",
        "RECOVERY_COMPLETE.json",
    }
    with _open_directory_fd(
        root,
        "Sweeten recovered root",
    ) as root_fd:
        if set(os.listdir(root_fd)) != expected_top_level:
            raise ValueError(
                "Sweeten recovered root has an unexpected inventory"
            )
        with _open_directory_fd(
            "extras",
            "Sweeten recovered extras",
            dir_fd=root_fd,
        ) as extras_fd:
            with _open_directory_fd(
                "comparisons",
                "Sweeten recovery comparisons",
                dir_fd=root_fd,
            ) as comparisons_fd:
                return _validate_recovered_sweeten_descriptors(
                    root,
                    root_fd,
                    extras_fd,
                    comparisons_fd,
                    mapping_rows=mapping_rows,
                    mapping_sha256=mapping_sha256,
                )


_RETRIEVE_ENV_NAMES = frozenset(
    env_name
    for group_name in ("retrieve_fruit", "retrieve_oil")
    for env_name, _ in TASK_GROUPS[group_name]
)
_RETRIEVE_MANIFEST_KEYS = {
    "schema_version",
    "dataset",
    "source_kind",
    "source_root",
    "combined_root",
    "mapping_method",
    "states_hash_encoding",
    "rows",
}
_RETRIEVE_ROW_KEYS = {
    "combined_episode",
    "combined_parquet",
    "combined_parquet_sha256",
    "environment_variant",
    "source_file",
    "source_file_sha256",
    "source_episode",
    "source_demo_key",
    "action_length",
    "layout_id",
    "style_id",
    "source_hashes",
}
_RETRIEVE_SOURCE_HASH_KEYS = {
    "ep_meta_sha256",
    "model_xml_sha256",
    "states_sha256",
}


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _reject_legacy_retrieve_path(path, label: str) -> None:
    if "remembench_im128" in str(path).lower():
        raise ValueError(
            f"{label} uses rejected legacy remembench_im128 data: {path}"
        )


def _reject_legacy_wash_path(path, label: str) -> None:
    if "remembench_im128" in str(path).lower():
        raise ValueError(
            f"{label} uses rejected legacy remembench_im128 data: {path}"
        )


def _validate_recorded_scene_payload(
    episode_dir: pathlib.Path,
    task_label: str,
) -> _ValidatedRecordedScenePayload:
    payload_bytes = {
        name: (episode_dir / name).read_bytes()
        for name in _RECORDED_SCENE_FILES
    }
    try:
        ep_meta = json.loads(payload_bytes["ep_meta.json"])
        model_xml = gzip.decompress(
            payload_bytes["model.xml.gz"]
        ).decode("utf-8")
        with np.load(
            io.BytesIO(payload_bytes["states.npz"]),
            allow_pickle=False,
        ) as archive:
            states = np.asarray(archive["states"])
    except (
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{episode_dir.name}: invalid {task_label} recorded scene payload"
        ) from error
    if (
        not isinstance(ep_meta, dict)
        or not model_xml
        or states.ndim != 2
        or states.shape[0] < 1
    ):
        raise ValueError(
            f"{episode_dir.name}: invalid {task_label} recorded scene contract"
        )
    file_hashes = types.MappingProxyType(
        {
            name: _sha256_bytes(payload_bytes[name])
            for name in _RECORDED_SCENE_FILES
        }
    )
    return _ValidatedRecordedScenePayload(
        ep_meta_json=payload_bytes["ep_meta.json"],
        model_xml=model_xml.encode("utf-8"),
        states_npz=payload_bytes["states.npz"],
        file_hashes=file_hashes,
    )


def _validate_wash_recorded_payload(
    episode_dir: pathlib.Path,
) -> _ValidatedRecordedScenePayload:
    return _validate_recorded_scene_payload(episode_dir, "Wash")


def _wash_recorded_scene_references(
    env_name: str,
    args: Args,
) -> tuple[RecordedSceneReference, ...]:
    source_root = pathlib.Path(args.wash_scene_root)
    _reject_legacy_wash_path(source_root, "Wash recorded scene root")
    try:
        resolved_root = source_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Wash recorded scene root is missing: {source_root}"
        ) from error
    if not resolved_root.is_dir():
        raise ValueError(
            f"Wash recorded scene root is not a directory: {source_root}"
        )

    dataset_path = resolved_root / env_name / "lerobot"
    extras_path = dataset_path / "extras"
    try:
        resolved_dataset = dataset_path.resolve(strict=True)
        resolved_extras = extras_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{env_name}: Wash recorded LeRobot extras are missing: "
            f"{extras_path}"
        ) from error
    if (
        not resolved_dataset.is_dir()
        or not resolved_extras.is_dir()
        or not resolved_dataset.is_relative_to(resolved_root)
        or not resolved_extras.is_relative_to(resolved_dataset)
    ):
        raise ValueError(
            f"{env_name}: Wash recorded LeRobot path is invalid"
        )

    expected_count = _WASH_SOURCE_EPISODE_COUNTS[env_name]
    expected_names = {
        f"episode_{episode_index:06d}"
        for episode_index in range(expected_count)
    }
    actual_names = {
        entry.name
        for entry in resolved_extras.iterdir()
        if entry.name.startswith("episode_")
    }
    if actual_names != expected_names:
        raise ValueError(
            f"{env_name}: Wash recorded episode inventory is invalid; "
            f"expected {expected_count}, found {len(actual_names)}"
        )

    references = []
    for episode_index in range(expected_count):
        episode_name = f"episode_{episode_index:06d}"
        episode_dir = resolved_extras / episode_name
        if episode_dir.is_symlink() or not episode_dir.is_dir():
            raise ValueError(
                f"{env_name}: Wash recorded episode is invalid: "
                f"{episode_dir}"
            )
        missing = [
            str(episode_dir / name)
            for name in _RECORDED_SCENE_FILES
            if not (episode_dir / name).is_file()
            or (episode_dir / name).is_symlink()
        ]
        if missing:
            raise FileNotFoundError(
                f"{episode_name}: missing Wash recorded scene files: "
                f"{missing}"
            )
        validated_payload = _validate_wash_recorded_payload(episode_dir)
        references.append(
            RecordedSceneReference(
                dataset_path=resolved_dataset,
                episode_index=episode_index,
                source_episode=episode_index,
                source_root=resolved_root,
                environment_variant=env_name,
                source_hashes=validated_payload.file_hashes,
                validated_scene_payload=validated_payload,
            )
        )
    return tuple(references)


def _recorded_extra_scene_references(
    env_name: str,
    dataset_path: pathlib.Path,
    source_ids: tuple[int, ...] | None = None,
    *,
    combined_episode_start: int | None = None,
) -> tuple[RecordedSceneReference, ...]:
    try:
        resolved_dataset = dataset_path.resolve(strict=True)
        resolved_extras = (resolved_dataset / "extras").resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{env_name}: recorded LeRobot extras are missing: "
            f"{dataset_path / 'extras'}"
        ) from error
    if (
        not resolved_dataset.is_dir()
        or not resolved_extras.is_dir()
        or not resolved_extras.is_relative_to(resolved_dataset)
    ):
        raise ValueError(f"{env_name}: recorded LeRobot path is invalid")

    references = []
    if source_ids is None:
        source_ids = RECORDED_SOURCE_EPISODES[env_name]
    for selection_index, episode_index in enumerate(source_ids):
        episode_name = f"episode_{episode_index:06d}"
        episode_dir = resolved_extras / episode_name
        if episode_dir.is_symlink() or not episode_dir.is_dir():
            raise FileNotFoundError(
                f"{env_name}: recorded episode is missing or invalid: "
                f"{episode_dir}"
            )
        missing = [
            str(episode_dir / name)
            for name in _RECORDED_SCENE_FILES
            if not (episode_dir / name).is_file()
            or (episode_dir / name).is_symlink()
        ]
        if missing:
            raise FileNotFoundError(
                f"{episode_name}: missing {env_name} recorded scene files: "
                f"{missing}"
            )
        validated_payload = _validate_recorded_scene_payload(
            episode_dir,
            env_name,
        )
        references.append(
            RecordedSceneReference(
                dataset_path=resolved_dataset,
                episode_index=episode_index,
                combined_episode=(
                    None
                    if combined_episode_start is None
                    else combined_episode_start + selection_index
                ),
                source_episode=episode_index,
                source_root=resolved_dataset,
                source_demo_key=episode_name,
                environment_variant=env_name,
                source_hashes=validated_payload.file_hashes,
                validated_scene_payload=validated_payload,
            )
        )
    return tuple(references)


def _parse_episode_ids(value: str, task_label: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise ValueError(f"{task_label} episode selection must be text")
    if not value:
        return ()
    tokens = value.split(",")
    if any(not re.fullmatch(r"\d+", token) for token in tokens):
        raise ValueError(f"{task_label} episode selection is invalid")
    episode_ids = tuple(int(token) for token in tokens)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError(f"{task_label} episode selection has duplicates")
    return episode_ids


def _rinse_profile_episode_ids(args: Args) -> tuple[int, ...]:
    canonical = RECORDED_SOURCE_EPISODES["RinseCuttingBoard"]
    selected = _parse_episode_ids(
        args.rinse_episode_ids,
        "Rinse",
    )
    if args.rinse_scene_profile == "canonical_full50":
        if selected and selected != canonical:
            raise ValueError(
                "Rinse canonical_full50 profile requires the full canonical "
                "episode selection"
            )
        return canonical
    if args.rinse_scene_profile == "postfix_0_3":
        if selected != _RINSE_POSTFIX_EPISODES:
            raise ValueError(
                "Rinse postfix_0_3 profile requires episode selection 0,3"
            )
        if not set(selected).issubset(canonical):
            raise ValueError(
                "Rinse postfix profile contains a noncanonical episode"
            )
        return selected
    raise ValueError(
        f"unsupported Rinse scene profile: {args.rinse_scene_profile}"
    )


def _sweeten_profile_references(
    root: pathlib.Path,
    args: Args,
) -> tuple[RecordedSceneReference, ...]:
    selected = _parse_episode_ids(
        args.sweeten_episode_ids,
        "Sweeten",
    )
    if args.sweeten_scene_profile == "full50":
        expected = tuple(
            row[0] for row in SWEETEN_RECOVERED_MAPPING
        )
        if selected and selected != expected:
            raise ValueError(
                "Sweeten full50 profile requires the full canonical episode "
                "selection"
            )
        return _validate_recovered_sweeten_root(root)
    if args.sweeten_scene_profile == "full25":
        expected = tuple(
            row[0] for row in _SWEETEN_FULL25_MAPPING
        )
        if selected and selected != expected:
            raise ValueError(
                "Sweeten full25 profile requires the full canonical episode "
                "selection"
            )
        return _validate_recovered_sweeten_root(
            root,
            mapping_rows=_SWEETEN_FULL25_MAPPING,
            mapping_sha256=_SWEETEN_FULL25_MAPPING_SHA256,
        )
    if args.sweeten_scene_profile == "part_b_210_217":
        if selected != _SWEETEN_PART_B_SELECTED_EPISODES:
            raise ValueError(
                "Sweeten part_b_210_217 profile requires episode selection "
                "210,211"
            )
        references = _validate_recovered_sweeten_root(
            root,
            mapping_rows=_SWEETEN_PART_B_MAPPING,
            mapping_sha256=_SWEETEN_PART_B_MAPPING_SHA256,
        )
        references_by_episode = {
            reference.combined_episode: reference
            for reference in references
        }
        if set(references_by_episode) != {
            row[0] for row in _SWEETEN_PART_B_MAPPING
        }:
            raise ValueError(
                "Sweeten Part-B validated reference inventory is invalid"
            )
        return tuple(
            references_by_episode[episode_id]
            for episode_id in selected
        )
    raise ValueError(
        f"unsupported Sweeten scene profile: {args.sweeten_scene_profile}"
    )


def _retrieve_environment_for_episode(combined_episode: int) -> str:
    if 0 <= combined_episode < 25:
        return "MemFruitInSinkLeftFar"
    if combined_episode < 50:
        return "MemFruitInSinkRightFar"
    if combined_episode < 63:
        return "MemRetrieveOilsFromCounterLL"
    if combined_episode < 76:
        return "MemRetrieveOilsFromCounterLR"
    if combined_episode < 88:
        return "MemRetrieveOilsFromCounterRL"
    if combined_episode < 100:
        return "MemRetrieveOilsFromCounterRR"
    raise ValueError(
        f"invalid Retrieve combined episode: {combined_episode}"
    )


def _retrieve_manifest_rows(args: Args) -> tuple[dict, ...]:
    manifest_path = pathlib.Path(args.retrieve_lineage_manifest)
    source_root = pathlib.Path(args.retrieve_source_root)
    _reject_legacy_retrieve_path(manifest_path, "Retrieve lineage manifest")
    _reject_legacy_retrieve_path(source_root, "Retrieve source root")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Retrieve lineage manifest is missing: {manifest_path}"
        ) from error
    if _sha256_bytes(manifest_bytes) != _RETRIEVE_LINEAGE_MANIFEST_SHA256:
        raise ValueError("Retrieve lineage manifest digest is invalid")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Retrieve lineage manifest is invalid JSON") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _RETRIEVE_MANIFEST_KEYS
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("dataset") != "Combined350-v3-oilv4"
        or manifest.get("source_kind")
        != "robocasa-1.0.1-patched-replay-build-input"
        or manifest.get("mapping_method")
        != "exact-reordered-action-tensor-sha256"
        or manifest.get("states_hash_encoding")
        != "numpy-c-order-raw-dtype-bytes"
        or not isinstance(manifest.get("source_root"), str)
        or not isinstance(manifest.get("combined_root"), str)
        or not isinstance(manifest.get("rows"), list)
    ):
        raise ValueError("Retrieve lineage manifest contract is invalid")
    _reject_legacy_retrieve_path(
        manifest["source_root"],
        "Retrieve manifest source root",
    )
    rows = manifest["rows"]
    if len(rows) != 100 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Retrieve lineage manifest must contain 100 rows")
    rows_by_episode = {
        row.get("combined_episode"): row
        for row in rows
        if type(row.get("combined_episode")) is int
    }
    if (
        len(rows_by_episode) != 100
        or set(rows_by_episode) != set(range(100))
    ):
        raise ValueError("Retrieve combined episode lineage is incomplete")
    try:
        resolved_root = source_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Retrieve source root is missing: {source_root}"
        ) from error
    if not resolved_root.is_dir():
        raise ValueError(
            f"Retrieve source root is not a directory: {source_root}"
        )

    checked_source_files = {}
    validated_rows = []
    for combined_episode in range(100):
        row = rows_by_episode[combined_episode]
        hashes = row.get("source_hashes")
        source_file = row.get("source_file")
        relative_source = pathlib.PurePosixPath(
            source_file if isinstance(source_file, str) else ""
        )
        source_demo_key = row.get("source_demo_key")
        expected_variant = _retrieve_environment_for_episode(
            combined_episode
        )
        if (
            set(row) != _RETRIEVE_ROW_KEYS
            or row.get("environment_variant") != expected_variant
            or relative_source.is_absolute()
            or ".." in relative_source.parts
            or relative_source.parts[:1] != (expected_variant,)
            or relative_source.name != "demo.hdf5"
            or type(row.get("source_episode")) is not int
            or row["source_episode"] < 0
            or source_demo_key != f"demo_{row['source_episode']}"
            or type(row.get("action_length")) is not int
            or row["action_length"] < 1
            or type(row.get("layout_id")) is not int
            or type(row.get("style_id")) is not int
            or row.get("combined_parquet")
            != f"episode_{combined_episode:06d}.parquet"
            or not _is_sha256(row.get("combined_parquet_sha256"))
            or not _is_sha256(row.get("source_file_sha256"))
            or not isinstance(hashes, dict)
            or set(hashes) != _RETRIEVE_SOURCE_HASH_KEYS
            or any(not _is_sha256(value) for value in hashes.values())
        ):
            raise ValueError(
                f"Retrieve lineage row {combined_episode} is invalid"
            )
        dataset_path = source_root.joinpath(*relative_source.parts)
        try:
            resolved_dataset = dataset_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Retrieve source artifact is missing: {dataset_path}"
            ) from error
        if not resolved_dataset.is_relative_to(resolved_root):
            raise ValueError(
                f"Retrieve source artifact escapes source root: {dataset_path}"
            )
        _reject_legacy_retrieve_path(
            resolved_dataset,
            "Retrieve source artifact",
        )
        previous_hash = checked_source_files.setdefault(
            resolved_dataset,
            row["source_file_sha256"],
        )
        if previous_hash != row["source_file_sha256"]:
            raise ValueError(
                f"Retrieve source file hash differs across lineage rows: "
                f"{dataset_path}"
            )
        validated_rows.append(
            {
                **row,
                "dataset_path": resolved_dataset,
                "source_root_path": resolved_root,
            }
        )
    for source_path, expected_hash in checked_source_files.items():
        if _sha256_file(source_path) != expected_hash:
            raise ValueError(
                f"Retrieve source artifact hash is invalid: {source_path}"
            )
    return tuple(validated_rows)


def recorded_scene_references(
    env_name: str,
    args: Args,
) -> tuple[RecordedSceneReference, ...]:
    if env_name in _RETRIEVE_ENV_NAMES:
        return tuple(
            RecordedSceneReference(
                dataset_path=row["dataset_path"],
                episode_index=row["combined_episode"],
                combined_episode=row["combined_episode"],
                source_episode=row["source_episode"],
                source_root=row["source_root_path"],
                source_demo_key=row["source_demo_key"],
                environment_variant=row["environment_variant"],
                source_hashes=types.MappingProxyType(
                    dict(row["source_hashes"])
                ),
                source_file_sha256=row["source_file_sha256"],
                source_layout_id=row["layout_id"],
                source_style_id=row["style_id"],
            )
            for row in _retrieve_manifest_rows(args)
            if row["environment_variant"] == env_name
        )
    if env_name in _WASH_ENV_NAMES:
        return _wash_recorded_scene_references(env_name, args)
    if env_name == "SweetenHotChocolate":
        return _sweeten_profile_references(
            pathlib.Path(args.sweeten_recovered_root),
            args,
        )
    if env_name in NEW5_RECORDED_SCENE_SOURCES:
        relative_dataset_path, source_ids = NEW5_RECORDED_SCENE_SOURCES[
            env_name
        ]
        return _recorded_extra_scene_references(
            env_name,
            pathlib.Path(args.new5_scene_root) / relative_dataset_path,
            source_ids,
            combined_episode_start=_NEW5_COMBINED_EPISODE_START[env_name],
        )
    dataset_paths = {
        "PanTransfer": args.pan_transfer_dataset,
        "RinseCuttingBoard": args.rinse_cutting_board_dataset,
    }
    source_ids = RECORDED_SOURCE_EPISODES.get(env_name, ())
    if not source_ids:
        return ()
    dataset_path = pathlib.Path(dataset_paths[env_name])
    if env_name == "RinseCuttingBoard":
        source_ids = _rinse_profile_episode_ids(args)
    return _recorded_extra_scene_references(
        env_name,
        dataset_path,
        tuple(source_ids),
    )


def load_recorded_scene(
    reference: RecordedSceneReference,
) -> DemoInitialization:
    retained_scene_variants = (
        set(_WASH_ENV_NAMES)
        | set(RECORDED_SOURCE_EPISODES)
        | set(NEW5_RECORDED_SCENE_SOURCES)
    )
    if reference.environment_variant in retained_scene_variants:
        episode_name = f"episode_{reference.episode_index:06d}"
        payload = reference.validated_scene_payload
        is_recorded_extra = (
            reference.environment_variant in RECORDED_SOURCE_EPISODES
            or reference.environment_variant in NEW5_RECORDED_SCENE_SOURCES
        )
        if (
            reference.source_root is None
            or reference.source_episode != reference.episode_index
            or reference.source_hashes is None
            or payload is None
            or dict(reference.source_hashes)
            != dict(payload.file_hashes)
            or (
                is_recorded_extra
                and (
                    reference.source_root != reference.dataset_path
                    or reference.source_demo_key != episode_name
                )
            )
        ):
            raise ValueError(
                f"{episode_name}: retained recorded scene lineage is incomplete"
            )
        try:
            ep_meta = json.loads(payload.ep_meta_json)
            model_file = payload.model_xml.decode("utf-8")
            with np.load(
                io.BytesIO(payload.states_npz),
                allow_pickle=False,
            ) as archive:
                states = np.asarray(
                    archive["states"],
                    dtype=np.float64,
                )
        except (
            KeyError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"{episode_name}: retained recorded payload is invalid"
            ) from error
        if (
            not isinstance(ep_meta, dict)
            or not model_file
            or states.ndim != 2
            or states.shape[0] < 1
        ):
            raise ValueError(
                f"{episode_name}: retained recorded scene contract is invalid"
            )
        return DemoInitialization(
            dataset_path=reference.dataset_path,
            demo_key=episode_name,
            ep_meta=ep_meta,
            model_file=model_file,
            initial_state=states[0].copy(),
            action_length=int(states.shape[0]),
            combined_episode=reference.combined_episode,
            source_episode=reference.source_episode,
            source_root=reference.source_root,
            source_demo_key=(
                reference.source_demo_key or episode_name
            ),
            environment_variant=reference.environment_variant,
            source_hashes=reference.source_hashes,
            recorded_states=np.asarray(states, dtype=np.float64),
        )

    if reference.source_demo_key is not None:
        if (
            reference.combined_episode != reference.episode_index
            or reference.source_root is None
            or reference.source_episode is None
            or reference.source_demo_key
            != f"demo_{reference.source_episode}"
            or reference.environment_variant not in _RETRIEVE_ENV_NAMES
            or reference.source_hashes is None
            or reference.source_file_sha256 is None
            or reference.source_layout_id is None
            or reference.source_style_id is None
        ):
            raise ValueError("Retrieve recorded scene lineage is incomplete")
        _reject_legacy_retrieve_path(
            reference.dataset_path,
            "Retrieve recorded scene",
        )
        if _sha256_file(reference.dataset_path) != reference.source_file_sha256:
            raise ValueError("Retrieve source artifact hash changed")

        import h5py

        with h5py.File(reference.dataset_path, "r") as dataset:
            demo_path = f"data/{reference.source_demo_key}"
            if demo_path not in dataset:
                raise ValueError(
                    f"Retrieve source demo is missing: {demo_path}"
                )
            source_demo = dataset[demo_path]
            ep_meta_value = source_demo.attrs.get("ep_meta")
            model_value = source_demo.attrs.get("model_file")
            if isinstance(ep_meta_value, str):
                ep_meta_bytes = ep_meta_value.encode("utf-8")
            elif isinstance(ep_meta_value, bytes):
                ep_meta_bytes = ep_meta_value
            else:
                raise ValueError("Retrieve source ep_meta is not text")
            if isinstance(model_value, str):
                model_bytes = model_value.encode("utf-8")
            elif isinstance(model_value, bytes):
                model_bytes = model_value
            else:
                raise ValueError("Retrieve source model XML is not text")
            if (
                "states" not in source_demo
                or "actions" not in source_demo
                or "policy_mode" not in source_demo
            ):
                raise ValueError("Retrieve source demo payload is incomplete")
            states = np.ascontiguousarray(source_demo["states"][...])
            actions = np.ascontiguousarray(source_demo["actions"][...])
            policy_mode = np.asarray(source_demo["policy_mode"][...]).reshape(-1)
            policy_rows = np.flatnonzero(policy_mode == 0)
            if not len(policy_rows):
                raise ValueError(
                    "Retrieve source demo has no policy-mode transition"
                )
            recorded_transition = int(policy_rows[0])
            action_length = int(actions.shape[0])

        expected_hashes = dict(reference.source_hashes)
        actual_hashes = {
            "ep_meta_sha256": _sha256_bytes(ep_meta_bytes),
            "model_xml_sha256": _sha256_bytes(model_bytes),
            "states_sha256": _sha256_bytes(states.tobytes()),
        }
        if actual_hashes != expected_hashes:
            raise ValueError("Retrieve source demo hashes are invalid")
        try:
            ep_meta = json.loads(ep_meta_bytes)
            model_file = model_bytes.decode("utf-8")
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Retrieve source payload is invalid") from error
        if (
            not isinstance(ep_meta, dict)
            or ep_meta.get("layout_id") != reference.source_layout_id
            or ep_meta.get("style_id") != reference.source_style_id
            or not model_file
            or states.ndim != 2
            or states.shape[0] < 1
            or action_length != states.shape[0]
            or actions.ndim != 2
            or actions.shape[1] != 12
            or policy_mode.shape != (action_length,)
            or not 0 < recorded_transition < action_length
        ):
            raise ValueError("Retrieve source scene contract is invalid")
        return DemoInitialization(
            dataset_path=reference.dataset_path,
            demo_key=reference.source_demo_key,
            ep_meta=ep_meta,
            model_file=model_file,
            initial_state=np.asarray(
                states[0],
                dtype=np.float64,
            ).copy(),
            action_length=action_length,
            combined_episode=reference.combined_episode,
            source_episode=reference.source_episode,
            source_root=reference.source_root,
            source_demo_key=reference.source_demo_key,
            environment_variant=reference.environment_variant,
            source_hashes=reference.source_hashes,
            source_file_sha256=reference.source_file_sha256,
            source_layout_id=reference.source_layout_id,
            source_style_id=reference.source_style_id,
            recorded_states=np.asarray(states, dtype=np.float64),
            recorded_actions=np.asarray(actions, dtype=np.float32),
            recorded_transition=recorded_transition,
        )

    episode_name = f"episode_{reference.episode_index:06d}"
    episode_dir = reference.dataset_path / "extras" / episode_name
    if reference.combined_episode is not None:
        if (
            reference.combined_episode != reference.episode_index
            or reference.source_episode is None
            or reference.goal_count is None
            or reference.recovery_metadata is None
            or reference.recovery_path != episode_dir / "recovery.json"
            or reference.recovery_report_entry is None
            or reference.validated_payload is None
        ):
            raise ValueError(f"{episode_name}: incomplete recovery lineage")
        validated_bundle = reference.validated_payload
        recovery_metadata = validated_bundle.metadata
        if recovery_metadata != reference.recovery_metadata:
            raise ValueError(
                f"{episode_name}: recovery metadata changed after selection"
            )
        try:
            model_file = validated_bundle.model_xml.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{episode_name}: recovered model is not UTF-8"
            ) from error
        return DemoInitialization(
            dataset_path=reference.dataset_path,
            demo_key=episode_name,
            ep_meta=_thaw_json(validated_bundle.ep_meta),
            model_file=model_file,
            initial_state=validated_bundle.initial_state[0].copy(),
            action_length=int(validated_bundle.initial_state.shape[0]),
            combined_episode=reference.combined_episode,
            source_episode=reference.source_episode,
            goal_count=reference.goal_count,
            recovery_metadata=recovery_metadata,
            recovery_path=reference.recovery_path,
            recovery_report_entry=reference.recovery_report_entry,
            validated_payload=validated_bundle,
        )

    ep_meta_path = episode_dir / "ep_meta.json"
    model_path = episode_dir / "model.xml.gz"
    states_path = episode_dir / "states.npz"
    missing = [
        str(path)
        for path in (ep_meta_path, model_path, states_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{episode_name}: missing recorded scene files: {missing}"
        )

    with ep_meta_path.open() as file:
        ep_meta = json.load(file)
    with gzip.open(model_path, "rt") as file:
        model_file = file.read()
    with np.load(states_path, allow_pickle=False) as archive:
        states = np.asarray(archive["states"], dtype=np.float64)
    if not isinstance(ep_meta, dict) or not model_file or states.ndim != 2:
        raise ValueError(f"{episode_name}: invalid recorded scene contract")
    if states.shape[0] < 1:
        raise ValueError(f"{episode_name}: recorded states are empty")

    return DemoInitialization(
        dataset_path=reference.dataset_path,
        demo_key=episode_name,
        ep_meta=ep_meta,
        model_file=model_file,
        initial_state=states[0].copy(),
        action_length=int(states.shape[0]),
        combined_episode=reference.combined_episode,
        source_episode=reference.source_episode,
        source_root=reference.source_root,
        source_demo_key=(
            episode_name if reference.source_root is not None else None
        ),
        environment_variant=reference.environment_variant,
        goal_count=reference.goal_count,
        recovery_metadata=None,
        recovery_path=reference.recovery_path,
        recovery_report_entry=copy.deepcopy(
            reference.recovery_report_entry
        ),
        validated_payload=None,
    )


def recovery_result_fields(demo: DemoInitialization | None) -> dict:
    source_fields = {
        "source_root": (
            str(demo.source_root)
            if demo is not None and demo.source_root is not None
            else None
        ),
        "source_demo_key": (
            demo.source_demo_key if demo is not None else None
        ),
        "environment_variant": (
            demo.environment_variant if demo is not None else None
        ),
        "source_hashes": (
            dict(demo.source_hashes)
            if demo is not None and demo.source_hashes is not None
            else None
        ),
    }
    if demo is None or demo.recovery_metadata is None:
        return {
            "combined_episode": (
                demo.combined_episode if demo is not None else None
            ),
            "source_episode": (
                demo.source_episode if demo is not None else None
            ),
            "goal_count": demo.goal_count if demo is not None else None,
            "selected_attempt": None,
            "layout_attempt": None,
            "score_summary": None,
            "codec_noise_envelope": None,
            "generation_commit": None,
            "recovery_input_hashes": None,
            "recovery_output_hashes": None,
            **source_fields,
        }
    validated_payload = demo.validated_payload
    if validated_payload is None:
        raise ValueError("recovered demo has no immutable validated payload")
    recovery = validated_payload.metadata
    return {
        "combined_episode": demo.combined_episode,
        "source_episode": demo.source_episode,
        "goal_count": demo.goal_count,
        # selected_attempt is the bounded build_candidate ordinal.
        "selected_attempt": recovery["selected_attempt"],
        "layout_attempt": recovery["layout_attempt"],
        "score_summary": {
            "selected_per_camera": copy.deepcopy(
                _thaw_json(recovery["selected_per_camera"])
            ),
            "best_aggregate_score": recovery["best_aggregate_score"],
            "second_aggregate_score": recovery["second_aggregate_score"],
            "aggregate_margin": recovery["aggregate_margin"],
        },
        "codec_noise_envelope": recovery["codec_noise_envelope"],
        "generation_commit": recovery["generation_commit"],
        "recovery_input_hashes": _thaw_json(
            validated_payload.input_hashes
        ),
        "recovery_output_hashes": _thaw_json(
            validated_payload.output_hashes
        ),
        **source_fields,
    }


def recorded_scene_lineage(
    demo: DemoInitialization | None,
) -> dict | None:
    if demo is None:
        return None
    return {
        "dataset_path": str(demo.dataset_path),
        "demo_key": demo.demo_key,
        "combined_episode": demo.combined_episode,
        "source_episode": demo.source_episode,
        "source_root": (
            str(demo.source_root) if demo.source_root is not None else None
        ),
        "source_demo_key": demo.source_demo_key,
        "environment_variant": demo.environment_variant,
        "source_hashes": (
            dict(demo.source_hashes)
            if demo.source_hashes is not None
            else None
        ),
        "source_layout_id": demo.ep_meta.get("layout_id"),
        "source_style_id": demo.ep_meta.get("style_id"),
    }


def memory_dataset_paths(
    data_root: str | os.PathLike[str],
    env_name: str,
) -> tuple[pathlib.Path, ...]:
    task_root = pathlib.Path(data_root) / env_name
    paths = []
    for timestamp_dir in sorted(task_root.iterdir()) if task_root.is_dir() else ():
        if not timestamp_dir.is_dir():
            continue
        processed = timestamp_dir / "demo_im128_notp.hdf5"
        raw = timestamp_dir / "demo.hdf5"
        if processed.is_file():
            paths.append(processed)
        elif raw.is_file():
            paths.append(raw)
    return tuple(paths)


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 2**31 - 1, name


def load_demo_references(
    dataset_paths: tuple[pathlib.Path, ...],
) -> tuple[DemoReference, ...]:
    import h5py

    references = []
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as dataset:
            demos = dataset["data"]
            for demo_key in sorted(demos, key=_demo_sort_key):
                references.append(
                    DemoReference(
                        dataset_path=dataset_path,
                        demo_key=demo_key,
                        action_length=int(demos[demo_key]["actions"].shape[0]),
                    )
                )
    return tuple(references)


def load_demo_initialization(reference: DemoReference) -> DemoInitialization:
    import h5py

    with h5py.File(reference.dataset_path, "r") as dataset:
        demo = dataset["data"][reference.demo_key]
        ep_meta = demo.attrs["ep_meta"]
        model_file = demo.attrs["model_file"]
        if isinstance(ep_meta, bytes):
            ep_meta = ep_meta.decode("utf-8")
        if isinstance(model_file, bytes):
            model_file = model_file.decode("utf-8")
        metadata = json.loads(ep_meta) if isinstance(ep_meta, str) else dict(ep_meta)
        initial_state = np.asarray(demo["states"][0], dtype=np.float64).copy()
    return DemoInitialization(
        dataset_path=reference.dataset_path,
        demo_key=reference.demo_key,
        ep_meta=metadata,
        model_file=model_file,
        initial_state=initial_state,
        action_length=reference.action_length,
    )


def _hide_debug_sites(sim) -> None:
    model = sim.model
    for site_id in range(model.nsite):
        legacy_lookup = getattr(model, "site_id2name", None)
        if callable(legacy_lookup):
            name = legacy_lookup(site_id) or ""
        else:
            try:
                name = model.site(site_id).name or ""
            except (AttributeError, IndexError, KeyError, ValueError):
                name = ""
        if _DEBUG_SITE_RE.search(name):
            model.site_rgba[site_id][3] = 0.0


def _compute_container_lifts(env) -> dict[int, float]:
    try:
        from scripts.replay.replay_memory_demos import (
            _compute_container_lifts as archived_compute_container_lifts,
        )
    except ModuleNotFoundError:
        helper_path = pathlib.Path(
            os.environ[
                "ROBOCASA_MEMORY_REPLAY_HELPER"
            ]
        )
        spec = importlib.util.spec_from_file_location(
            "_memory_replay_helper",
            helper_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Cannot load memory replay helper: {helper_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        archived_compute_container_lifts = module._compute_container_lifts

    return archived_compute_container_lifts(env)


def _lift_seed_state(env, state0: np.ndarray) -> np.ndarray:
    lifts = _compute_container_lifts(env)
    if not lifts:
        return state0
    lifted_state = np.asarray(state0).copy()
    for qpos_z_address, lift in lifts.items():
        lifted_state[1 + qpos_z_address] += lift
    return lifted_state


def reset_to_demo(env, demo: DemoInitialization) -> dict:
    raw_env = env.env
    raw_env.set_ep_meta(copy.deepcopy(demo.ep_meta))
    raw_env.reset()
    model_xml = raw_env.edit_model_xml(demo.model_file)
    raw_env.reset_from_xml_string(model_xml)
    raw_env.sim.reset()
    raw_env.sim.set_state_from_flattened(demo.initial_state)
    raw_env.sim.forward()
    raw_env.update_state()
    if demo.environment_variant in _WASH_ENV_NAMES:
        lifted_state = _lift_seed_state(env, demo.initial_state)
        if lifted_state is not demo.initial_state:
            raw_env.sim.set_state_from_flattened(lifted_state)
            raw_env.sim.forward()
            raw_env.update_state()
    if raw_env.__class__.__name__ == "FryingPanAdjustment":
        restored_start_loc = raw_env._check_obj_location_on_stove("obj")
        if restored_start_loc is None:
            raise RuntimeError(
                "FryingPanAdjustment restored pan is not on a burner"
            )
        raw_env.start_loc = restored_start_loc
    _hide_debug_sites(raw_env.sim)
    raw_observation = raw_env._get_observations(force_update=True)
    return env.get_observation(raw_observation)


@dataclasses.dataclass(frozen=True)
class SweetenContract:
    goal_count: int
    original_cube_names: tuple[str, ...]
    distractor_cube_names: tuple[str, ...] = ()

    @property
    def all_cube_names(self) -> tuple[str, ...]:
        return self.original_cube_names + self.distractor_cube_names


def sweeten_contract_from_ep_meta(ep_meta: dict) -> SweetenContract:
    augmentation = ep_meta.get("augmentation")
    if isinstance(augmentation, dict):
        goal_count = augmentation.get("goal_count")
        original_names = augmentation.get("original_cube_names")
        distractor_names = augmentation.get("distractor_names", ())
    else:
        refs = ep_meta.get("refs", {})
        goal_count = refs.get("num_sugar")
        original_names = [
            config.get("name")
            for config in ep_meta.get("object_cfgs", ())
            if isinstance(config, dict)
            and isinstance(config.get("name"), str)
            and config["name"].startswith("sugar")
            and not config["name"].startswith("sugar_aug")
        ]
        distractor_names = ()
        if isinstance(goal_count, int) and not original_names:
            original_names = [f"sugar{index}" for index in range(goal_count)]

    if (
        isinstance(goal_count, bool)
        or not isinstance(goal_count, int)
        or goal_count < 1
        or not isinstance(original_names, (list, tuple))
        or not isinstance(distractor_names, (list, tuple))
    ):
        raise ValueError("Sweeten metadata has no valid goal contract")
    original_names = tuple(original_names)
    distractor_names = tuple(distractor_names)
    if (
        goal_count > len(original_names)
        or len(set(original_names)) != len(original_names)
        or len(set(distractor_names)) != len(distractor_names)
        or set(original_names) & set(distractor_names)
        or any(
            not isinstance(name, str)
            or not name
            or name.startswith("sugar_aug")
            for name in original_names
        )
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"sugar_aug\d+", name) is None
            for name in distractor_names
        )
    ):
        raise ValueError("Sweeten metadata has invalid original cube names")
    return SweetenContract(goal_count, original_names, distractor_names)


def require_augmented_sweeten_contract(ep_meta: dict) -> SweetenContract:
    if not isinstance(ep_meta.get("augmentation"), dict):
        raise ValueError(
            "Sweeten evaluation requires recorded augmented scene metadata"
        )
    return sweeten_contract_from_ep_meta(ep_meta)


def require_recovered_sweeten_contract(
    demo: DemoInitialization | None,
    ep_meta: dict,
) -> tuple[SweetenContract, str]:
    if (
        demo is None
        or demo.recovery_metadata is None
        or demo.combined_episode is None
        or demo.source_episode is None
        or demo.goal_count is None
    ):
        raise ValueError("Sweeten evaluation requires a recovered scene bundle")
    contract = require_augmented_sweeten_contract(ep_meta)
    augmentation = ep_meta.get("augmentation")
    expected_lineage = {
        "combined_episode": demo.combined_episode,
        "source_episode": demo.source_episode,
        "goal_count": demo.goal_count,
    }
    if (
        contract.goal_count != demo.goal_count
        or not isinstance(augmentation, dict)
        or any(
            augmentation.get(name) != value
            for name, value in expected_lineage.items()
        )
        or any(
            demo.recovery_metadata.get(name) != value
            for name, value in expected_lineage.items()
        )
    ):
        raise ValueError("Sweeten goal contract disagrees with recovery lineage")
    instruction = sweeten_instruction(demo.goal_count)
    if demo.ep_meta.get("lang") != instruction:
        raise ValueError("Sweeten language disagrees with recovery lineage")
    return contract, instruction


def sweeten_instruction(goal_count: int) -> str:
    words = {1: "one", 2: "two", 3: "three"}
    if goal_count not in words:
        raise ValueError(f"unsupported Sweeten goal count: {goal_count}")
    noun = "cube" if goal_count == 1 else "cubes"
    pronoun = "it" if goal_count == 1 else "them"
    return (
        f"Pick up {words[goal_count]} sugar {noun} and place {pronoun} "
        "in the saucepan with hot chocolate."
    )


@dataclasses.dataclass(frozen=True)
class RobotPose:
    eef_position: np.ndarray
    eef_quaternion: np.ndarray
    torso: float
    arm_qpos: np.ndarray
    mobile_qpos: np.ndarray


def _normalized_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.shape != (4,) or not np.isfinite(norm) or norm == 0.0:
        raise ValueError("robot quaternion must be finite and nonzero")
    return value / norm


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(
        float(
            np.dot(
                _normalized_quaternion(first),
                _normalized_quaternion(second),
            )
        )
    )
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _validate_finite_robot_pose(pose: RobotPose) -> None:
    values = (
        pose.eef_position,
        pose.eef_quaternion,
        pose.torso,
        pose.arm_qpos,
        pose.mobile_qpos,
    )
    if any(not np.all(np.isfinite(np.asarray(value))) for value in values):
        raise ValueError("Sweeten requires finite RobotPose values")


def discover_sweeten_cube_names(
    env,
    contract: SweetenContract,
) -> tuple[str, ...]:
    model = getattr(getattr(env.env, "sim", None), "model", None)
    if model is None or type(getattr(model, "nbody", None)) is not int:
        raise ValueError("Sweeten physical sugar cube inventory is unavailable")
    physical_names = set()
    for body_id in range(model.nbody):
        body_name = model.body(body_id).name
        match = (
            re.fullmatch(r"(sugar(?:\d+|_aug\d+))_main", body_name)
            if isinstance(body_name, str)
            else None
        )
        if match is not None:
            physical_names.add(match.group(1))
    expected_names = contract.all_cube_names
    if physical_names != set(expected_names):
        raise ValueError(
            "Sweeten physical sugar cube inventory does not match "
            f"recorded metadata: expected={sorted(expected_names)}, "
            f"actual={sorted(physical_names)}"
        )
    return expected_names


def _replayed_object_geometry(
    raw_env,
    object_name: str,
) -> tuple[int, list[str]]:
    model = raw_env.sim.model
    body_id = int(model.body(f"{object_name}_main").id)
    geom_names = [
        model.geom(index).name
        for index in range(model.ngeom)
        if (
            int(model.geom_bodyid[index]) == body_id
            and isinstance(model.geom(index).name, str)
            and not model.geom(index).name.endswith("_reg_bbox")
        )
    ]
    if not geom_names:
        raise ValueError(f"{object_name}: replayed object has no contact geoms")
    return body_id, geom_names


class SweetenSuccessTracker:
    def __init__(
        self,
        contract: SweetenContract,
        initial_pose: RobotPose,
        *,
        cube_names: tuple[str, ...] | None = None,
        stable_frames: int = 10,
        max_return_position_m: float = 0.02,
        max_return_orientation_rad: float = math.radians(5.0),
        max_return_torso: float = 0.02,
        max_return_arm_joint_error_rad: float = 0.01,
        max_mobile_drift: float = 2e-4,
    ):
        if stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        _validate_finite_robot_pose(initial_pose)
        self.contract = contract
        self.cube_names = (
            contract.all_cube_names
            if cube_names is None
            else tuple(cube_names)
        )
        if (
            self.cube_names != contract.all_cube_names
            or len(set(self.cube_names)) != len(self.cube_names)
        ):
            raise ValueError(
                "Sweeten tracker cube inventory does not match contract"
            )
        self.initial_pose = initial_pose
        self.stable_frames = int(stable_frames)
        self.max_return_position_m = float(max_return_position_m)
        self.max_return_orientation_rad = float(max_return_orientation_rad)
        self.max_return_torso = float(max_return_torso)
        self.max_return_arm_joint_error_rad = float(
            max_return_arm_joint_error_rad
        )
        self.max_mobile_drift = float(max_mobile_drift)
        self._stable_counts = {
            name: 0 for name in self.cube_names
        }
        self._first_frames = {name: None for name in self.cube_names}
        self._emitted = set()
        self._events = []
        self._last_frame = None
        self._observed_mobile_drift = 0.0
        self._goal_retention_lost = False
        self._accepted_post_placement_disappearances = set()
        self._invalid_unplaced_disappearances = set()
        self._last_errors = {}
        self._step_trace = []

    @property
    def goal_cube_names(self) -> tuple[str, ...]:
        return tuple(
            event["cube_name"] for event in self._events
        )[: self.contract.goal_count]

    @property
    def distractor_placement_count(self) -> int:
        return sum(
            event["cube_name"] in self.contract.distractor_cube_names
            for event in self._events
        )

    @property
    def diagnostics(self) -> dict:
        return {
            "goal_count": self.contract.goal_count,
            "original_cube_names": list(self.contract.original_cube_names),
            "distractor_cube_names": list(
                self.contract.distractor_cube_names
            ),
            "all_cube_names": list(self.cube_names),
            "goal_cube_names": list(self.goal_cube_names),
            "stable_placement_count": len(self._events),
            "distractor_placement_count": self.distractor_placement_count,
            "placement_events": [
                dict(event) for event in self._events
            ],
            "goal_retention_lost": self._goal_retention_lost,
            "accepted_post_placement_disappearances": sorted(
                self._accepted_post_placement_disappearances
            ),
            "invalid_unplaced_disappearances": sorted(
                self._invalid_unplaced_disappearances
            ),
            "max_mobile_joint_drift": self._observed_mobile_drift,
            "initial_robot_pose": {
                "eef_position_relative_m": {
                    axis: float(value)
                    for axis, value in zip(
                        ("x", "y", "z"),
                        self.initial_pose.eef_position,
                    )
                },
                "eef_quaternion": [
                    float(value)
                    for value in self.initial_pose.eef_quaternion
                ],
                "torso": float(self.initial_pose.torso),
                "arm_qpos": [
                    float(value) for value in self.initial_pose.arm_qpos
                ],
                "mobile_qpos": [
                    float(value) for value in self.initial_pose.mobile_qpos
                ],
            },
            "step_trace": [dict(item) for item in self._step_trace],
            **self._last_errors,
        }

    def update(
        self,
        frame: int,
        status: dict[str, tuple[bool, bool] | None],
        pose: RobotPose,
        *,
        gripper_far: bool | None = None,
    ) -> bool:
        _validate_finite_robot_pose(pose)
        if gripper_far is not None and type(gripper_far) is not bool:
            raise ValueError("Sweeten gripper_far must be a bool")
        names = self.cube_names
        if set(status) != set(names):
            raise ValueError("Sweeten status does not match physical cubes")
        if self._last_frame is not None and frame != self._last_frame + 1:
            self._stable_counts = {name: 0 for name in names}
            self._first_frames = {name: None for name in names}
        self._last_frame = int(frame)

        for name in names:
            cube_status = status[name]
            if cube_status is None:
                self._stable_counts[name] = 0
                self._first_frames[name] = None
                if name in self._emitted:
                    self._accepted_post_placement_disappearances.add(name)
                else:
                    self._invalid_unplaced_disappearances.add(name)
                continue
            inside, grasped = cube_status
            if name in self._emitted:
                continue
            if inside and not grasped:
                if self._stable_counts[name] == 0:
                    self._first_frames[name] = int(frame)
                self._stable_counts[name] += 1
                if self._stable_counts[name] == self.stable_frames:
                    self._events.append(
                        {
                            "cube_name": name,
                            "first_frame": self._first_frames[name],
                            "confirmation_frame": int(frame),
                        }
                    )
                    self._emitted.add(name)
            else:
                self._stable_counts[name] = 0
                self._first_frames[name] = None

        for name in self.goal_cube_names:
            cube_status = status[name]
            if cube_status is None:
                continue
            inside, grasped = cube_status
            if not inside or grasped:
                self._goal_retention_lost = True

        mobile_drift = float(
            np.max(
                np.abs(
                    np.asarray(pose.mobile_qpos)
                    - np.asarray(self.initial_pose.mobile_qpos)
                )
            )
        )
        self._observed_mobile_drift = max(
            self._observed_mobile_drift,
            mobile_drift,
        )
        position_error = float(
            np.linalg.norm(
                np.asarray(pose.eef_position)
                - np.asarray(self.initial_pose.eef_position)
            )
        )
        orientation_error = _quaternion_distance(
            pose.eef_quaternion,
            self.initial_pose.eef_quaternion,
        )
        torso_error = abs(float(pose.torso) - float(self.initial_pose.torso))
        arm_error = float(
            np.max(
                np.abs(
                    np.asarray(pose.arm_qpos)
                    - np.asarray(self.initial_pose.arm_qpos)
                )
            )
        )
        self._last_errors = {
            "return_position_m": position_error,
            "return_orientation_rad": orientation_error,
            "return_torso": torso_error,
            "return_arm_joint_error_rad": arm_error,
            "gripper_far_from_saucepan": gripper_far,
        }

        goal_names = self.goal_cube_names
        inside_names = tuple(
            name
            for name in names
            if status[name] is not None and status[name][0]
        )
        grasped_names = tuple(
            name
            for name in names
            if status[name] is not None and status[name][1]
        )
        visible_goal_names = tuple(
            name for name in goal_names if status[name] is not None
        )
        eef_position = np.asarray(pose.eef_position, dtype=np.float64)
        eef_delta = (
            eef_position
            - np.asarray(self.initial_pose.eef_position, dtype=np.float64)
        )
        self._step_trace.append(
            {
                "frame": int(frame),
                "eef_position_relative_m": {
                    axis: float(value)
                    for axis, value in zip(("x", "y", "z"), eef_position)
                },
                "eef_delta_from_initial_m": {
                    axis: float(value)
                    for axis, value in zip(("x", "y", "z"), eef_delta)
                },
                "return_position_m": position_error,
                "return_orientation_rad": orientation_error,
                "torso": float(pose.torso),
                "return_torso": torso_error,
                "arm_qpos": [float(value) for value in pose.arm_qpos],
                "return_arm_joint_error_rad": arm_error,
                "mobile_qpos": [
                    float(value) for value in pose.mobile_qpos
                ],
                "mobile_joint_drift": mobile_drift,
                "max_mobile_joint_drift": self._observed_mobile_drift,
                "gripper_far_from_saucepan": gripper_far,
                "stable_placement_count": len(self._events),
                "inside_cube_names": list(inside_names),
                "grasped_cube_names": list(grasped_names),
                "cube_status": {
                    name: (
                        None
                        if status[name] is None
                        else {
                            "inside": bool(status[name][0]),
                            "grasped": bool(status[name][1]),
                        }
                    )
                    for name in names
                },
            }
        )
        returned = (
            position_error <= self.max_return_position_m
            and orientation_error <= self.max_return_orientation_rad
            and torso_error <= self.max_return_torso
            and arm_error <= self.max_return_arm_joint_error_rad
        )
        return (
            len(self._events) == self.contract.goal_count
            and set(inside_names) == set(visible_goal_names)
            and not grasped_names
            and not self._goal_retention_lost
            and not self._invalid_unplaced_disappearances
            and self._observed_mobile_drift <= self.max_mobile_drift
            and returned
        )


def _require_retrieve_target(env, target_name: str):
    raw_env = env.env
    objects = getattr(raw_env, "objects", None)
    if not isinstance(objects, dict) or target_name not in objects:
        raise ValueError(
            f"Retrieve target object is missing: {target_name}"
        )
    target_object = objects[target_name]
    if target_object is None:
        raise ValueError(
            f"Retrieve target object is invalid: {target_name}"
        )
    try:
        gripper = raw_env.robots[0].gripper["right"]
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("Retrieve right gripper evidence is missing") from error
    return raw_env, target_object, gripper


def _retrieve_target_grasped(
    env,
    target_name: str,
    target_object,
) -> bool:
    raw_env, current_target, gripper = _require_retrieve_target(
        env,
        target_name,
    )
    if current_target is not target_object:
        raise ValueError(
            f"Retrieve target identity changed: {target_name}"
        )
    return bool(raw_env._check_grasp(gripper, target_object))


def _retrieve_object_position(
    env,
    target_name: str,
    target_object,
) -> np.ndarray:
    from robocasa.utils import object_utils

    raw_env, current_target, _ = _require_retrieve_target(env, target_name)
    if current_target is not target_object:
        raise ValueError(
            f"Retrieve target identity changed: {target_name}"
        )
    position = np.asarray(
        object_utils.get_obj_pos(raw_env, obj_name=target_name),
        dtype=np.float64,
    )
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(
            f"Retrieve target position is invalid: {target_name}"
        )
    return position.copy()


_FRUIT_SINK_Z_PAD_M = 0.03
_FRUIT_RETURN_LR_M = 0.08
_FRUIT_RETURN_SAME_M = 0.05
_FRUIT_GRIPPER_OPEN_GAP = 0.075
_FRUIT_GRIPPER_FAR_M = 0.10


def _raw_object_body_position(raw_env, object_name: str) -> np.ndarray:
    objects = getattr(raw_env, "objects", None)
    if not isinstance(objects, dict) or object_name not in objects:
        raise ValueError(f"Object is missing: {object_name}")
    object_model = objects[object_name]
    body_name = getattr(object_model, "name", None)
    body_ids = getattr(raw_env, "obj_body_id", None)
    if not body_name or not isinstance(body_ids, dict):
        raise ValueError(f"Object body evidence is missing: {object_name}")
    try:
        position = np.asarray(
            raw_env.sim.data.body_xpos[body_ids[body_name]],
            dtype=np.float64,
        )
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError(
            f"Object body position is missing: {object_name}"
        ) from error
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"Object body position is invalid: {object_name}")
    return position.copy()


def _fruit_in_sink_region(raw_env) -> bool:
    from robocasa.utils import object_utils

    objects = getattr(raw_env, "objects", None)
    if not isinstance(objects, dict) or objects.get("fruit") is None:
        raise ValueError("Fruit sink-region evidence is missing")
    fruit = objects["fruit"]
    body_name = getattr(fruit, "name", None)
    body_ids = getattr(raw_env, "obj_body_id", None)
    sink = getattr(raw_env, "sink", None)
    if (
        not body_name
        or not isinstance(body_ids, dict)
        or body_name not in body_ids
        or sink is None
        or not callable(getattr(sink, "get_int_sites", None))
    ):
        raise ValueError("Fruit sink-region evidence is missing")
    body_id = body_ids[body_name]
    try:
        fruit_position = np.asarray(
            raw_env.sim.data.body_xpos[body_id],
            dtype=np.float64,
        )
        fruit_quaternion = np.asarray(
            raw_env.sim.data.body_xquat[body_id],
            dtype=np.float64,
        )
        interior_sites = sink.get_int_sites(relative=False)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("Fruit sink-region evidence is missing") from error
    if not isinstance(interior_sites, dict) or not interior_sites:
        raise ValueError("Fruit sink interior sites are missing")
    for points in interior_sites.values():
        if len(points) != 4:
            raise ValueError("Fruit sink interior region is invalid")
        p0, px, py, pz = (
            np.asarray(point, dtype=np.float64).copy()
            for point in points
        )
        pz[2] += _FRUIT_SINK_Z_PAD_M
        if object_utils.obj_in_region(
            fruit,
            fruit_position,
            fruit_quaternion,
            p0,
            px,
            py,
            pz,
        ):
            return True
    return False


def _fruit_release_metrics(raw_env) -> tuple[float, float]:
    fruit_position = _raw_object_body_position(raw_env, "fruit")
    try:
        gripper_qpos = np.asarray(
            raw_env.sim.data.qpos[[11, 12]],
            dtype=np.float64,
        )
        gripper_site_id = raw_env.robots[0].eef_site_id["right"]
        gripper_position = np.asarray(
            raw_env.sim.data.site_xpos[gripper_site_id],
            dtype=np.float64,
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
    ) as error:
        raise ValueError("Fruit gripper release evidence is missing") from error
    finger_gap = float(np.sum(np.abs(gripper_qpos)))
    gripper_distance = float(
        np.linalg.norm(gripper_position - fruit_position)
    )
    if not np.isfinite(finger_gap) or not np.isfinite(gripper_distance):
        raise ValueError("Fruit gripper release evidence is invalid")
    return finger_gap, gripper_distance


def _fruit_released(raw_env) -> bool:
    finger_gap, gripper_distance = _fruit_release_metrics(raw_env)
    return (
        finger_gap > _FRUIT_GRIPPER_OPEN_GAP
        and gripper_distance > _FRUIT_GRIPPER_FAR_M
    )


class RetrieveFruitSuccessTracker:
    target_name = "fruit"

    def __init__(self, target_object):
        self.target_object = target_object
        self._target_grasp_observed = False
        self._updates = 0
        self._inside_sink_basin = False
        self._grasped = False
        self._gripper_far = False
        self._gripper_open = False
        self._gripper_finger_gap = 0.0
        self._gripper_fruit_distance_m = 0.0
        self._basin_location_full = "unavailable"
        self._basin_location_partial = "unavailable"
        self._basin_location_observed = False

    @classmethod
    def from_restored_env(cls, env):
        _, target_object, _ = _require_retrieve_target(
            env,
            cls.target_name,
        )
        return cls(target_object)

    @property
    def diagnostics(self) -> dict:
        return {
            "tracker": "retrieve_fruit",
            "target_name": self.target_name,
            "updates": self._updates,
            "target_grasp_observed": self._target_grasp_observed,
            "inside_sink_basin": self._inside_sink_basin,
            "grasped": self._grasped,
            "gripper_far": self._gripper_far,
            "gripper_open": self._gripper_open,
            "gripper_finger_gap": self._gripper_finger_gap,
            "gripper_open_threshold": _FRUIT_GRIPPER_OPEN_GAP,
            "gripper_fruit_distance_m": (
                self._gripper_fruit_distance_m
            ),
            "gripper_far_threshold_m": _FRUIT_GRIPPER_FAR_M,
            "sink_z_pad_m": _FRUIT_SINK_Z_PAD_M,
            "basin_location_full": self._basin_location_full,
            "basin_location_partial": self._basin_location_partial,
            "basin_location_observed": self._basin_location_observed,
        }

    def update(self, env) -> bool:
        raw_env, current_target, _ = _require_retrieve_target(
            env,
            self.target_name,
        )
        if current_target is not self.target_object:
            raise ValueError(
                f"Retrieve target identity changed: {self.target_name}"
            )
        sink = getattr(raw_env, "sink", None)
        if sink is None:
            raise ValueError("Retrieve Fruit sink evidence is missing")
        get_obj_basin_loc = getattr(sink, "get_obj_basin_loc", None)
        if not callable(get_obj_basin_loc):
            raise ValueError("Retrieve Fruit sink basin evidence is missing")
        basin_location_full = str(
            get_obj_basin_loc(
                raw_env,
                self.target_name,
                partial_check=False,
            )
        )
        basin_location_partial = str(
            get_obj_basin_loc(
                raw_env,
                self.target_name,
                partial_check=True,
            )
        )
        inside_sink_basin = basin_location_full != "none"
        grasped = _retrieve_target_grasped(
            env,
            self.target_name,
            self.target_object,
        )
        finger_gap, gripper_distance = _fruit_release_metrics(raw_env)
        gripper_open = finger_gap > _FRUIT_GRIPPER_OPEN_GAP
        gripper_far = gripper_distance > _FRUIT_GRIPPER_FAR_M
        released = (
            gripper_open
            and gripper_far
        )
        self._updates += 1
        self._inside_sink_basin = inside_sink_basin
        self._grasped = grasped
        self._gripper_far = gripper_far
        self._gripper_open = gripper_open
        self._gripper_finger_gap = finger_gap
        self._gripper_fruit_distance_m = gripper_distance
        self._basin_location_full = basin_location_full
        self._basin_location_partial = basin_location_partial
        self._basin_location_observed |= (
            basin_location_full != "none"
            or basin_location_partial != "none"
        ) and (
            basin_location_full != "unavailable"
            or basin_location_partial != "unavailable"
        )
        self._target_grasp_observed |= grasped
        return inside_sink_basin and released


class RetrieveOilSuccessTracker:
    target_name = "olive_oil"

    def __init__(
        self,
        target_object,
        initial_position: np.ndarray,
        *,
        minimum_lift_m: float = 0.08,
    ):
        self.target_object = target_object
        self.initial_position = np.asarray(
            initial_position,
            dtype=np.float64,
        ).copy()
        self.minimum_lift_m = float(minimum_lift_m)
        self._updates = 0
        self._target_grasp_observed = False
        self._valid_grasped_lift_observed = False
        self._awaiting_elevated_release = False
        self._current_z = float(self.initial_position[2])
        self._grasped = False

    @classmethod
    def from_restored_env(cls, env):
        _, target_object, _ = _require_retrieve_target(
            env,
            cls.target_name,
        )
        initial_position = _retrieve_object_position(
            env,
            cls.target_name,
            target_object,
        )
        return cls(target_object, initial_position)

    @property
    def diagnostics(self) -> dict:
        return {
            "tracker": "retrieve_oil",
            "target_name": self.target_name,
            "updates": self._updates,
            "initial_z": float(self.initial_position[2]),
            "current_z": self._current_z,
            "lift_m": self._current_z - float(self.initial_position[2]),
            "minimum_lift_m": self.minimum_lift_m,
            "target_grasp_observed": self._target_grasp_observed,
            "valid_grasped_lift_observed": (
                self._valid_grasped_lift_observed
            ),
            "awaiting_elevated_release": (
                self._awaiting_elevated_release
            ),
            "grasped": self._grasped,
        }

    def update(self, env) -> bool:
        current_position = _retrieve_object_position(
            env,
            self.target_name,
            self.target_object,
        )
        grasped = _retrieve_target_grasped(
            env,
            self.target_name,
            self.target_object,
        )
        current_z = float(current_position[2])
        above_lift_boundary = (
            current_z
            > float(self.initial_position[2]) + self.minimum_lift_m
        )
        self._updates += 1
        self._current_z = current_z
        self._grasped = grasped
        self._target_grasp_observed |= grasped
        self._valid_grasped_lift_observed |= (
            grasped and above_lift_boundary
        )
        self._awaiting_elevated_release = False
        return above_lift_boundary


def make_retrieve_success_tracker(env_name: str, env):
    if env_name.startswith("MemFruitInSink"):
        return RetrieveFruitSuccessTracker.from_restored_env(env)
    if env_name.startswith("MemRetrieveOilsFromCounter"):
        return RetrieveOilSuccessTracker.from_restored_env(env)
    raise ValueError(f"{env_name}: no Retrieve success tracker")


def _require_wash_target(env, target_name: str):
    raw_env = env.env
    objects = getattr(raw_env, "objects", None)
    if not isinstance(objects, dict) or target_name not in objects:
        raise ValueError(f"Wash target object is missing: {target_name}")
    target_object = objects[target_name]
    if target_object is None:
        raise ValueError(f"Wash target object is invalid: {target_name}")
    try:
        gripper = raw_env.robots[0].gripper["right"]
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("Wash right gripper evidence is missing") from error
    return raw_env, target_object, gripper


def _wash_target_position(
    env,
    target_name: str,
    target_object,
) -> np.ndarray:
    from robocasa.utils import object_utils

    raw_env, current_target, _ = _require_wash_target(env, target_name)
    if current_target is not target_object:
        raise ValueError(f"Wash target identity changed: {target_name}")
    position = np.asarray(
        object_utils.get_obj_pos(raw_env, obj_name=target_name),
        dtype=np.float64,
    )
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"Wash target position is invalid: {target_name}")
    return position.copy()


class WashSuccessTracker:
    """Mirror the archived Wash-and-Return task success state machine."""

    target_name = "fruit"

    def __init__(
        self,
        target_object,
        initial_position: np.ndarray,
        return_mode: str,
        *,
        destination_container_name: str | None = None,
        return_tolerance_m: float | None = None,
    ):
        if return_mode not in {"source_container", "same_location"}:
            raise ValueError(f"Unsupported Wash return mode: {return_mode}")
        if (
            return_mode == "source_container"
            and destination_container_name != "fruit_container"
        ):
            raise ValueError("Wash destination container is invalid")
        self.target_object = target_object
        self.initial_position = np.asarray(
            initial_position,
            dtype=np.float64,
        ).copy()
        self.return_mode = return_mode
        self.destination_container_name = destination_container_name
        if return_tolerance_m is None:
            return_tolerance_m = (
                _FRUIT_RETURN_LR_M
                if return_mode == "source_container"
                else _FRUIT_RETURN_SAME_M
            )
        self.return_tolerance_m = float(return_tolerance_m)
        self._updates = 0
        self._sink_entry_update = None
        self._done = False
        self._inside_sink_basin = False
        self._current_position = self.initial_position.copy()
        self._return_distance_xy_m = 0.0
        self._return_abs_x_m = 0.0
        self._return_abs_y_m = 0.0
        self._in_destination_container = False
        self._gripper_far = False
        self._gripper_open = False
        self._gripper_finger_gap = 0.0
        self._gripper_fruit_distance_m = 0.0
        self._grasped = False
        self._destination_contact = None
        self._destination_distance_xy_m = None
        self._destination_threshold_m = None

    @classmethod
    def from_restored_env(cls, env, *, env_name: str):
        raw_env, target_object, _ = _require_wash_target(
            env,
            cls.target_name,
        )
        initial_position = _wash_target_position(
            env,
            cls.target_name,
            target_object,
        )
        if env_name == "MemWashAndReturnSameLocation":
            return cls(
                target_object,
                initial_position,
                "same_location",
            )
        if env_name in {
            "MemWashAndReturnLeft",
            "MemWashAndReturnRight",
        }:
            objects = getattr(raw_env, "objects", None)
            if (
                not isinstance(objects, dict)
                or objects.get("fruit_container") is None
            ):
                raise ValueError("Wash destination container is missing")
            return cls(
                target_object,
                initial_position,
                "source_container",
                destination_container_name="fruit_container",
            )
        raise ValueError(f"{env_name}: no Wash success tracker")

    @property
    def diagnostics(self) -> dict:
        return {
            "tracker": "wash_and_return",
            "target_name": self.target_name,
            "updates": self._updates,
            "initial_position": self.initial_position.tolist(),
            "current_position": self._current_position.tolist(),
            "return_mode": self.return_mode,
            "destination_container_name": self.destination_container_name,
            "return_tolerance_m": self.return_tolerance_m,
            "return_distance_xy_m": self._return_distance_xy_m,
            "return_abs_x_m": self._return_abs_x_m,
            "return_abs_y_m": self._return_abs_y_m,
            "sink_entry_observed": self._sink_entry_update is not None,
            "sink_entry_update": self._sink_entry_update,
            "inside_sink_basin": self._inside_sink_basin,
            "in_destination_container": self._in_destination_container,
            "gripper_far": self._gripper_far,
            "gripper_open": self._gripper_open,
            "gripper_finger_gap": self._gripper_finger_gap,
            "gripper_open_threshold": _FRUIT_GRIPPER_OPEN_GAP,
            "gripper_fruit_distance_m": (
                self._gripper_fruit_distance_m
            ),
            "gripper_far_threshold_m": _FRUIT_GRIPPER_FAR_M,
            "sink_z_pad_m": _FRUIT_SINK_Z_PAD_M,
            "grasped": self._grasped,
            "destination_contact": self._destination_contact,
            "destination_distance_xy_m": (
                self._destination_distance_xy_m
            ),
            "destination_threshold_m": self._destination_threshold_m,
        }

    def update(self, env) -> bool:
        from robocasa.utils import object_utils

        raw_env, current_target, gripper = _require_wash_target(
            env,
            self.target_name,
        )
        if current_target is not self.target_object:
            raise ValueError(
                f"Wash target identity changed: {self.target_name}"
            )
        sink = getattr(raw_env, "sink", None)
        if sink is None:
            raise ValueError("Wash sink evidence is missing")
        current_position = _wash_target_position(
            env,
            self.target_name,
            self.target_object,
        )
        inside_sink_basin = _fruit_in_sink_region(raw_env)
        finger_gap, gripper_distance = _fruit_release_metrics(raw_env)
        gripper_open = finger_gap > _FRUIT_GRIPPER_OPEN_GAP
        gripper_far = gripper_distance > _FRUIT_GRIPPER_FAR_M
        released = gripper_open and gripper_far
        grasped = bool(
            raw_env._check_grasp(gripper, self.target_object)
        )
        return_distance_xy_m = float(
            np.linalg.norm(
                current_position[:2] - self.initial_position[:2]
            )
        )
        return_abs_x_m = float(
            abs(current_position[0] - self.initial_position[0])
        )
        return_abs_y_m = float(
            abs(current_position[1] - self.initial_position[1])
        )
        in_destination_container = False
        destination_contact = None
        destination_distance_xy_m = None
        destination_threshold_m = None
        if self.return_mode == "source_container":
            destination_object = raw_env.objects[
                self.destination_container_name
            ]
            destination_position = np.asarray(
                object_utils.get_obj_pos(
                    raw_env,
                    obj_name=self.destination_container_name,
                ),
                dtype=np.float64,
            )
            destination_abs_x_m = float(
                abs(current_position[0] - destination_position[0])
            )
            destination_abs_y_m = float(
                abs(current_position[1] - destination_position[1])
            )
            in_destination_container = (
                destination_abs_x_m < self.return_tolerance_m
                and destination_abs_y_m < self.return_tolerance_m
            )
            destination_distance_xy_m = float(
                np.linalg.norm(
                    current_position[:2] - destination_position[:2]
                )
            )
            destination_threshold_m = self.return_tolerance_m
            try:
                destination_contact = bool(
                    raw_env.check_contact(
                        self.target_object,
                        destination_object,
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

        self._updates += 1
        self._inside_sink_basin = inside_sink_basin
        self._current_position = current_position
        self._return_distance_xy_m = return_distance_xy_m
        self._return_abs_x_m = return_abs_x_m
        self._return_abs_y_m = return_abs_y_m
        self._in_destination_container = in_destination_container
        self._gripper_far = gripper_far
        self._gripper_open = gripper_open
        self._gripper_finger_gap = finger_gap
        self._gripper_fruit_distance_m = gripper_distance
        self._grasped = grasped
        self._destination_contact = destination_contact
        self._destination_distance_xy_m = destination_distance_xy_m
        self._destination_threshold_m = destination_threshold_m
        if self._sink_entry_update is None:
            if inside_sink_basin:
                self._sink_entry_update = self._updates
            return False
        if self._done:
            return True
        if self.return_mode == "source_container":
            self._done = in_destination_container and released
        else:
            self._done = (
                return_abs_x_m < self.return_tolerance_m
                and return_abs_y_m < self.return_tolerance_m
                and released
            )
        return self._done


def make_wash_success_tracker(env_name: str, env):
    if env_name in _WASH_ENV_NAMES:
        return WashSuccessTracker.from_restored_env(
            env,
            env_name=env_name,
        )
    raise ValueError(f"{env_name}: no Wash success tracker")


def _rotation_matrix_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalized_quaternion(quaternion)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _pan_pose(env, pan_object) -> tuple[np.ndarray, np.ndarray]:
    raw_env = env.env
    body_ids = getattr(raw_env, "obj_body_id", None)
    if not isinstance(body_ids, dict):
        raise ValueError("Pan body identity evidence is missing")
    pan_name = getattr(pan_object, "name", "vegetable_container")
    if pan_name not in body_ids:
        pan_name = "vegetable_container"
    if pan_name not in body_ids:
        raise ValueError("Pan body identity evidence is missing")
    try:
        body_id = body_ids[pan_name]
        position = np.asarray(
            raw_env.sim.data.body_xpos[body_id],
            dtype=np.float64,
        )
        quaternion = np.asarray(
            raw_env.sim.data.body_xquat[body_id],
            dtype=np.float64,
        )
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("Pan pose evidence is missing") from error
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(quaternion))
    ):
        raise ValueError("Pan pose evidence is invalid")
    _normalized_quaternion(quaternion)
    return position.copy(), quaternion.copy()


def _pan_yaw(rotation: np.ndarray) -> float:
    forward_xy = np.asarray(rotation[:2, 0], dtype=np.float64)
    norm = float(np.linalg.norm(forward_xy))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Pan yaw evidence is undefined")
    forward_xy /= norm
    return float(math.atan2(forward_xy[1], forward_xy[0]))


def _wrapped_angle_error(first: float, second: float) -> float:
    return abs(float((first - second + math.pi) % (2.0 * math.pi) - math.pi))


def _require_pan_objects(env):
    raw_env = env.env
    objects = getattr(raw_env, "objects", None)
    if not isinstance(objects, dict):
        raise ValueError("Pan object registry is missing")
    for name in ("vegetable_container", "vegetable", "plate"):
        if name not in objects or objects[name] is None:
            label = "pan object" if name == "vegetable_container" else name
            raise ValueError(f"Pan {label} is missing")
    try:
        gripper = raw_env.robots[0].gripper["right"]
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("Pan right gripper evidence is missing") from error
    stove = getattr(raw_env, "stove", None)
    burner_sites = getattr(stove, "burner_sites", None)
    if stove is None or not isinstance(burner_sites, dict) or not burner_sites:
        raise ValueError("Pan stove burner evidence is missing")
    return (
        raw_env,
        objects["vegetable_container"],
        objects["vegetable"],
        gripper,
        stove,
    )


def _pan_burner_evidence(
    env,
    pan_position: np.ndarray,
    *,
    tolerance_m: float,
) -> tuple[str | None, dict[str, np.ndarray], dict[str, float], bool]:
    from robocasa.utils import object_utils

    raw_env, _, _, _, stove = _require_pan_objects(env)
    try:
        pan_on_stove = bool(
            object_utils.check_obj_fixture_contact(
                raw_env,
                "vegetable_container",
                stove,
            )
        )
    except Exception as error:
        raise ValueError("Pan stove contact evidence is missing") from error

    burner_positions = {}
    burner_distances = {}
    for burner_name, site in stove.burner_sites.items():
        if site is None:
            continue
        try:
            site_name = site.get("name")
            site_position = np.asarray(
                raw_env.sim.data.get_site_xpos(site_name),
                dtype=np.float64,
            )
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ValueError("Pan burner site evidence is missing") from error
        if (
            not isinstance(burner_name, str)
            or not burner_name
            or not isinstance(site_name, str)
            or not site_name
            or site_position.shape != (3,)
            or not np.all(np.isfinite(site_position))
        ):
            raise ValueError("Pan burner site evidence is invalid")
        burner_positions[burner_name] = site_position.copy()
        burner_distances[burner_name] = float(
            np.linalg.norm(pan_position[:2] - site_position[:2])
        )
    if not burner_positions:
        raise ValueError("Pan stove has no usable burner sites")

    matching_burners = (
        [
            name
            for name, distance in burner_distances.items()
            if distance < tolerance_m
        ]
        if pan_on_stove
        else []
    )
    current_burner = (
        matching_burners[0] if len(matching_burners) == 1 else None
    )
    return (
        current_burner,
        burner_positions,
        burner_distances,
        pan_on_stove,
    )


class PanTransferSuccessTracker:
    pan_name = "vegetable_container"
    vegetable_name = "vegetable"
    plate_name = "plate"

    def __init__(
        self,
        pan_object,
        vegetable_object,
        initial_position: np.ndarray,
        initial_quaternion: np.ndarray,
        recorded_burner: str,
        recorded_burner_position: np.ndarray,
        recorded_burner_distances: dict[str, float],
        *,
        burner_tolerance_m: float = 0.08,
        upright_tolerance_rad: float = math.radians(15.0),
        yaw_tolerance_rad: float = math.radians(15.0),
    ):
        self.pan_object = pan_object
        self.vegetable_object = vegetable_object
        self.initial_position = np.asarray(
            initial_position,
            dtype=np.float64,
        ).copy()
        self.initial_quaternion = _normalized_quaternion(
            initial_quaternion
        ).copy()
        self.recorded_burner = recorded_burner
        self.recorded_burner_position = np.asarray(
            recorded_burner_position,
            dtype=np.float64,
        ).copy()
        self.recorded_burner_distances = {
            str(name): float(distance)
            for name, distance in recorded_burner_distances.items()
        }
        self.burner_tolerance_m = float(burner_tolerance_m)
        self.upright_tolerance_rad = float(upright_tolerance_rad)
        self.yaw_tolerance_rad = float(yaw_tolerance_rad)

        initial_rotation = _rotation_matrix_from_wxyz(
            self.initial_quaternion
        )
        self.initial_upright_axis = initial_rotation[:, 2].copy()
        initial_world_dot = float(self.initial_upright_axis[2])
        if (
            not isinstance(recorded_burner, str)
            or not recorded_burner
            or self.initial_position.shape != (3,)
            or self.recorded_burner_position.shape != (3,)
            or initial_world_dot
            <= math.cos(self.upright_tolerance_rad)
        ):
            raise ValueError("Pan restored pose is not valid and upright")
        self.initial_yaw = _pan_yaw(initial_rotation)

        self._updates = 0
        self._current_burner = recorded_burner
        self._current_position = self.initial_position.copy()
        self._current_upright_axis = self.initial_upright_axis.copy()
        self._current_yaw = self.initial_yaw
        self._position_error_m = float(
            np.linalg.norm(
                self.initial_position[:2]
                - self.recorded_burner_position[:2]
            )
        )
        self._upright_dot = initial_world_dot
        self._upright_angle_rad = float(
            math.acos(float(np.clip(initial_world_dot, -1.0, 1.0)))
        )
        self._yaw_error_rad = 0.0
        self._pan_on_stove = True
        self._vegetable_transferred = False
        self._transfer_event_observed = False
        self._pan_grasped = False
        self._robot_food_contact_current = False
        self._robot_food_contact_observed = False
        self._current_burner_distances = dict(
            self.recorded_burner_distances
        )

    @classmethod
    def from_restored_env(cls, env):
        from robocasa.utils import object_utils

        raw_env, pan_object, vegetable_object, _, _ = _require_pan_objects(
            env
        )
        try:
            vegetable_in_pan = bool(
                object_utils.check_obj_in_receptacle(
                    raw_env,
                    cls.vegetable_name,
                    cls.pan_name,
                )
            )
            vegetable_on_plate = bool(
                object_utils.check_obj_in_receptacle(
                    raw_env,
                    cls.vegetable_name,
                    cls.plate_name,
                )
            )
        except Exception as error:
            raise ValueError(
                "Pan restored vegetable placement evidence is missing"
            ) from error
        if not vegetable_in_pan or vegetable_on_plate:
            raise ValueError(
                "Pan vegetable must start in the pan and not on the plate"
            )
        initial_position, initial_quaternion = _pan_pose(env, pan_object)
        burner_tolerance_m = 0.08
        (
            recorded_burner,
            burner_positions,
            burner_distances,
            _,
        ) = _pan_burner_evidence(
            env,
            initial_position,
            tolerance_m=burner_tolerance_m,
        )
        if recorded_burner is None:
            raise ValueError(
                "Pan is not unambiguously on one recorded burner"
            )
        return cls(
            pan_object,
            vegetable_object,
            initial_position,
            initial_quaternion,
            recorded_burner,
            burner_positions[recorded_burner],
            burner_distances,
            burner_tolerance_m=burner_tolerance_m,
        )

    @property
    def diagnostics(self) -> dict:
        return {
            "tracker": "pan_transfer",
            "updates": self._updates,
            "recorded_burner": self.recorded_burner,
            "current_burner": self._current_burner,
            "initial_pan_position": self.initial_position.tolist(),
            "current_pan_position": self._current_position.tolist(),
            "recorded_burner_position": (
                self.recorded_burner_position.tolist()
            ),
            "recorded_burner_candidate_distances_m": dict(
                self.recorded_burner_distances
            ),
            "current_burner_candidate_distances_m": dict(
                self._current_burner_distances
            ),
            "burner_tolerance_m": self.burner_tolerance_m,
            "position_error_m": self._position_error_m,
            "initial_upright_axis": self.initial_upright_axis.tolist(),
            "current_upright_axis": self._current_upright_axis.tolist(),
            "upright_dot": self._upright_dot,
            "upright_angle_rad": self._upright_angle_rad,
            "upright_tolerance_rad": self.upright_tolerance_rad,
            "initial_yaw_rad": self.initial_yaw,
            "current_yaw_rad": self._current_yaw,
            "yaw_error_rad": self._yaw_error_rad,
            "yaw_tolerance_rad": self.yaw_tolerance_rad,
            "pan_on_stove": self._pan_on_stove,
            "vegetable_transferred": self._vegetable_transferred,
            "transfer_event_observed": self._transfer_event_observed,
            "pan_grasped": self._pan_grasped,
            "pan_released": not self._pan_grasped,
            "robot_food_contact_current": (
                self._robot_food_contact_current
            ),
            "robot_food_contact_observed": (
                self._robot_food_contact_observed
            ),
        }

    def update(self, env) -> bool:
        from robocasa.utils import object_utils

        raw_env, pan_object, vegetable_object, gripper, _ = (
            _require_pan_objects(env)
        )
        if (
            pan_object is not self.pan_object
            or vegetable_object is not self.vegetable_object
        ):
            raise ValueError("Pan object identity changed after reset")
        current_position, current_quaternion = _pan_pose(
            env,
            self.pan_object,
        )
        current_rotation = _rotation_matrix_from_wxyz(current_quaternion)
        current_upright_axis = current_rotation[:, 2]
        upright_dot = float(
            np.clip(current_upright_axis[2], -1.0, 1.0)
        )
        upright_angle_rad = float(math.acos(upright_dot))
        current_yaw = _pan_yaw(current_rotation)
        yaw_error_rad = _wrapped_angle_error(
            current_yaw,
            self.initial_yaw,
        )
        (
            current_burner,
            _,
            current_burner_distances,
            pan_on_stove,
        ) = _pan_burner_evidence(
            env,
            current_position,
            tolerance_m=self.burner_tolerance_m,
        )
        position_error_m = float(
            np.linalg.norm(
                current_position[:2]
                - self.recorded_burner_position[:2]
            )
        )
        try:
            vegetable_transferred = bool(
                object_utils.check_obj_in_receptacle(
                    raw_env,
                    self.vegetable_name,
                    self.plate_name,
                )
            )
            pan_grasped = bool(
                raw_env._check_grasp(gripper, self.pan_object)
            )
            robot_food_contact = bool(
                raw_env.check_contact(gripper, self.vegetable_object)
            )
        except Exception as error:
            raise ValueError("Pan terminal physical evidence is missing") from error

        self._updates += 1
        self._current_burner = current_burner
        self._current_position = current_position
        self._current_upright_axis = current_upright_axis.copy()
        self._current_yaw = current_yaw
        self._position_error_m = position_error_m
        self._upright_dot = upright_dot
        self._upright_angle_rad = upright_angle_rad
        self._yaw_error_rad = yaw_error_rad
        self._pan_on_stove = pan_on_stove
        self._vegetable_transferred = vegetable_transferred
        self._transfer_event_observed |= vegetable_transferred
        self._pan_grasped = pan_grasped
        self._robot_food_contact_current = robot_food_contact
        self._robot_food_contact_observed |= robot_food_contact
        self._current_burner_distances = dict(
            current_burner_distances
        )

        return (
            vegetable_transferred
            and self._transfer_event_observed
            and not pan_grasped
            and not self._robot_food_contact_observed
            and pan_on_stove
            and current_burner == self.recorded_burner
            and position_error_m < self.burner_tolerance_m
            and upright_angle_rad < self.upright_tolerance_rad
        )


def make_pan_transfer_success_tracker(env_name: str, env):
    if env_name == "PanTransfer":
        return PanTransferSuccessTracker.from_restored_env(env)
    raise ValueError(f"{env_name}: no Pan Transfer success tracker")


def _require_rinse_evidence(env, sink, cutting_board):
    raw_env = env.env
    objects = getattr(raw_env, "objects", None)
    if (
        not isinstance(objects, dict)
        or "cutting_board" not in objects
        or objects["cutting_board"] is None
    ):
        raise ValueError("Rinse cutting board is missing")
    if objects["cutting_board"] is not cutting_board:
        raise ValueError("Rinse cutting board identity changed after reset")
    if getattr(raw_env, "sink", None) is not sink:
        raise ValueError("Rinse sink identity changed after reset")
    if not callable(getattr(sink, "get_handle_state", None)):
        raise ValueError("Rinse faucet state evidence is missing")
    if not callable(getattr(sink, "check_obj_under_water", None)):
        raise ValueError("Rinse board-under-water evidence is missing")
    return raw_env


class RinseSuccessTracker:
    target_name = "cutting_board"

    def __init__(
        self,
        sink,
        cutting_board,
        *,
        required_consecutive_steps: int = 100,
        minimum_water_temp: float = 0.6,
        initial_handle_joint: float = 0.0,
        handle_close_tolerance_rad: float = 0.2,
    ):
        if required_consecutive_steps != 100:
            raise ValueError("Rinse requires exactly 100 consecutive steps")
        self.sink = sink
        self.cutting_board = cutting_board
        self.required_consecutive_steps = required_consecutive_steps
        self.minimum_water_temp = float(minimum_water_temp)
        self.initial_handle_joint = float(initial_handle_joint)
        self.handle_close_tolerance_rad = float(
            handle_close_tolerance_rad
        )
        if (
            not np.isfinite(self.initial_handle_joint)
            or not 0.0 < self.handle_close_tolerance_rad < math.pi
        ):
            raise ValueError("Rinse initial faucet state is invalid")
        self._updates = 0
        self._consecutive_duration = 0
        self._max_consecutive = 0
        self._completion_latched = False
        self._water_on = False
        self._water_temp = 0.0
        self._water_hot = False
        self._board_under_water = False
        self._terminal_faucet_off = False
        self._handle_joint = self.initial_handle_joint
        self._handle_close_error_rad = 0.0

    @classmethod
    def from_restored_env(cls, env):
        raw_env = env.env
        objects = getattr(raw_env, "objects", None)
        if (
            not isinstance(objects, dict)
            or "cutting_board" not in objects
            or objects["cutting_board"] is None
        ):
            raise ValueError("Rinse cutting board is missing")
        sink = getattr(raw_env, "sink", None)
        if sink is None:
            raise ValueError("Rinse sink evidence is missing")
        handle_state = sink.get_handle_state(raw_env)
        if not isinstance(handle_state, dict):
            raise ValueError("Rinse faucet state evidence is invalid")
        try:
            initial_handle_joint = float(handle_state["handle_joint"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Rinse faucet handle evidence is invalid") from error
        tracker = cls(
            sink,
            objects["cutting_board"],
            initial_handle_joint=initial_handle_joint,
        )
        _require_rinse_evidence(
            env,
            tracker.sink,
            tracker.cutting_board,
        )
        return tracker

    @property
    def diagnostics(self) -> dict:
        return {
            "tracker": "rinse_cutting_board",
            "target_name": self.target_name,
            "updates": self._updates,
            "required_consecutive_steps": self.required_consecutive_steps,
            "consecutive_duration": self._consecutive_duration,
            "max_consecutive": self._max_consecutive,
            "completion_latched": self._completion_latched,
            "water_on": self._water_on,
            "water_temp": self._water_temp,
            "water_hot": self._water_hot,
            "board_under_water": self._board_under_water,
            "terminal_faucet_off": self._terminal_faucet_off,
            "initial_handle_joint": self.initial_handle_joint,
            "handle_joint": self._handle_joint,
            "handle_close_error_rad": self._handle_close_error_rad,
            "handle_close_tolerance_rad": (
                self.handle_close_tolerance_rad
            ),
        }

    def update(self, env) -> bool:
        raw_env = _require_rinse_evidence(
            env,
            self.sink,
            self.cutting_board,
        )
        handle_state = self.sink.get_handle_state(raw_env)
        if not isinstance(handle_state, dict):
            raise ValueError("Rinse faucet state evidence is invalid")
        water_on = handle_state.get("water_on")
        if not isinstance(water_on, (bool, np.bool_)):
            raise ValueError("Rinse water-on evidence is invalid")
        try:
            water_temp = float(handle_state["water_temp"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Rinse water temperature is invalid") from error
        if not np.isfinite(water_temp):
            raise ValueError("Rinse water temperature is invalid")
        try:
            handle_joint = float(handle_state["handle_joint"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Rinse faucet handle evidence is invalid") from error
        if not np.isfinite(handle_joint):
            raise ValueError("Rinse faucet handle evidence is invalid")
        handle_close_error_rad = _wrapped_angle_error(
            handle_joint,
            self.initial_handle_joint,
        )
        board_under_water = self.sink.check_obj_under_water(
            raw_env,
            self.target_name,
        )
        if not isinstance(board_under_water, (bool, np.bool_)):
            raise ValueError(
                "Rinse board-under-water evidence is invalid"
            )

        water_on = bool(water_on)
        board_under_water = bool(board_under_water)
        water_hot = water_temp >= self.minimum_water_temp
        valid_rinse_observation = (
            water_on and water_hot and board_under_water
        )

        self._updates += 1
        self._water_on = water_on
        self._water_temp = water_temp
        self._water_hot = water_hot
        self._board_under_water = board_under_water
        self._handle_joint = handle_joint
        self._handle_close_error_rad = handle_close_error_rad
        if valid_rinse_observation:
            self._consecutive_duration += 1
            self._max_consecutive = max(
                self._max_consecutive,
                self._consecutive_duration,
            )
            if (
                self._consecutive_duration
                > self.required_consecutive_steps
            ):
                self._completion_latched = True
        elif not self._completion_latched:
            self._consecutive_duration = 0

        self._terminal_faucet_off = (
            self._completion_latched
            and not water_on
            and handle_close_error_rad
            <= self.handle_close_tolerance_rad
        )
        return self._terminal_faucet_off


def make_rinse_success_tracker(env_name: str, env):
    if env_name == "RinseCuttingBoard":
        return RinseSuccessTracker.from_restored_env(env)
    raise ValueError(f"{env_name}: no Rinse success tracker")


@contextlib.contextmanager
def _suppress_mutating_rinse_success(env_name: str, env):
    if env_name != "RinseCuttingBoard":
        yield
        return

    raw_env = env.env
    instance_attributes = getattr(raw_env, "__dict__", {})
    had_instance_override = "_check_success" in instance_attributes
    previous_override = instance_attributes.get("_check_success")
    raw_env._check_success = lambda: False
    try:
        yield
    finally:
        if had_instance_override:
            raw_env._check_success = previous_override
        else:
            del raw_env._check_success


def policy_env_step(env_name: str, env, action):
    with _suppress_mutating_rinse_success(env_name, env):
        return env.step(action)


def history_views(obs: dict) -> np.ndarray:
    """Return the memory views in the order used by the combined-350 store."""
    return np.ascontiguousarray(
        np.stack(
            (
                obs["video.robot0_agentview_left"],
                obs["video.robot0_agentview_right"],
                obs["video.robot0_eye_in_hand"],
            ),
            axis=0,
        )
    )


def state_vector(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ),
        axis=0,
    ).astype(np.float32)


def memory_element(obs: dict, previous_action: np.ndarray) -> dict:
    action = np.asarray(previous_action, dtype=np.float32)
    if action.shape != (12,):
        raise ValueError(
            f"memory action must have shape (12,), got {action.shape}"
        )
    return {
        "images": history_views(obs)[None, ...],
        "state": state_vector(obs)[None, ...],
        "actions": action[None, ...],
    }


def memory_prompt(task_description: str) -> str:
    return str(task_description)


def policy_prompt(task_description: str) -> str:
    return str(task_description).lower()


_MEMORY_STORE_FALLBACK_EPISODES = {
    "MemWashAndReturnLeft": 100,
    "MemWashAndReturnRight": 100,
    "MemWashAndReturnSameLocation": 150,
    "RinseCuttingBoard": 250,
    "PanTransfer": 300,
}


def load_memory_language_embedding(
    store_root: str | os.PathLike,
    env_name: str,
    demo: DemoInitialization | None,
) -> np.ndarray:
    combined_episode = (
        demo.combined_episode if demo is not None else None
    )
    if combined_episode is None:
        combined_episode = _MEMORY_STORE_FALLBACK_EPISODES.get(env_name)
    if combined_episode is None:
        raise ValueError(
            f"{env_name}: combined episode is required for language embedding"
        )
    path = (
        pathlib.Path(store_root)
        / "features"
        / f"episode_{combined_episode}"
        / "lang_emb.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"{env_name}: stored language embedding is missing: {path}"
        )
    language = np.load(path, allow_pickle=False).astype(
        np.float32,
        copy=False,
    )
    if language.shape != (2048,) or not np.all(np.isfinite(language)):
        raise ValueError(
            f"{env_name}: stored language embedding is invalid: "
            f"{path} shape={language.shape}"
        )
    return np.ascontiguousarray(language)


class MemoryActionStream:
    def __init__(self, alignment: str):
        if alignment not in (
            "causal_raw_previous",
            "stage2_selected_previous",
        ):
            raise ValueError(f"unknown memory action alignment: {alignment}")
        self.alignment = alignment
        self._previous = np.zeros(12, dtype=np.float32)

    def align(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (12,):
            raise ValueError(
                f"memory action must have shape (12,), got {action.shape}"
            )
        if self.alignment == "causal_raw_previous":
            return action
        aligned = self._previous.copy()
        self._previous = action.copy()
        return aligned


def exploration_policy_name(env_name: str) -> str | None:
    if env_name.startswith("MemRetrieveOils"):
        return "LRExplorationPolicy"
    if env_name.startswith("MemFruitInSink"):
        return "RotateExplorationPolicy"
    return None


def make_retrieve_exploration_policy(env_name: str, raw_env):
    policy_name = exploration_policy_name(env_name)
    if policy_name is None:
        raise ValueError(f"{env_name}: no retrieve exploration policy")
    from robocasa.models import exploration_policy

    return getattr(exploration_policy, policy_name)(raw_env)


def run_retrieve_exploration(
    env,
    obs: dict,
    client,
    *,
    max_steps: int,
    policy_factory,
    action_stream=None,
    on_step=None,
) -> tuple[dict, int, bool, bool]:
    if max_steps < 1:
        raise ValueError("retrieve exploration max_steps must be positive")
    raw_env = env.env
    raw_obs = raw_env._get_observations(force_update=True)
    policy = policy_factory(raw_env)
    policy.begin(init_obs=raw_obs)
    if action_stream is None:
        action_stream = MemoryActionStream("causal_raw_previous")
    if client is not None:
        client.add_buffer(
            memory_element(
                obs,
                action_stream.align(np.zeros(12, dtype=np.float32)),
            )
        )

    for step in range(max_steps):
        action = policy.get_action(raw_obs)
        if action is None:
            return obs, step, False, False
        step_result = raw_env.step(action)
        if len(step_result) == 4:
            raw_obs, _, terminated, _ = step_result
            truncated = False
        elif len(step_result) == 5:
            raw_obs, _, terminated, truncated, _ = step_result
        else:
            raise ValueError(
                "retrieve exploration env.step returned invalid result"
            )
        obs = env.get_observation(raw_obs)
        if client is not None:
            client.add_buffer(
                memory_element(obs, action_stream.align(action))
            )
        if on_step is not None:
            on_step(step, obs)
        if terminated or truncated:
            return obs, step + 1, bool(terminated), bool(truncated)

    raise RuntimeError(
        f"retrieve exploration did not finish within {max_steps} steps"
    )


def run_recorded_retrieve_exploration(
    env,
    obs: dict,
    client,
    demo,
    *,
    action_stream=None,
    on_step=None,
) -> tuple[dict, int, bool, bool]:
    states = np.asarray(demo.recorded_states)
    actions = np.asarray(demo.recorded_actions)
    transition = int(demo.recorded_transition)
    if (
        states.ndim != 2
        or actions.ndim != 2
        or actions.shape[1] != 12
        or states.shape[0] != actions.shape[0]
        or not 0 < transition < states.shape[0]
    ):
        raise ValueError("recorded Retrieve exploration is invalid")
    if action_stream is None:
        action_stream = MemoryActionStream("causal_raw_previous")

    raw_env = env.env
    for state_index in range(transition + 1):
        raw_env.sim.set_state_from_flattened(states[state_index])
        raw_env.sim.forward()
        raw_env.update_state()
        raw_obs = raw_env._get_observations(force_update=True)
        obs = env.get_observation(raw_obs)
        previous_action = (
            np.zeros(12, dtype=np.float32)
            if state_index == 0
            else actions[state_index - 1]
        )
        if client is not None:
            client.add_buffer(
                memory_element(
                    obs,
                    action_stream.align(previous_action),
                )
            )
        if on_step is not None:
            on_step(state_index, obs)
        if state_index < transition:
            raw_env.step(actions[state_index])
    return obs, transition, False, False


def run_recorded_wash_reset(
    env,
    obs: dict,
    client,
    demo,
    *,
    action_stream=None,
    on_step=None,
) -> tuple[dict, int, bool, bool]:
    """Add the exact recorded reset frame to memory without replaying actions.

    Wash-and-return has no separately scripted exploration phase in the
    training data. Replaying its task actions would expose the demonstration
    itself, so evaluation starts policy control at the recorded reset state.
    """
    states = np.asarray(demo.recorded_states)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("recorded Wash reset is invalid")
    if action_stream is None:
        action_stream = MemoryActionStream("causal_raw_previous")

    raw_env = env.env
    raw_env.sim.set_state_from_flattened(states[0])
    raw_env.sim.forward()
    raw_env.update_state()
    raw_obs = raw_env._get_observations(force_update=True)
    obs = env.get_observation(raw_obs)
    if client is not None:
        client.add_buffer(
            memory_element(
                obs,
                action_stream.align(np.zeros(12, dtype=np.float32)),
            )
        )
    if on_step is not None:
        on_step(0, obs)
    return obs, 0, False, False


def _joint_qpos_address(model, joint_name: str) -> int:
    legacy_lookup = getattr(model, "joint_name2id", None)
    if callable(legacy_lookup):
        joint_id = legacy_lookup(joint_name)
    else:
        joint_id = model.joint(joint_name).id
    return int(model.jnt_qposadr[joint_id])


def robot_pose(env, obs: dict) -> RobotPose:
    raw_env = env.env
    model = raw_env.sim.model
    qpos = np.asarray(raw_env.sim.data.qpos)
    torso = qpos[_joint_qpos_address(model, _TORSO_JOINT)]
    mobile_qpos = np.asarray(
        [
            qpos[_joint_qpos_address(model, joint_name)]
            for joint_name in _MOBILE_JOINTS
        ],
        dtype=np.float64,
    )
    arm_controller = raw_env.robots[0].composite_controller.part_controllers[
        "right"
    ]
    arm_indexes = np.asarray(arm_controller.qpos_index, dtype=np.int64)
    return RobotPose(
        eef_position=np.asarray(
            obs["state.end_effector_position_relative"],
            dtype=np.float64,
        ).copy(),
        eef_quaternion=np.asarray(
            obs["state.end_effector_rotation_relative"],
            dtype=np.float64,
        ).copy(),
        torso=float(torso),
        arm_qpos=np.asarray(qpos[arm_indexes], dtype=np.float64).copy(),
        mobile_qpos=mobile_qpos.copy(),
    )


def sweeten_placement_status(
    env,
    cube_names: tuple[str, ...],
) -> dict[str, tuple[bool, bool] | None]:
    raw_env = env.env
    saucepan = raw_env.objects["saucepan"]
    saucepan_position = np.asarray(
        raw_env.sim.data.body_xpos[raw_env.obj_body_id["saucepan"]],
        dtype=np.float64,
    )
    status = {}
    for cube_name in cube_names:
        try:
            cube_body_id, cube_geoms = _replayed_object_geometry(
                raw_env,
                cube_name,
            )
            grasped = raw_env._check_grasp(
                raw_env.robots[0].gripper["right"],
                cube_geoms,
            )
            cube_position = np.asarray(
                raw_env.sim.data.body_xpos[cube_body_id],
                dtype=np.float64,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            status[cube_name] = None
            continue
        if cube_position.shape != (3,) or not np.all(
            np.isfinite(cube_position)
        ):
            status[cube_name] = None
            continue
        inside = (
            raw_env.check_contact(cube_geoms, saucepan)
            and np.linalg.norm(cube_position[:2] - saucepan_position[:2])
            < saucepan.horizontal_radius * 0.7
        )
        status[cube_name] = (bool(inside), bool(grasped))
    return status


def sweeten_gripper_far(env) -> bool:
    from robocasa.utils import object_utils

    return bool(object_utils.gripper_obj_far(env.env, "saucepan"))


def policy_element(obs: dict, task_description: str, resize_size: int) -> dict:
    from openpi_client import image_tools

    def resize(key: str) -> np.ndarray:
        image = np.ascontiguousarray(obs[key])
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image, resize_size, resize_size)
        )

    return {
        "observation/image": resize("video.robot0_agentview_left"),
        "observation/wrist_image": resize("video.robot0_eye_in_hand"),
        "observation/right_image": resize("video.robot0_agentview_right"),
        "observation/state": state_vector(obs),
        "prompt": policy_prompt(task_description),
    }


def task_action(env_name: str, proposed_action: np.ndarray) -> np.ndarray:
    action = np.asarray(proposed_action)[:12].copy()
    if action.shape != (12,):
        raise ValueError(
            f"{env_name}: expected at least 12 action dimensions, "
            f"got {np.asarray(proposed_action).shape}"
        )
    if env_name == "SweetenHotChocolate":
        action[_MOBILE_ACTION_SLICE] = 0.0
    return action


def task_horizon(env_name: str, env, default_horizon: int) -> int:
    try:
        from robocasa.utils.dataset_registry_utils import get_task_horizon
    except Exception:  # pragma: no cover - old patched RoboCasa snapshots
        get_task_horizon = None

    if get_task_horizon is not None:
        try:
            horizon = int(get_task_horizon(env_name))
            if horizon > 0:
                return horizon
        except Exception:
            pass
    try:
        horizon = env.spec.max_episode_steps
        if horizon:
            return int(horizon)
    except Exception:
        pass
    return default_horizon


def task_success(env, info: dict) -> bool:
    try:
        return bool(env.unwrapped._check_success())
    except Exception:
        return bool(info.get("success", False))


def requires_task_specific_tracker(env_name: str) -> bool:
    return env_name in (
        _RETRIEVE_ENV_NAMES
        | _WASH_ENV_NAMES
        | {"PanTransfer", "RinseCuttingBoard", "SweetenHotChocolate"}
    )


def make_environment(env_name: str, args: Args):
    import gymnasium as gym

    return gym.make(
        f"robocasa/{env_name}",
        split=args.split,
        seed=args.seed,
    ).unwrapped


def evaluate_variant(
    env_name: str,
    num_trials: int,
    group_name: str,
    args: Args,
) -> dict:
    demo_references = ()
    recorded_references = recorded_scene_references(env_name, args)
    recovered_demos = ()
    if env_name in _RETRIEVE_ENV_NAMES:
        if not recorded_references:
            raise ValueError(
                f"{env_name}: Retrieve has no immutable lineage references"
            )
        missing_sources = [
            str(reference.dataset_path)
            for reference in recorded_references
            if not reference.dataset_path.is_file()
        ]
        if missing_sources:
            raise FileNotFoundError(
                f"{env_name}: recorded Retrieve sources are missing: "
                f"{missing_sources}"
            )
    elif env_name in _WASH_ENV_NAMES:
        if not recorded_references:
            raise ValueError(
                f"{env_name}: Wash has no recorded scene references"
            )
    elif env_name in MEMORY_ENV_NAMES:
        dataset_paths = memory_dataset_paths(args.memory_data_root, env_name)
        if not dataset_paths:
            raise FileNotFoundError(
                f"{env_name}: no demo HDF5 under {args.memory_data_root}"
            )
        demo_references = load_demo_references(dataset_paths)
        if not demo_references:
            raise ValueError(f"{env_name}: demo HDF5 files contain no episodes")
    elif (
        recorded_references
        and not recorded_references[0].dataset_path.is_dir()
    ):
        raise FileNotFoundError(
            f"{env_name}: recorded scene dataset is missing: "
            f"{recorded_references[0].dataset_path}"
        )
    if env_name == "SweetenHotChocolate":
        if not recorded_references:
            raise ValueError("Sweeten evaluation has no recovered references")
        recovered_demos = tuple(
            load_recorded_scene(reference)
            for reference in recorded_references
        )

    import tqdm
    from openpi_client import image_tools
    from robocasa.utils.env_utils import convert_action

    variant_dir = pathlib.Path(args.log_dir) / group_name / env_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    client = make_client(args.host, args.port)
    env = make_environment(env_name, args)
    horizon = task_horizon(env_name, env, args.default_horizon)
    successes = 0
    episode_stats = []

    for episode_idx in tqdm.tqdm(range(num_trials), desc=env_name):
        demo = None
        if demo_references:
            reference = demo_references[
                (args.seed + episode_idx) % len(demo_references)
            ]
            demo = load_demo_initialization(reference)
            obs = reset_to_demo(env, demo)
        elif recovered_demos:
            demo = recovered_demo_for_episode(
                recovered_demos,
                episode_idx,
                args.episode_offset,
            )
            obs = reset_to_demo(env, demo)
        elif recorded_references:
            reference = recorded_reference_for_episode(
                recorded_references,
                episode_idx,
                args.episode_offset,
            )
            demo = load_recorded_scene(reference)
            obs = reset_to_demo(env, demo)
        else:
            if env_name == "SweetenHotChocolate":
                raise RuntimeError(
                    "Sweeten must reset from a recovered scene bundle"
                )
            obs, _ = env.reset(seed=args.seed + episode_idx)
        task_description = str(obs["annotation.human.task_description"])
        sweeten_tracker = None
        sweeten_tracker_error = None
        retrieve_tracker = None
        retrieve_tracker_error = None
        wash_tracker = None
        wash_tracker_error = None
        pan_tracker = None
        pan_tracker_error = None
        rinse_tracker = None
        rinse_tracker_error = None
        if env_name == "SweetenHotChocolate":
            ep_meta = copy.deepcopy(env.env.get_ep_meta())
            contract, task_description = require_recovered_sweeten_contract(
                demo,
                ep_meta,
            )
            try:
                sweeten_cube_names = discover_sweeten_cube_names(
                    env,
                    contract,
                )
                sweeten_tracker = SweetenSuccessTracker(
                    contract,
                    robot_pose(env, obs),
                    cube_names=sweeten_cube_names,
                )
            except Exception as error:
                sweeten_tracker_error = error
        elif env_name in _RETRIEVE_ENV_NAMES:
            try:
                retrieve_tracker = make_retrieve_success_tracker(
                    env_name,
                    env,
                )
            except Exception as error:
                retrieve_tracker_error = error
        elif env_name in _WASH_ENV_NAMES:
            try:
                wash_tracker = make_wash_success_tracker(env_name, env)
            except Exception as error:
                wash_tracker_error = error
        elif env_name == "PanTransfer":
            try:
                pan_tracker = make_pan_transfer_success_tracker(
                    env_name,
                    env,
                )
            except Exception as error:
                pan_tracker_error = error
        elif env_name == "RinseCuttingBoard":
            try:
                rinse_tracker = make_rinse_success_tracker(
                    env_name,
                    env,
                )
            except Exception as error:
                rinse_tracker_error = error
        if env_name in _RETRIEVE_ENV_NAMES:
            if retrieve_tracker is None and retrieve_tracker_error is None:
                retrieve_tracker_error = RuntimeError(
                    f"{env_name}: Retrieve tracker initialization returned "
                    "no tracker"
                )
        elif env_name in _WASH_ENV_NAMES:
            if wash_tracker is None and wash_tracker_error is None:
                wash_tracker_error = RuntimeError(
                    f"{env_name}: Wash tracker initialization returned "
                    "no tracker"
                )
        elif env_name == "PanTransfer":
            if pan_tracker is None and pan_tracker_error is None:
                pan_tracker_error = RuntimeError(
                    "PanTransfer tracker initialization returned no tracker"
                )
        elif env_name == "RinseCuttingBoard":
            if rinse_tracker is None and rinse_tracker_error is None:
                rinse_tracker_error = RuntimeError(
                    "RinseCuttingBoard tracker initialization returned no "
                    "tracker"
                )
        elif env_name == "SweetenHotChocolate":
            if sweeten_tracker is None and sweeten_tracker_error is None:
                sweeten_tracker_error = RuntimeError(
                    "SweetenHotChocolate tracker initialization returned no "
                    "tracker"
                )
        action_plan = collections.deque()
        frames = (
            {view_name: [] for view_name in _VIDEO_VIEW_KEYS}
            if args.save_all_camera_videos
            else []
        )
        success = False
        if args.use_memory_buffer:
            prompt = memory_prompt(task_description)
            if args.memory_store:
                client.reset(
                    prompt=prompt,
                    lang_emb=load_memory_language_embedding(
                        args.memory_store,
                        env_name,
                        demo,
                    ),
                )
            else:
                client.reset(prompt=prompt)
        memory_action_stream = MemoryActionStream(
            args.memory_action_alignment
        )

        def capture_frame(
            step_index: int,
            observation: dict | None = None,
            *,
            force: bool = False,
        ) -> None:
            if args.save_video and (step_index % 2 == 0 or force):
                if args.save_all_camera_videos:
                    views = capture_video_views(
                        obs if observation is None else observation
                    )
                    for view_name, image in views.items():
                        frames[view_name].append(
                            image_tools.convert_to_uint8(image)
                        )
                else:
                    frames.append(
                        image_tools.convert_to_uint8(
                            np.ascontiguousarray(env.render())
                        )
                    )

        elapsed_steps = 0
        exploration_terminated = False
        exploration_truncated = False
        has_retrieve_exploration = (
            exploration_policy_name(env_name) is not None
            and retrieve_tracker_error is None
            and args.retrieve_exploration_mode != "none"
        )
        has_wash_recorded_reset = (
            env_name in _WASH_ENV_NAMES
            and wash_tracker_error is None
            and args.wash_exploration_mode == "recorded_state_replay"
            and demo is not None
            and demo.recorded_states is not None
        )
        if has_retrieve_exploration:
            np.random.seed(args.seed + episode_idx)
            if args.retrieve_exploration_mode == "recorded_state_replay":
                (
                    obs,
                    elapsed_steps,
                    exploration_terminated,
                    exploration_truncated,
                ) = run_recorded_retrieve_exploration(
                    env,
                    obs,
                    client if args.use_memory_buffer else None,
                    demo,
                    action_stream=memory_action_stream,
                    on_step=lambda step_index, step_obs: capture_frame(
                        step_index,
                        step_obs,
                    ),
                )
            else:
                exploration_steps = min(800, max(1, horizon // 2))
                (
                    obs,
                    elapsed_steps,
                    exploration_terminated,
                    exploration_truncated,
                ) = run_retrieve_exploration(
                    env,
                    obs,
                    client if args.use_memory_buffer else None,
                    max_steps=exploration_steps,
                    policy_factory=lambda raw_env: (
                        make_retrieve_exploration_policy(env_name, raw_env)
                    ),
                    action_stream=memory_action_stream,
                    on_step=lambda step_index, step_obs: capture_frame(
                        step_index,
                        step_obs,
                    ),
                )
        elif has_wash_recorded_reset:
            (
                obs,
                elapsed_steps,
                exploration_terminated,
                exploration_truncated,
            ) = run_recorded_wash_reset(
                env,
                obs,
                client if args.use_memory_buffer else None,
                demo,
                action_stream=memory_action_stream,
                on_step=lambda step_index, step_obs: capture_frame(
                    step_index,
                    step_obs,
                ),
            )

        policy_steps = 0
        steps = elapsed_steps
        if (
            retrieve_tracker_error is None
            and wash_tracker_error is None
            and pan_tracker_error is None
            and rinse_tracker_error is None
            and sweeten_tracker_error is None
            and not exploration_terminated
            and not exploration_truncated
        ):
            if (
                args.use_memory_buffer
                and not has_wash_recorded_reset
                and exploration_policy_name(env_name) is None
            ):
                client.add_buffer(
                    memory_element(
                        obs,
                        memory_action_stream.align(
                            np.zeros(12, dtype=np.float32)
                        ),
                    )
                )
            for policy_step in range(horizon):
                step = elapsed_steps + policy_step
                if not action_plan:
                    chunk = np.asarray(
                        client.infer(
                            policy_element(
                                obs,
                                task_description,
                                args.resize_size,
                            )
                        )["actions"]
                    )
                    if len(chunk) < args.replan_steps:
                        raise RuntimeError(
                            f"{env_name}: policy returned {len(chunk)} "
                            f"actions, need {args.replan_steps}"
                        )
                    action_plan.extend(chunk[: args.replan_steps])

                action_vector = task_action(
                    env_name,
                    action_plan.popleft(),
                )
                action = convert_action(action_vector)
                obs, _, terminated, truncated, info = policy_env_step(
                    env_name,
                    env,
                    action,
                )
                if args.use_memory_buffer:
                    client.add_buffer(
                        memory_element(
                            obs,
                            memory_action_stream.align(action_vector),
                        )
                    )
                policy_steps = policy_step + 1
                steps = elapsed_steps + policy_steps
                if retrieve_tracker is not None:
                    try:
                        success = retrieve_tracker.update(env)
                    except Exception as error:
                        retrieve_tracker_error = error
                        success = False
                elif wash_tracker is not None:
                    try:
                        success = wash_tracker.update(env)
                    except Exception as error:
                        wash_tracker_error = error
                        success = False
                elif pan_tracker is not None:
                    try:
                        success = pan_tracker.update(env)
                    except Exception as error:
                        pan_tracker_error = error
                        success = False
                elif rinse_tracker is not None:
                    try:
                        success = rinse_tracker.update(env)
                    except Exception as error:
                        rinse_tracker_error = error
                        success = False
                elif sweeten_tracker is not None:
                    try:
                        success = sweeten_tracker.update(
                            step,
                            sweeten_placement_status(
                                env,
                                sweeten_tracker.cube_names,
                            ),
                            robot_pose(env, obs),
                            gripper_far=sweeten_gripper_far(env),
                        )
                    except Exception as error:
                        sweeten_tracker_error = error
                        success = False
                else:
                    if requires_task_specific_tracker(env_name):
                        success = False
                    else:
                        success = task_success(env, info)

                capture_frame(step, obs, force=success)

                if (
                    success
                    or terminated
                    or truncated
                    or retrieve_tracker_error is not None
                    or wash_tracker_error is not None
                    or pan_tracker_error is not None
                    or rinse_tracker_error is not None
                    or sweeten_tracker_error is not None
                ):
                    break

        if success:
            successes += 1
        recovery_fields = recovery_result_fields(demo)
        retrieve_validation = None
        if env_name in _RETRIEVE_ENV_NAMES:
            retrieve_validation = (
                dict(retrieve_tracker.diagnostics)
                if retrieve_tracker is not None
                else {}
            )
            retrieve_validation.update(
                {
                    "exploration_steps": elapsed_steps,
                    "exploration_terminated": exploration_terminated,
                    "exploration_truncated": exploration_truncated,
                }
            )
            if retrieve_tracker_error is not None:
                retrieve_validation.update(
                    {
                        "error_type": type(retrieve_tracker_error).__name__,
                        "error": str(retrieve_tracker_error),
                    }
                )
        wash_validation = None
        if env_name in _WASH_ENV_NAMES:
            wash_validation = (
                dict(wash_tracker.diagnostics)
                if wash_tracker is not None
                else {}
            )
            if wash_tracker_error is not None:
                wash_validation.update(
                    {
                        "error_type": type(wash_tracker_error).__name__,
                        "error": str(wash_tracker_error),
                    }
                )
        pan_validation = None
        if env_name == "PanTransfer":
            pan_validation = (
                dict(pan_tracker.diagnostics)
                if pan_tracker is not None
                else {}
            )
            if pan_tracker_error is not None:
                pan_validation.update(
                    {
                        "error_type": type(pan_tracker_error).__name__,
                        "error": str(pan_tracker_error),
                    }
                )
        rinse_validation = None
        if env_name == "RinseCuttingBoard":
            rinse_validation = (
                dict(rinse_tracker.diagnostics)
                if rinse_tracker is not None
                else {}
            )
            if rinse_tracker_error is not None:
                rinse_validation.update(
                    {
                        "error_type": type(rinse_tracker_error).__name__,
                        "error": str(rinse_tracker_error),
                    }
                )
        sweeten_validation = None
        if env_name == "SweetenHotChocolate":
            sweeten_validation = (
                dict(sweeten_tracker.diagnostics)
                if sweeten_tracker is not None
                else {}
            )
            if sweeten_tracker_error is not None:
                sweeten_validation.update(
                    {
                        "error_type": type(sweeten_tracker_error).__name__,
                        "error": str(sweeten_tracker_error),
                    }
                )
        episode_stats.append(
            {
                "episode": episode_idx,
                "seed": args.seed + episode_idx,
                "success": success,
                "steps": steps,
                "exploration_steps": elapsed_steps,
                "policy_steps": policy_steps,
                "total_steps": steps,
                "task_description": task_description,
                "dataset_path": (
                    str(demo.dataset_path) if demo is not None else None
                ),
                "demo_key": demo.demo_key if demo is not None else None,
                "source_action_length": (
                    demo.action_length if demo is not None else None
                ),
                "source_layout_id": (
                    demo.ep_meta.get("layout_id") if demo is not None else None
                ),
                "source_style_id": (
                    demo.ep_meta.get("style_id") if demo is not None else None
                ),
                "retrieve_validation": retrieve_validation,
                "wash_validation": wash_validation,
                "pan_validation": pan_validation,
                "rinse_validation": rinse_validation,
                "sweeten_validation": sweeten_validation,
                "recorded_scene_lineage": recorded_scene_lineage(demo),
                **(
                    recovery_fields
                    if (
                        recovery_fields["combined_episode"] is not None
                        or recovery_fields["source_root"] is not None
                    )
                    else {}
                ),
            }
        )

        if args.save_video and frames:
            import imageio

            suffix = "success" if success else "failure"
            if args.save_all_camera_videos:
                for view_name, view_frames in frames.items():
                    imageio.mimwrite(
                        variant_dir
                        / (
                            f"rollout_{episode_idx:03d}_{view_name}_"
                            f"{suffix}.mp4"
                        ),
                        view_frames,
                        fps=20,
                    )
            else:
                imageio.mimwrite(
                    variant_dir / f"rollout_{episode_idx:03d}_{suffix}.mp4",
                    frames,
                    fps=20,
                )

        if recovery_fields["combined_episode"] is None:
            logging.info(
                "[%s] %d/%d success=%s steps=%d running_sr=%.3f",
                env_name,
                episode_idx + 1,
                num_trials,
                success,
                steps,
                successes / (episode_idx + 1),
            )
        else:
            logging.info(
                "[%s] %d/%d success=%s steps=%d running_sr=%.3f "
                "combined_episode=%s source_episode=%s goal_count=%s "
                "selected_attempt=%s layout_attempt=%s score_summary=%s "
                "codec_envelope=%s "
                "generation_commit=%s recovery_input_hashes=%s "
                "recovery_output_hashes=%s",
                env_name,
                episode_idx + 1,
                num_trials,
                success,
                steps,
                successes / (episode_idx + 1),
                recovery_fields["combined_episode"],
                recovery_fields["source_episode"],
                recovery_fields["goal_count"],
                recovery_fields["selected_attempt"],
                recovery_fields["layout_attempt"],
                recovery_fields["score_summary"],
                recovery_fields["codec_noise_envelope"],
                recovery_fields["generation_commit"],
                recovery_fields["recovery_input_hashes"],
                recovery_fields["recovery_output_hashes"],
            )

    env.close()
    stats = {
        "task_group": group_name,
        "env_name": env_name,
        "split": args.split,
        "num_episodes": num_trials,
        "num_successes": successes,
        "success_rate": successes / num_trials,
        "horizon": horizon,
        "episodes": episode_stats,
    }
    with (variant_dir / "stats.json").open("w") as file:
        json.dump(stats, file, indent=2)
    return stats


def evaluate_group(group_name: str, args: Args) -> dict:
    variants = [
        evaluate_variant(env_name, trials, group_name, args)
        for env_name, trials in group_variants(
            group_name,
            pilot_trials=args.pilot_trials,
        )
    ]
    total = sum(item["num_episodes"] for item in variants)
    successes = sum(item["num_successes"] for item in variants)
    stats = {
        "task_group": group_name,
        "num_episodes": total,
        "num_successes": successes,
        "success_rate": successes / total,
        "variants": variants,
    }
    group_dir = pathlib.Path(args.log_dir) / group_name
    with (group_dir / "group_stats.json").open("w") as file:
        json.dump(stats, file, indent=2)
    return stats


def main(args: Args) -> None:
    import gymnasium as gym
    import robocasa  # noqa: F401 - registers robocasa/* gym environments
    import robocasa.environments.kitchen.memory  # noqa: F401
    from robocasa.wrappers.gym_wrapper import create_robocasa_gym_env_class

    register_memory_gym_envs(gym.registry, create_robocasa_gym_env_class)

    if not args.log_dir:
        raise ValueError("--log-dir is required")
    groups = [name.strip() for name in args.task_groups.split(",") if name.strip()]
    if not groups:
        raise ValueError("--task-groups is required")
    unknown = sorted(
        set(groups) - set(TASK_GROUPS) - set(PILOT_TASK_GROUPS)
    )
    if unknown:
        raise ValueError(f"unknown task groups: {unknown}")

    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.info(
        "Starting groups=%s port=%d run=%s",
        groups,
        args.port,
        datetime.now().isoformat(),
    )
    results = [evaluate_group(group, args) for group in groups]
    with (pathlib.Path(args.log_dir) / f"worker_stats_{args.port}.json").open("w") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(parse_args())
