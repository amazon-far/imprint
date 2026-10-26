import dataclasses
import os
import pathlib
from collections.abc import Mapping

import numpy as np

from openpi import transforms
import mme_vla_suite.policies.policy_config as mme_policy_config
from mme_vla_suite.training import config as mme_config


CAMERA_ORDER = ("head_camera", "left_camera", "right_camera")
ACTION_WIDTH = 16
DEFAULT_HISTORY_CONFIG = (
    "cosmo-wan-bottleneck-modul-rmbench-cover-nl256-s2nojitter.yaml"
)


def native_to_model(native: np.ndarray) -> np.ndarray:
    native = np.asarray(native, dtype=np.float32)
    if native.shape[-1] != 14:
        raise ValueError(f"expected 14 native dimensions, got {native.shape}")
    model = np.zeros((*native.shape[:-1], ACTION_WIDTH), dtype=np.float32)
    model[..., :6] = native[..., :6]
    model[..., 7:13] = native[..., 7:13]
    model[..., 14] = native[..., 6]
    model[..., 15] = native[..., 13]
    return model


def model_to_native(model: np.ndarray) -> np.ndarray:
    model = np.asarray(model, dtype=np.float32)
    if model.shape[-1] < ACTION_WIDTH:
        raise ValueError(f"expected at least 16 model dimensions, got {model.shape}")
    return np.concatenate(
        (
            model[..., :6],
            model[..., 14:15],
            model[..., 7:13],
            model[..., 15:16],
        ),
        axis=-1,
    )


def camera_stack(cameras: Mapping[str, np.ndarray]) -> np.ndarray:
    missing = [name for name in CAMERA_ORDER if name not in cameras]
    if missing:
        raise KeyError(f"missing RM-Bench cameras: {missing}")
    return np.stack([np.asarray(cameras[name]) for name in CAMERA_ORDER], axis=0)


def policy_observation(
    cameras: Mapping[str, np.ndarray],
    native_state: np.ndarray,
    prompt: str,
) -> dict:
    return {
        "observation/image": np.asarray(cameras["head_camera"]),
        "observation/wrist_image": np.asarray(cameras["left_camera"]),
        "observation/right_image": np.asarray(cameras["right_camera"]),
        "observation/state": native_to_model(native_state),
        "prompt": prompt,
    }


@dataclasses.dataclass(frozen=True)
class RMBenchOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_WIDTH])}


@dataclasses.dataclass(frozen=True)
class RMBenchDataConfig(mme_config.RoboMMEDataConfig):
    def create(self, assets_dirs, model_config):
        data_config = super().create(assets_dirs, model_config)
        output_transforms = data_config.data_transforms.outputs
        if (
            not output_transforms
            or type(output_transforms[-1]).__name__ != "RoboMMEOutputs"
        ):
            raise RuntimeError(
                "expected RoboMMEOutputs as the final training output transform"
            )
        return dataclasses.replace(
            data_config,
            data_transforms=transforms.Group(
                inputs=data_config.data_transforms.inputs,
                # Keep AbsoluteActions so model-predicted deltas are converted
                # back to joint targets. Only replace RoboMME's 8-D slicer.
                outputs=(*output_transforms[:-1], RMBenchOutputs()),
            ),
        )


def build_train_config(
    config_name: str,
    history_config: str | None = DEFAULT_HISTORY_CONFIG,
    *,
    prediction_horizon: int = 50,
):
    train_config = mme_config.get_config(config_name)
    data = train_config.data
    return dataclasses.replace(
        train_config,
        model=dataclasses.replace(
            train_config.model,
            action_horizon=prediction_horizon,
            history_config=history_config,
            use_history=history_config is not None,
        ),
        data=RMBenchDataConfig(
            repo_id=data.repo_id,
            assets=data.assets,
            base_config=data.base_config,
        ),
    )


def ensure_checkpoint_history(
    checkpoint_dir: pathlib.Path, history_config: str | None
) -> None:
    path = checkpoint_dir.parent / "history_config.txt"
    if not path.exists():
        return
    recorded = path.read_text().strip() or None
    if recorded == history_config:
        return
    if os.environ.get("IMPRINT_PREFER_REQUESTED_HISTORY_CONFIG") == "1":
        return
    if history_config is None:
        raise ValueError(f"{path} records memory for a no-memory checkpoint")
    if recorded != history_config:
        raise ValueError(f"{path} does not match requested history config")


def create_policy(
    checkpoint_dir: pathlib.Path,
    *,
    config_name: str = "mme_wan_stage2_base",
    history_config: str | None = DEFAULT_HISTORY_CONFIG,
    prediction_horizon: int = 50,
    seed: int = 42,
    wan_chunk_alignment: str | None = None,
):
    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    ensure_checkpoint_history(checkpoint_dir, history_config)
    train_config = build_train_config(
        config_name,
        history_config,
        prediction_horizon=prediction_horizon,
    )
    return mme_policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        seed=seed,
        wan_chunk_alignment=wan_chunk_alignment,
    )


def server_metadata(
    *,
    config_name: str,
    history_config: str,
    prediction_horizon: int,
) -> dict:
    return {
        "config": config_name,
        "history_config": history_config,
        "prediction_horizon": prediction_horizon,
        "action_width": ACTION_WIDTH,
        "camera_order": list(CAMERA_ORDER),
    }
