import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import dataclasses

import openpi.models.model as _model
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms

import mme_vla_suite.policies.policy as _policy
import mme_vla_suite.training.config as _config


def _effective_history_config(
    checkpoint_dir: pathlib.Path,
    requested_history_config: str | None,
) -> str | None:
    """Resolve checkpoint metadata without depending on its original filesystem."""
    metadata_path = checkpoint_dir.parent / "history_config.txt"
    if not metadata_path.exists():
        return requested_history_config

    recorded = metadata_path.read_text().strip() or None
    if os.environ.get("IMPRINT_PREFER_REQUESTED_HISTORY_CONFIG") == "1":
        if recorded != requested_history_config:
            logging.info(
                "Using requested history config %r instead of checkpoint metadata %r",
                requested_history_config,
                recorded,
            )
        return requested_history_config

    if recorded is None:
        return None

    recorded_path = pathlib.Path(recorded).expanduser()
    bundled_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "models/config/robomme"
        / recorded_path.name
    )
    if recorded_path.is_absolute() and not recorded_path.exists():
        if bundled_path.exists():
            logging.warning(
                "Checkpoint history config %s is unavailable; using bundled %s",
                recorded_path,
                bundled_path.name,
            )
            return bundled_path.name
        if requested_history_config is not None:
            logging.warning(
                "Checkpoint history config %s is unavailable; using requested %s",
                recorded_path,
                requested_history_config,
            )
            return requested_history_config
    return recorded


def _apply_wan_chunk_alignment(history_config, checkpoint_dir, requested):
    """Set a requested Wan chunk alignment and refuse a mismatch with the checkpoint record.

    The memory works only with the alignment that it was trained with, so this check also
    applies when IMPRINT_PREFER_REQUESTED_HISTORY_CONFIG=1 selects the requested YAML.
    """
    from omegaconf import open_dict

    from mme_vla_suite.shared import wan_alignment

    cbm = None if history_config is None else getattr(history_config, "cosmo_bottleneck_memory", None)
    if requested is not None:
        requested = wan_alignment.resolve_wan_chunk_alignment(requested)
        if cbm is not None:
            with open_dict(cbm):
                cbm.wan_chunk_alignment = requested
        elif requested != wan_alignment.NATIVE:
            raise ValueError(f"wan_chunk_alignment={requested!r} needs a cosmo_bottleneck_memory")
    run_dir = checkpoint_dir.parent
    wan_alignment.require_recorded_alignment(
        wan_alignment.read_alignment_record(run_dir),
        cbm,
        where=run_dir / wan_alignment.ALIGNMENT_RECORD,
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    seed: int = 42,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    wan_chunk_alignment: str | None = None,
) -> _policy.MME_VLA_Policy:

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    repack_transforms = repack_transforms or transforms.Group()

    logging.info("Checking history config")
    history_config = _effective_history_config(
        checkpoint_dir,
        train_config.model.history_config,
    )
    if train_config.model.history_config != history_config:
        print(
            f" == You are using {train_config.model.history_config}, "
            f"changing to {history_config} =="
        )
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model,
                history_config=history_config,
                use_history=history_config is not None,
            ),
        )


    logging.info("Loading model...")
    _restored = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(_restored)
    # None keeps the alignment of the history config. See shared/wan_alignment.py.
    _apply_wan_chunk_alignment(model.history_config, checkpoint_dir, wan_chunk_alignment)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    print("Training config: ", train_config)
    print("Data config: ", data_config)

    return _policy.MME_VLA_Policy(
        model,
        seed=seed,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm
    )
