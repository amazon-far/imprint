"""Training entrypoint for the ROBOMME memory VLA (mme_vla_suite).

Stock `scripts/train.py` resolves configs from openpi's registry and builds
openpi's LeRobot data loader. The mme suite instead:
  - registers its TrainConfigs in `mme_vla_suite.training.config` (pi05_baseline,
    mme_vla_suite), with a `dataset_path` field + `model.history_config`,
  - loads data via `mme_vla_suite.training.dataloader.create_data_loader(
        dataset_path, data_config, history_config, action_horizon, batch_size, ...)`
    which reads the offline RoboMMEDataset feature store.

This script reuses stock train.py's train_step / init_train_state / wandb logic
verbatim (they are config-agnostic) and only swaps the config CLI and the data
loader construction. Run:

  python scripts/train_mme.py mme_vla_suite \
      --model.history_config=perceptual-framesamp-modul.yaml \
      --dataset_path=<store> --exp-name=... --batch_size=64 \
      --num_train_steps=80000 --fsdp_devices=4 [--overwrite]
"""
import functools
import logging
import platform

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

import openpi.training.checkpoints as _checkpoints
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

# Reuse stock train.py's helpers (config-agnostic). scripts/ isn't a package, so
# load the sibling train.py by file path.
import importlib.util as _ilu
import pathlib as _pl
_train_path = _pl.Path(__file__).resolve().parent / "train.py"
_spec = _ilu.spec_from_file_location("openpi_train_helpers", _train_path)
_train = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_train)
init_logging = _train.init_logging
init_wandb = _train.init_wandb
# Use the UNDECORATED functions: openpi wraps these with @at.typecheck annotating
# config: openpi.TrainConfig, which rejects the mme TrainConfig (a distinct class).
# __wrapped__ is the original function without the runtime typecheck.
init_train_state = getattr(_train.init_train_state, "__wrapped__", _train.init_train_state)

# The HistoryPi0 memory model's compute_loss returns (per_token_loss, stats) — a
# tuple — whereas stock train_step's loss_fn assumes a bare array and calls
# jnp.mean on it. Define an mme-aware train_step that unpacks the tuple. Mirrors
# stock train.py:train_step exactly otherwise.
import dataclasses as _dataclasses
import jax as _jax
import flax.nnx as _nnx
import optax as _optax
import openpi.shared.nnx_utils as _nnx_utils
import openpi.training.utils as _training_utils


def _apply_sample_loss_weight(chunked_loss, sample_loss_weight):
    if sample_loss_weight is None:
        return chunked_loss
    weight = jnp.asarray(sample_loss_weight, dtype=chunked_loss.dtype)
    while weight.ndim < chunked_loss.ndim:
        weight = weight[..., None]
    return chunked_loss * weight


def train_step(config, rng, state, batch):
    model = _nnx.merge(state.model_def, state.params)
    model.train()

    # train_fraction = step/num_train_steps in [0,1]; used by the aux-lambda annealer
    # (compute_loss ignores it unless the model config sets aux_anneal=True). Traced scalar -> jit-safe.
    _nts = float(getattr(config, "num_train_steps", 0) or 0)
    train_fraction = (state.step / _nts) if _nts > 0 else None

    def loss_fn(model, rng, observation, actions):
        out = model.compute_loss(rng, observation, actions, train=True, train_fraction=train_fraction)
        chunked_loss = out[0] if isinstance(out, tuple) else out  # (loss, stats) or loss
        # capture aux/loss stats (dict) so they can be logged; {} when compute_loss returns bare loss
        aux_stats = dict(out[1]) if (isinstance(out, tuple) and isinstance(out[1], dict)) else {}
        sample_loss_weight = getattr(observation, "sample_loss_weight", None)
        chunked_loss = _apply_sample_loss_weight(chunked_loss, sample_loss_weight)
        if sample_loss_weight is not None:
            aux_stats["loss/sample_weight_mean"] = jnp.asarray(sample_loss_weight).mean()
        return jnp.mean(chunked_loss), aux_stats

    train_rng = _jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = _nnx.DiffState(0, config.trainable_filter)
    (loss, aux_stats), grads = _nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = _optax.apply_updates(params, updates)

    _nnx.update(model, new_params)
    new_params = _nnx.state(model)

    new_state = _dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = _dataclasses.replace(
            new_state,
            ema_params=_jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = _nnx.state(
        model,
        _nnx.All(
            _nnx.Param,
            _nnx.Not(_nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": _optax.global_norm(grads),
        "param_norm": _optax.global_norm(kernel_params),
    }
    # surface aux/loss_* scalar stats (e.g. loss/aux_total, loss/aux_align, aux/lambda) to wandb
    for _k, _v in (aux_stats or {}).items():
        try:
            info[_k] = jnp.asarray(_v).mean()
        except Exception:
            pass
    return new_state, info

# mme registry + mme data loader (NOT openpi's).
from mme_vla_suite.training import config as _mme_config
from mme_vla_suite.training import dataloader as _mme_dataloader


def main(config):
    init_logging()
    logging.info(f"Running ROBOMME memory training on: {platform.node()}")
    logging.info(f"config={config.name} exp={config.exp_name} "
                 f"history_config={getattr(config.model, 'history_config', None)} "
                 f"dataset_path={config.dataset_path}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by device count {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    history_config_name = getattr(config.model, "history_config", None)
    if history_config_name is not None:
        (_pl.Path(config.checkpoint_dir) / "history_config.txt").write_text(
            str(history_config_name),
            encoding="utf-8",
        )
        # The policy load refuses an evaluation config with another Wan chunk alignment.
        from mme_vla_suite.models.config.utils import get_history_config
        from mme_vla_suite.shared.wan_alignment import record_training_alignment
        from openpi.shared import download as _download

        _stage1_ckpt = getattr(config.weight_loader, "stage1_ckpt", None)
        record_training_alignment(
            config.checkpoint_dir,
            getattr(get_history_config(history_config_name), "cosmo_bottleneck_memory", None),
            stage1_ckpt=_download.maybe_download(_stage1_ckpt) if _stage1_ckpt else None,
        )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # ---- mme data loader: builds the DataConfig from the factory, then the
    # RoboMMEDataset feature-store loader. ----
    data_config = config.data.create(config.assets_dirs, config.model)
    # Norm stats live INSIDE the built store at <dataset_path>/meta/assets/robomme
    # (computed by compute_store_normstats.py with quantiles, since pi05 uses
    # quantile norm and the baseline ckpt stats lack q01/q99). Load them directly
    # and inject, overriding whatever (possibly None) the factory resolved.
    import dataclasses as _dc
    import openpi.shared.normalize as _normalize
    _ns_dir = _pl.Path(config.dataset_path) / "meta" / "assets" / "robomme"
    if (_ns_dir / "norm_stats.json").exists():
        _ns = _normalize.load(_ns_dir)
        data_config = _dc.replace(data_config, norm_stats=_ns)
        logging.info(f"Loaded store norm_stats from {_ns_dir} "
                     f"(state dim={_ns['state'].mean.shape}, actions dim={_ns['actions'].mean.shape})")
    else:
        logging.warning(f"No store norm_stats at {_ns_dir}; using factory norm_stats "
                        f"(may crash if None / lacks quantiles).")
    data_loader = _mme_dataloader.create_data_loader(
        dataset_path=config.dataset_path,
        data_config=data_config,
        history_config=getattr(config.model, "history_config", None),
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        late_episode_loss_weight=getattr(config.model, "late_episode_loss_weight", 1.0),
        late_episode_loss_fraction=getattr(config.model, "late_episode_loss_fraction", 0.5),
        length_bucketed_batches=config.length_bucketed_batches,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized mme data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log first-batch images for sanity (HistAugObservation has .images like Observation).
    try:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)
    except Exception as e:  # don't let logging break the run
        logging.warning(f"camera_views log skipped: {e}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if config.name in {
        "mme_wan_stage2_pi0_robomme_imitation4_memft_freeze_language",
        "mme_wan_stage2_pi05_robomme_imitation4_memft_freeze_language",
    }:
        path_text = lambda path: "/".join(map(str, path))
        frozen_paths = [
            path_text(path)
            for path, _ in train_state.params.filter(config.freeze_filter).flat_state().items()
        ]
        trainable_paths = [
            path_text(path)
            for path, _ in train_state.params.filter(config.trainable_filter).flat_state().items()
        ]

        def is_action_expert(path):
            return "/llm/" in path and "_1/" in path

        def is_memory_seam(path):
            return any(
                name in path
                for name in (
                    "mem_attn",
                    "mem_rms_norm",
                    "q_einsum_mem",
                    "kv_einsum_mem",
                    "out_einsum_mem",
                )
            )

        assert any("/img/" in path for path in frozen_paths)
        assert any("/llm/embedder/" in path for path in frozen_paths)
        assert any(
            "/llm/layers/" in path
            and not is_action_expert(path)
            and not is_memory_seam(path)
            for path in frozen_paths
        )
        assert not any(is_action_expert(path) for path in frozen_paths)
        assert not any(is_memory_seam(path) for path in frozen_paths)
        assert any(is_action_expert(path) for path in trainable_paths)
        assert any(is_memory_seam(path) for path in trainable_paths)
        assert any("mem_encoder/" in path for path in trainable_paths)
        logging.info(
            "FREEZE_AUDIT image_frozen=true language_encoder_frozen=true "
            "action_expert_trainable=true memory_encoder_trainable=true "
            "memory_seam_trainable=true frozen_leaves=%d trainable_leaves=%d",
            len(frozen_paths),
            len(trainable_paths),
        )

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_mme_config.cli())
