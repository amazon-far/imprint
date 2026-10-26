"""Stage-1 bottleneck pretraining trainer.

Pretrains the 256-token memory bottleneck + reconstruction decoder via the
composite alignment loss (doc §6.4). Stage-1 = OPTION B: the bottleneck FiLM is
ON (memory is language-conditioned); the decoder is language-free. Only the
bottleneck + decoder train; π0.5 / tokenizer are absent and targets are
precomputed & stop-grad.

Usage (once store is built):
    JAX_PLATFORMS=cuda \\
        python -m mme_vla_suite.scripts.pretrain_bottleneck \\
        --config models/config/robomme/cosmo-wan-bottleneck-modul.yaml \\
        --store /path/to/cosmo_bottleneck_store \\
        --num_steps 80000 --batch_size 256 --fsdp_devices 4 \\
        --checkpoint_dir /path/to/checkpoints/stage1_bottleneck
"""
import argparse
import bisect
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import queue
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import os as _os
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_utils

# --- OPTION-1: z-score normalize action/proprio DECODE targets with the VLA's norm_stats
# (mean/std), so the reconstruction loss is scale-consistent with the policy and action
# vs proprio are auto-balanced. q01/q99 absent -> use openpi _normalize (x-mean)/(std+1e-6).
import json as _json
_NORM_MEAN_STD = {}
_NORM_TARGETS = _os.environ.get("NORM_TARGETS","0") == "1"  # OPTION: z-score decode targets (default OFF=original raw)
def _load_norm(store):
    import numpy as _np
    p = _os.path.join(store, "meta", "assets", "robomme", "norm_stats.json")
    ns = _json.load(open(p))["norm_stats"]
    out = {}
    for key, dim in (("state", 16), ("actions", 12)):
        mean = _np.asarray(ns[key]["mean"], _np.float32)[:dim]
        std  = _np.asarray(ns[key]["std"],  _np.float32)[:dim]
        out[key] = (mean, std)
    return out
import optax
from flax import nnx
from omegaconf import OmegaConf

import openpi.training.sharding as sharding
from mme_vla_suite.training.pretrain_loss import make_stage1_modules, composite_alignment_loss
from mme_vla_suite.shared.variant_actions import load_variant_causal_actions
from mme_vla_suite.shared.visual_streams import load_dino_manifest, load_visual_stream
from mme_vla_suite.shared.sample_locations import sample_location
from mme_vla_suite.training.length_bucketing import (
    build_length_bucketed_epoch_batches,
    max_steps_for_latent_bucket,
)

logger = logging.getLogger(__name__)

# Default composite-loss weights (doc §6.4). Reward is OFF until RoboMETER labels
# exist; set --w_reward>0 (and provide reward_target batches) to activate it.
DEFAULT_WEIGHTS = {"align": 1.0, "action": 1.0, "proprio": 1.0, "reward": 0.0}


def _pool_square_latents(lat, pool_hw, source_side=8):
    """Pool an unpooled square grid, or preserve an already-pooled grid."""
    pool_hw = int(pool_hw)
    pooled_spatial = pool_hw * pool_hw
    if lat.shape[1] % pooled_spatial == 0 and lat.shape[1] < source_side * source_side:
        return lat

    spatial = source_side * source_side
    assert lat.shape[1] % spatial == 0, (
        f"unexpected spatial layout {lat.shape} (grid {source_side}x{source_side})"
    )
    assert source_side % pool_hw == 0, (
        f"pool {pool_hw} must divide source grid {source_side}"
    )
    ncur = lat.shape[0]
    views = lat.shape[1] // spatial
    latent_dim = lat.shape[-1]
    block = source_side // pool_hw
    grid = lat.reshape(ncur, views, source_side, source_side, latent_dim)
    grid = grid.reshape(
        ncur,
        views,
        pool_hw,
        block,
        pool_hw,
        block,
        latent_dim,
    ).mean(axis=(3, 5))
    return grid.reshape(ncur, views * pooled_spatial, latent_dim)


def _configure_compact_alignment(config, store):
    """Match the Stage-1 decoder to a compact store's alignment token count."""
    if not store:
        return
    cbm = getattr(config, "cosmo_bottleneck_memory", None)
    if str(getattr(cbm, "visual_feature_stream", "siglip")) == "dinov3":
        # This manifest describes the inherited compact SigLIP target which was
        # replaced by dinov3.npy; DINO geometry comes from dino_store_manifest.
        return
    manifest_path = os.path.join(store, "meta", "stage1_compact.json")
    if not os.path.isfile(manifest_path):
        return
    with open(manifest_path) as handle:
        manifest = _json.load(handle)
    output_tail = manifest["transforms"]["alignment"]["output_tail"]
    align_dim = int(OmegaConf.select(config, "cosmo_decoder.align_dim", default=2048))
    if len(output_tail) != 2 or int(output_tail[1]) != align_dim:
        raise ValueError(f"invalid compact alignment tail: {output_tail}")
    align_num_patches = int(output_tail[0])
    config.cosmo_decoder.align_num_patches = align_num_patches
    logger.info(
        "compact store alignment patches=%d from %s",
        align_num_patches,
        manifest_path,
    )


def _configure_visual_stream(config, store):
    """Configure the copied DINO trainer from the immutable store manifest."""
    cbm = getattr(config, "cosmo_bottleneck_memory", None)
    stream = str(getattr(cbm, "visual_feature_stream", "siglip")) if cbm else "siglip"
    if stream != "dinov3":
        return
    if not store:
        raise ValueError("DINOv3 training requires --store")

    manifest = load_dino_manifest(store)
    config.memory_feature.img.input_dim = manifest["feature_dim"]
    config.cosmo_bottleneck_memory.siglip_dim = manifest["feature_dim"]
    config.cosmo_decoder.align_dim = manifest["feature_dim"]
    config.cosmo_decoder.align_num_patches = (
        manifest["num_views"] * manifest["tokens_per_view"]
    )
    logger.info(
        "DINOv3 visual stream: dim=%d views=%d tokens/view=%d",
        manifest["feature_dim"],
        manifest["num_views"],
        manifest["tokens_per_view"],
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage-1 bottleneck pretraining (composite reconstruction loss).")
    p.add_argument("--config", required=True, help="Path to the wan/modulation variant config yaml.")
    p.add_argument("--store", default=None, help="Path to the Part-2 cosmo_bottleneck store (Part-3 dataloader source).")
    p.add_argument("--num_steps", type=int, default=80000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--fsdp_devices", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--w_align", type=float, default=DEFAULT_WEIGHTS["align"])
    p.add_argument("--w_action", type=float, default=DEFAULT_WEIGHTS["action"])
    p.add_argument("--w_proprio", type=float, default=DEFAULT_WEIGHTS["proprio"])
    p.add_argument("--w_reward", type=float, default=DEFAULT_WEIGHTS["reward"])
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume_stage1_ckpt", default=None,
                        help="Optional stage1_step*.npz for model-only continuation.")
    resume.add_argument("--resume_full_state", default=None,
                        help="Optional full_state/<step> checkpoint restoring model, optimizer, and step.")
    p.add_argument("--start_step", type=int, default=0,
                   help="Absolute step represented by --resume_stage1_ckpt; checkpoints/logs continue from here.")
    p.add_argument("--lr_decay_steps", type=int, default=None,
                   help="Cosine-decay horizon; defaults to --num_steps and can remain fixed when extending a run.")
    p.add_argument("--project_name", default="Robocasa")
    p.add_argument("--exp_name", default=None, help="wandb run name (distinguishes the 6 Stage-1 ablations).")
    p.add_argument("--wandb_enabled", action="store_true")
    p.add_argument("--order_probe_labels", default=None, help="JSON labels from make_robodojo_order_labels.py.")
    p.add_argument("--order_probe_every", type=int, default=5000, help="Run bottleneck color-order probe every N steps.")
    p.add_argument("--order_probe_folds", type=int, default=5)
    p.add_argument("--order_probe_steps_per_episode", type=int, default=8)
    p.add_argument("--order_probe_out_dir", default=None)
    p.add_argument("--latent_buckets", default=None,
                   help="Optional comma-separated latent padding buckets. If omitted, infer small buckets for pooled offline stores.")
    p.add_argument(
        "--length_bucketed_batches",
        action="store_true",
        help=(
            "Group samples by a conservative step-index history bound so latent "
            "and action/proprio padding are exact and no sample is truncated."
        ),
    )
    p.add_argument("--max_recent_tokens", type=int, default=0)
    p.add_argument("--max_step_tokens", type=int, default=0)
    p.add_argument(
        "--disable_recent_memory",
        action="store_true",
        help=(
            "Exclude recent visual features from memory input while retaining "
            "visual alignment targets and their loss."
        ),
    )
    p.add_argument(
        "--episode_cache_size",
        type=int,
        default=0,
        help="Number of episode feature bundles to cache; 0 preserves the original loader.",
    )
    p.add_argument(
        "--gather_workers",
        type=int,
        default=1,
        help="Thread count for offline sample gathering. One preserves the original RNG order.",
    )
    p.add_argument(
        "--prefetch_batches",
        type=int,
        default=0,
        help="Number of complete host batches prepared by a background thread.",
    )
    p.add_argument(
        "--dino_transport_dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Host/transfer dtype for DINO tensors; loss math remains float32.",
    )
    p.add_argument(
        "--module_compute_dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help=(
            "Compute dtype for Stage-1 bottleneck and decoder matmuls. "
            "Parameters, normalization, recurrent state, and loss math remain float32."
        ),
    )
    p.add_argument("--profile_every", type=int, default=0)
    p.add_argument("--profile_warmup_steps", type=int, default=10)
    p.add_argument("--profile_blocking_steps", type=int, default=0)
    p.add_argument("--profile_jsonl", default=None)
    p.add_argument("--skip_final_checkpoint", action="store_true")
    return p.parse_args(argv)


def build_optimizer(lr, weight_decay, warmup_steps, num_steps, clip_norm=1.0):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup_steps,
        decay_steps=max(num_steps, warmup_steps + 1), end_value=lr * 0.1)
    # Gradient clipping is REQUIRED: online-Wan buffers can be very long (tens of k
    # tokens) and a bad step otherwise explodes (observed in the Stage-1 smoke test).
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(schedule, weight_decay=weight_decay),
    )


class _EpisodeFeatureCache:
    """Bounded cache for immutable per-episode metadata and array handles."""

    def __init__(self, max_size, n_variants):
        self.max_size = max(0, int(max_size))
        self.n_variants = int(n_variants)
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def _load(self, epi_dir):
        with open(os.path.join(epi_dir, "variant_index.json")) as handle:
            variant_index = _json.load(handle)
        action_proprio = np.load(
            os.path.join(epi_dir, "action_proprio.npy"), allow_pickle=True
        ).item()
        latents = tuple(
            np.load(os.path.join(epi_dir, f"latents_v{v}.npy"))
            for v in range(self.n_variants)
        )
        causal_actions = []
        for v in range(self.n_variants):
            path = os.path.join(epi_dir, f"action_causal_v{v}.npy")
            causal_actions.append(
                np.load(path, allow_pickle=False) if os.path.isfile(path) else None
            )
        dino_path = os.path.join(epi_dir, "dinov3.npy")
        dino = np.load(dino_path, mmap_mode="r") if os.path.isfile(dino_path) else None
        lang_path = os.path.join(epi_dir, "lang_emb.npy")
        lang = np.load(lang_path) if os.path.isfile(lang_path) else None
        reward_path = os.path.join(epi_dir, "reward.npy")
        reward = (
            np.load(reward_path, allow_pickle=True).item()
            if os.path.isfile(reward_path)
            else None
        )
        return {
            "variant_index": variant_index,
            "action_proprio": action_proprio,
            "latents": latents,
            "causal_actions": tuple(causal_actions),
            "dino": dino,
            "lang": lang,
            "reward": reward,
        }

    def get(self, epi_dir):
        if self.max_size <= 0:
            return None
        with self.lock:
            entry = self.entries.pop(epi_dir, None)
            if entry is not None:
                self.entries[epi_dir] = entry
                self.hits += 1
                return entry
            self.misses += 1

        entry = self._load(epi_dir)
        with self.lock:
            existing = self.entries.pop(epi_dir, None)
            if existing is not None:
                entry = existing
            self.entries[epi_dir] = entry
            while len(self.entries) > self.max_size:
                self.entries.popitem(last=False)
        return entry

    def stats(self):
        with self.lock:
            return {
                "cache_hits": self.hits,
                "cache_misses": self.misses,
                "cache_entries": len(self.entries),
            }


def _prefetch(iterator, depth):
    """Prepare complete host batches in a single background producer."""
    depth = int(depth)
    if depth <= 0:
        return iterator

    sentinel = object()
    output = queue.Queue(maxsize=depth)

    def produce():
        try:
            for item in iterator:
                output.put(item)
        except BaseException as exc:
            output.put(exc)
        finally:
            output.put(sentinel)

    threading.Thread(target=produce, daemon=True, name="stage1-prefetch").start()

    def consume():
        while True:
            item = output.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    return consume()


def _load_stage1_checkpoint_into_modules(modules, ckpt_path, *, cosmo_bottleneck_memory):
    import flax.traverse_util as _trav

    from mme_vla_suite.shared.wan_alignment import require_recorded_alignment
    from mme_vla_suite.shared.wan_alignment import stage1_checkpoint_alignment

    s1 = np.load(ckpt_path, allow_pickle=True)
    expected_architecture = str(
        getattr(modules["bottleneck"], "architecture_name", "perceiver_v1")
    )
    if "metadata/memory_architecture" in s1.files:
        actual_architecture = str(np.asarray(s1["metadata/memory_architecture"]).item())
        if actual_architecture != expected_architecture:
            raise ValueError(
                "Stage-1 checkpoint architecture mismatch: "
                f"{actual_architecture!r} != {expected_architecture!r}"
            )
    elif expected_architecture != "perceiver_v1":
        raise ValueError(
            f"Stage-1 checkpoint {ckpt_path} has no architecture metadata; "
            f"refusing to load it into {expected_architecture!r}"
        )
    # The resumed weights must use the Wan chunk alignment of this run (no record: native).
    require_recorded_alignment(
        stage1_checkpoint_alignment(s1), cosmo_bottleneck_memory, where=ckpt_path
    )

    def _load_one(module, prefix):
        gd, st = nnx.split(module)
        pure = st.to_pure_dict() if hasattr(st, "to_pure_dict") else nnx.to_pure_dict(st)
        flat = _trav.flatten_dict(pure)
        nset = 0
        missing = []
        for key in s1.files:
            if not key.startswith(prefix + "/"):
                continue
            suffix = key[len(prefix) + 1:]
            tup = tuple(int(x) if x.isdigit() else x for x in suffix.split("/"))
            if tup in flat:
                arr = np.asarray(s1[key])
                flat[tup] = arr.astype(np.asarray(flat[tup]).dtype) if arr.dtype != np.asarray(flat[tup]).dtype else arr
                nset += 1
            else:
                missing.append(key)
        if missing:
            raise KeyError(f"Stage-1 resume missing {len(missing)} {prefix} params, first={missing[:5]}")
        if nset != len(flat):
            loaded = {
                tuple(int(x) if x.isdigit() else x for x in key[len(prefix) + 1:].split("/"))
                for key in s1.files
                if key.startswith(prefix + "/")
            }
            absent = [key for key in flat if key not in loaded]
            raise KeyError(
                f"Stage-1 resume filled {nset}/{len(flat)} {prefix} params; "
                f"checkpoint is missing targets {absent[:5]}"
            )
        st.replace_by_pure_dict(_trav.unflatten_dict(flat))
        return nnx.merge(gd, st), nset

    modules["bottleneck"], nb = _load_one(modules["bottleneck"], "bottleneck")
    modules["decoder"], nd = _load_one(modules["decoder"], "decoder")
    logger.info("resumed stage1 checkpoint %s (%d bottleneck, %d decoder params)", ckpt_path, nb, nd)
    return modules


def init_wandb(args, *, enabled):
    import wandb
    if not enabled:
        wandb.init(mode="disabled")
        return
    wandb.init(project=args.project_name, name=args.exp_name, config=vars(args))


def _resize_frames(fr, res):
    """(n, V, H, W, 3) uint8 -> (n, V, res, res, 3) via cv2 area-resize. No-op if H==res."""
    import cv2
    if fr.shape[2] == res and fr.shape[3] == res:
        return fr
    n, V = fr.shape[:2]
    out = np.empty((n, V, res, res, 3), np.uint8)
    for i in range(n):
        for v in range(V):
            out[i, v] = cv2.resize(fr[i, v], (res, res), interpolation=cv2.INTER_AREA)
    return out


def _batched_online_wan(wan, frames_list, boundary_list, wan_res=None, pool_hw=None):
    """BATCHED online Wan over a whole minibatch (one/few GPU calls, not B calls).

    wan_res: resize frames to this square resolution before encoding (e.g. 128 -> 64 tok/chunk
      vs 256 -> 256 tok/chunk). pool_hw: spatially pool each chunk grid to pool_hw^2 tokens.
    Both reduce the bottleneck KV. Groups samples by finalized length -> one VAE call per group.
    """
    tf = wan.temporal_factor
    out = [None] * len(frames_list)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, (fr, b) in enumerate(zip(frames_list, boundary_list)):
        if b < tf:
            out[i] = np.zeros((0, wan.latent_dim), np.float32)
        else:
            groups[b].append(i)
    for L, idxs in groups.items():
        clip = np.stack([frames_list[i][:L] for i in idxs], 0)   # (G, L, V, R, R, 3) native res
        # resize is done ON GPU inside the frontend (resize_to=wan_res) -> no CPU cv2 loop,
        # and the CPU->GPU transfer is of the raw frames uploaded once (interp is fused).
        lats = wan.finalized_encode_multiview_batched(clip, pool_hw=pool_hw, resize_to=wan_res)
        for k, i in enumerate(idxs):
            out[i] = np.asarray(lats[k], np.float32).reshape(-1, wan.latent_dim)
    return out


def make_dataloader(args, config):
    """Iterate the ds=1 cosmo_bottleneck store; ONLINE-Wan the jittered frames.

    Emits the batch dict composite_alignment_loss expects. cosmo_latent_emb is
    computed here (online Wan on the jittered frames returned by the RoboMMEDataset
    gather), padded+masked to the batch max; every other stream is stacked/padded.
    """
    if args.store is None:
        raise ValueError("--store <ds=1 cosmo_bottleneck store> is required.")
    if _NORM_TARGETS:
        _NORM_MEAN_STD.update(_load_norm(args.store))
    import json as _json
    from mme_vla_suite.training.dataset import RoboMMEDataset
    from mme_vla_suite.models.representation.wan_frontend import WanFrontend

    # gather-only dataset (no norm-stat compute); history_config drives jitter/aug.
    ds = RoboMMEDataset(
        dataset_path=args.store, data_config=None,
        history_config=config, action_horizon=1, compute_norm_stats=True,
    )
    samples = _json.load(open(os.path.join(args.store, "meta", "samples.json")))
    cbm = getattr(config, "cosmo_bottleneck_memory", None)
    # latent_source: "online" = encode Wan per-batch (slow, exact jitter); "offline" = read a
    # random precomputed variant's latents from the var8 store (fast, ~0.1s/iter). Both index the
    # per-frame streams (recent/action/proprio/reward) at the SAME frame_idx as the latents.
    LATENT_SOURCE = str(getattr(cbm, "latent_source", "online")) if cbm else "online"
    from mme_vla_suite.shared.wan_alignment import NATIVE as _WAN_NATIVE
    from mme_vla_suite.shared.wan_alignment import require_feature_store_alignment
    from mme_vla_suite.shared.wan_alignment import wan_chunk_alignment_from_config
    if LATENT_SOURCE == "offline":
        # Fail before training if any episode's latents use another Wan chunk alignment.
        require_feature_store_alignment(args.store, cbm)
    elif wan_chunk_alignment_from_config(cbm) != _WAN_NATIVE:
        raise NotImplementedError(
            "wan_chunk_alignment other than 'native' is implemented for latent_source: offline "
            "only; re-encode the store with mme_vla_suite.dataset_builder.reencode_wan_latents"
        )
    WAN_RES = int(getattr(cbm, "wan_input_res", 0)) or None
    POOL_HW = int(getattr(cbm, "pool_hw", 0)) or None
    POOL_H = int(getattr(cbm, "pool_h", 0)) or None
    POOL_W = int(getattr(cbm, "pool_w", 0)) or None
    WAN_H = int(getattr(cbm, "wan_h", 192)); WAN_W = int(getattr(cbm, "wan_w", 256))
    VISUAL_STREAM = str(getattr(cbm, "visual_feature_stream", "siglip")) if cbm else "siglip"
    # align_num_patches lives on the top-level cosmo_decoder (sibling of cosmo_bottleneck_memory).
    # OmegaConf.select returns None for missing keys (no raise). robocasa store=192, RoboDojo=64.
    from omegaconf import OmegaConf as _OC
    ALIGN_NP = _OC.select(config, "cosmo_decoder.align_num_patches", default=None)
    ALIGN_NP = int(ALIGN_NP) if ALIGN_NP is not None else 192
    ALIGN_DIM = _OC.select(config, "cosmo_decoder.align_dim", default=None)
    ALIGN_DIM = int(ALIGN_DIM) if ALIGN_DIM is not None else 2048
    _ad = _OC.select(config, "cosmo_decoder.action_dim", default=None)
    ACT_DIM = int(_ad) if _ad is not None else 12   # robocasa 12, RoboDojo 14
    _pd = _OC.select(config, "cosmo_decoder.proprio_dim", default=None)
    PRO_DIM = int(_pd) if _pd is not None else 16   # robocasa 16, RoboDojo 14
    wan = None
    if LATENT_SOURCE == "online":
        wan = WanFrontend(device="cuda",
                          compile_encode=bool(getattr(cbm, "wan_compile", False)) if cbm else False)

    rng = np.random.default_rng(args.seed)
    B = args.batch_size
    order = np.arange(len(samples))
    # BUCKETED padding -> a SMALL set of fixed latent-token sizes (jit compiles once PER tier)
    # instead of one huge cap. For pooled offline RoboDojo stores, full-episode latents are only
    # O(1k) tokens, so using the old unpooled 8192-token minimum wastes most of the attention.
    # Infer compact buckets from the store when possible; keep the legacy large buckets for
    # unpooled/offline-large and online paths.
    legacy_latent_buckets = [8192, 16384, 24576, 32768, 49152, 65536]
    if args.latent_buckets:
        LATENT_BUCKETS = [int(x) for x in str(args.latent_buckets).split(",") if str(x).strip()]
    elif LATENT_SOURCE == "offline":
        import glob as _glob
        max_latent_tokens = 0
        for lp in _glob.glob(os.path.join(args.store, "features", "episode_*", "latents_v0.npy")):
            arr = np.load(lp, mmap_mode="r")
            if arr.ndim == 3:
                max_latent_tokens = max(max_latent_tokens, int(arr.shape[0]) * int(arr.shape[1]))
        if 0 < max_latent_tokens <= 4096:
            LATENT_BUCKETS = []
            b = 256
            while b < max_latent_tokens:
                LATENT_BUCKETS.append(b)
                b *= 2
            LATENT_BUCKETS.append(b)
        else:
            LATENT_BUCKETS = legacy_latent_buckets
    else:
        LATENT_BUCKETS = legacy_latent_buckets
    logger.info("latent_buckets=%s", LATENT_BUCKETS)
    disable_recent_memory = bool(getattr(args, "disable_recent_memory", False))
    # Keep a stable non-empty tensor shape when the stream is disabled. Its mask
    # remains all false, so the placeholder cannot participate in attention.
    MAX_RECENT = (
        1
        if disable_recent_memory
        else (int(getattr(args, "max_recent_tokens", 0)) or 2048)
    )
    # action/proprio length = # buffer steps (<= episode frames). Fixed cap so the jitted
    # step shape is CONSTANT (varying it recompiles every iter -> the ~20s/iter bug).
    MAX_STEPS = int(getattr(args, "max_step_tokens", 0)) or 800
    temporal_factor = int(getattr(cbm, "temporal_factor", 1)) if cbm else 1
    pool_h = int(getattr(cbm, "pool_h", 0) or getattr(cbm, "pool_hw", 0) or 1)
    pool_w = int(getattr(cbm, "pool_w", 0) or getattr(cbm, "pool_hw", 0) or 1)
    tokens_per_chunk = (
        int(getattr(config, "num_views", 1)) * pool_h * pool_w
    )

    def _bucket(n):
        for b in LATENT_BUCKETS:
            if n <= b:
                return b
        return LATENT_BUCKETS[-1]   # over the top tier -> truncate to it

    def pad_stack(arrs, fixed_L=None, *, require_fit=False):
        """Pad variable-length (L, D) arrays to (B, Lpad, D) + bool mask (B, Lpad).
        fixed_L (or bucketed) -> constant-per-tier shape (bounds jit recompilation)."""
        Lmax = max(1, max(a.shape[0] for a in arrs))
        Lpad = fixed_L if fixed_L is not None else Lmax
        if require_fit and Lmax > Lpad:
            raise ValueError(
                f"exact length bucket overflow: observed={Lmax}, bucket={Lpad}"
            )
        D = arrs[0].shape[-1]
        out = np.zeros((len(arrs), Lpad, D), dtype=arrs[0].dtype)
        msk = np.zeros((len(arrs), Lpad), bool)
        for i, a in enumerate(arrs):
            a = a[-Lpad:] if a.shape[0] > Lpad else a   # rare over-cap: keep MOST RECENT tokens
            if a.shape[0]:
                out[i, :a.shape[0]] = a
                msk[i, :a.shape[0]] = True
        return out, msk

    num_query = int(getattr(cbm, "num_query_samples", 16)) if cbm else 16
    n_variants = int(getattr(cbm, "n_variants", 8)) if cbm else 8
    sample_frame_counts = None
    if args.length_bucketed_batches:
        variant_frame_cache = {}
        sample_frame_counts = np.empty((len(samples),), dtype=np.int32)
        for sample_index, sample in enumerate(samples):
            episode_index, step_index = sample_location(sample)
            frame_lists = variant_frame_cache.get(episode_index)
            if frame_lists is None:
                variant_index_path = os.path.join(
                    args.store,
                    "features",
                    f"episode_{episode_index}",
                    "variant_index.json",
                )
                with open(variant_index_path) as handle:
                    variant_index = _json.load(handle)
                frame_lists = tuple(
                    variant_index[f"v{variant}"]["frame_idx"]
                    for variant in range(n_variants)
                )
                variant_frame_cache[episode_index] = frame_lists
            sample_frame_counts[sample_index] = max(
                bisect.bisect_right(frame_indices, step_index)
                for frame_indices in frame_lists
            )
        max_frame_count = (
            int(sample_frame_counts.max()) if len(sample_frame_counts) else 0
        )
        logger.info(
            "exact length buckets use variant-frame counts: samples=%d "
            "episodes=%d max_frames=%d",
            len(samples),
            len(variant_frame_cache),
            max_frame_count,
        )
    episode_cache = _EpisodeFeatureCache(args.episode_cache_size, n_variants)
    gather_pool = (
        ThreadPoolExecutor(max_workers=args.gather_workers, thread_name_prefix="stage1-gather")
        if args.gather_workers > 1
        else None
    )
    dino_dtype = np.float16 if args.dino_transport_dtype == "float16" else np.float32

    def _gather_offline(epi, t, rng):
        """Read a random precomputed variant's latents + frame-index the per-frame streams at
        the SAME frame_idx (so latents and targets are consistent). Optional train-time 2x2 pool.
        """
        epi_dir = os.path.join(args.store, "features", f"episode_{epi}")
        cached = episode_cache.get(epi_dir)
        if cached is None:
            with open(os.path.join(epi_dir, "variant_index.json")) as handle:
                vi = _json.load(handle)
        else:
            vi = cached["variant_index"]
        # pick a variant whose latents were built from frames <= t (causal); fall back to canonical.
        cands = [v for v in range(n_variants) if vi[f"v{v}"]["frame_idx"] and vi[f"v{v}"]["frame_idx"][0] <= t]
        v = int(rng.choice(cands)) if cands else 0
        fidx = np.array(vi[f"v{v}"]["frame_idx"], np.int64)
        fidx = fidx[fidx <= t]                              # causal: only frames up to root step t
        n_jit = len(fidx)
        lat = (
            np.load(os.path.join(epi_dir, f"latents_v{v}.npy"))
            if cached is None
            else cached["latents"][v]
        )   # (nc_full, V*spatial, 48)
        # keep chunks whose frames are all <= t (finalized within [0,t]); drop terminal like online.
        # The recent window starts at kept frame 4*nc. With native Wan latents chunk c >= 1
        # holds kept frames 4c-3..4c, not 4c..4c+3 (see shared/wan_alignment.py).
        nc = max(0, n_jit // 4)
        latent_dim = lat.shape[-1]
        # FIX(bug1): allow ZERO chunks for early samples (n_jit<4 -> nc=0). Forcing >=1
        # chunk would load latents_v*[:1], which encodes FUTURE frames -> causal leak.
        kept = min(lat.shape[0], nc)
        lat = lat[:kept]
        if POOL_H and POOL_W and lat.shape[0] > 0:
            # RECTANGULAR grid pool (RoboDojo non-square 12x16). Block-mean (H,W) -> (POOL_H,POOL_W)
            # OUTPUT tokens/view. robocasa square path (POOL_HW) unchanged in the elif below.
            ncur = lat.shape[0]; gh = WAN_H // 16; gw = WAN_W // 16
            ph = int(POOL_H); pw = int(POOL_W)
            pooled_spatial = ph * pw
            if lat.shape[1] % pooled_spatial == 0 and lat.shape[1] < gh * gw:
                pass  # already pooled on disk: (chunks, views*pool_h*pool_w, dim)
            else:
                spatial = gh * gw; V = lat.shape[1] // spatial
                assert lat.shape[1] % spatial == 0, f"unexpected spatial layout {lat.shape} (grid {gh}x{gw})"
                assert gh % ph == 0 and gw % pw == 0, f"grid {gh}x{gw} must divide by pool ({ph},{pw})"
                bh = gh // ph; bw = gw // pw
                gg = lat.reshape(ncur, V, gh, gw, latent_dim)
                gg = gg.reshape(ncur, V, ph, bh, pw, bw, latent_dim).mean(axis=(3, 5))
                lat = gg.reshape(ncur, V * ph * pw, latent_dim)
        elif POOL_HW and lat.shape[0] > 0:
            # train-time spatial pool of the stored UNPOOLED grid: spatial=8x8=64 @128px ->
            # block-mean to POOL_HW x POOL_HW per view. lat: (nc, V*64, D).
            lat = _pool_square_latents(lat, pool_hw=POOL_HW)
        lat_flat = np.ascontiguousarray(lat).reshape(-1, latent_dim).astype(np.float32)
        boundary = (lat.shape[0]) * 4
        rec_idx = (
            np.empty((0,), dtype=np.int64)
            if disable_recent_memory
            else (fidx[boundary:] if boundary < n_jit else fidx[n_jit:])
        )
        ap = (
            np.load(os.path.join(epi_dir, "action_proprio.npy"), allow_pickle=True).item()
            if cached is None
            else cached["action_proprio"]
        )
        cached_causal = None if cached is None else cached["causal_actions"][v]
        if cached_causal is None:
            action = load_variant_causal_actions(epi_dir, v, fidx, ap["action"])
        else:
            if len(cached_causal) < len(fidx):
                raise ValueError(
                    f"cached causal action stream v{v} is shorter than {len(fidx)}"
                )
            action = np.ascontiguousarray(cached_causal[:len(fidx)]).astype(np.float32)
        proprio = np.ascontiguousarray(ap["proprio"][fidx]).astype(np.float32)
        act_enc = str(getattr(cbm, "action_encoding", "raw_delta")) if cbm else "raw_delta"
        pro_enc = str(getattr(cbm, "proprio_encoding", "raw_delta")) if cbm else "raw_delta"
        if act_enc == "raw":
            act_emb = action
        else:
            da = np.zeros_like(action); da[1:] = action[1:] - action[:-1]
            act_emb = np.concatenate([action, da], -1)
        if pro_enc == "raw":
            pro_emb = proprio
        else:
            dp = np.zeros_like(proprio); dp[1:] = proprio[1:] - proprio[:-1]
            pro_emb = np.concatenate([proprio, dp], -1)
        # queries: sample from the variant's frames
        K = min(num_query, n_jit)
        q_local = np.arange(n_jit) if K >= n_jit else np.sort(rng.choice(n_jit, K, replace=False))
        qfi = fidx[q_local]
        if VISUAL_STREAM == "dinov3":
            rec, align_target = load_visual_stream(
                epi_dir,
                stream="dinov3",
                recent_idx=rec_idx,
                align_idx=qfi,
                features=None if cached is None else cached["dino"],
                recent_dtype=dino_dtype,
                alignment_dtype=dino_dtype,
            )
        elif VISUAL_STREAM == "siglip":
            recent_mm = np.load(os.path.join(epi_dir, "recent_siglip.npy"), mmap_mode="r")
            siglip_dim = recent_mm.shape[-1]
            rec = (
                np.ascontiguousarray(recent_mm[rec_idx]).reshape(-1, siglip_dim).astype(np.float32)
                if len(rec_idx)
                else np.zeros((0, siglip_dim), np.float32)
            )
            align_mm = np.load(os.path.join(epi_dir, "align_prefix.npy"), mmap_mode="r")
            align_target = np.ascontiguousarray(align_mm[qfi]).astype(np.float32)
        else:
            raise ValueError(f"unsupported visual feature stream {VISUAL_STREAM!r}")
        # FIX(bug4): real language FiLM. Read the per-episode pooled PaliGemma text
        # embedding (precomputed into the store by augment_store_lang.py, from the
        # episode's GENERIC instruction -- no left/right leak). Same vector Stage-2 /
        # inference feed, so the bottleneck's language conditioning cannot drift.
        # Missing file -> zeros: lang_film has zero bias, so FiLM(0)=identity == the
        # language-dropout path (graceful degrade, no crash).
        lang_dim = int(getattr(cbm, "lang_emb_dim", 2048)) if cbm else 2048
        lang_p = os.path.join(epi_dir, "lang_emb.npy")
        cached_lang = None if cached is None else cached["lang"]
        lang = (
            np.asarray(cached_lang, np.float32).reshape(-1)
            if cached_lang is not None
            else (
                np.load(lang_p).astype(np.float32).reshape(-1)
                if os.path.exists(lang_p)
                else np.zeros((lang_dim,), np.float32)
            )
        )
        out = {"_latent": lat_flat, "cosmo_recent_emb": rec, "cosmo_action_emb": act_emb,
               "cosmo_proprio_emb": pro_emb, "cosmo_lang_emb": lang, "query_steps": q_local.astype(np.int64),
               "align_target": align_target,
               "action_target": (((np.ascontiguousarray(ap["action"][qfi]).astype(np.float32) - _NORM_MEAN_STD["actions"][0]) / (_NORM_MEAN_STD["actions"][1] + 1e-6)) if _NORM_TARGETS else np.ascontiguousarray(ap["action"][qfi]).astype(np.float32)),
               "proprio_target": (((np.ascontiguousarray(ap["proprio"][qfi]).astype(np.float32) - _NORM_MEAN_STD["state"][0]) / (_NORM_MEAN_STD["state"][1] + 1e-6)) if _NORM_TARGETS else np.ascontiguousarray(ap["proprio"][qfi]).astype(np.float32))}
        rp = os.path.join(epi_dir, "reward.npy")
        cached_reward = None if cached is None else cached["reward"]
        if cached_reward is not None or os.path.exists(rp):
            rd = (
                cached_reward
                if cached_reward is not None
                else np.load(rp, allow_pickle=True).item()
            )
            rm = np.stack([rd[m] for m in ("incremental","forward","backward")],0).mean(0)
            out["reward_target"] = np.float32(np.clip(rm[int(fidx[-1])], 0, 1))
        return out

    while True:
        if args.length_bucketed_batches:
            epoch_batches = build_length_bucketed_epoch_batches(
                samples,
                batch_size=B,
                buckets=LATENT_BUCKETS,
                temporal_factor=temporal_factor,
                tokens_per_chunk=tokens_per_chunk,
                rng=rng,
                sample_frame_counts=sample_frame_counts,
            )
        else:
            rng.shuffle(order)
            epoch_batches = [
                (None, order[b0 : b0 + B])
                for b0 in range(0, len(order) - B + 1, B)
            ]
        for expected_latent_bucket, idxs in epoch_batches:
            gather_start = time.perf_counter()
            locations = [sample_location(samples[si]) for si in idxs]
            if LATENT_SOURCE == "offline":
                if gather_pool is None:
                    gathered = [_gather_offline(epi, step, rng) for epi, step in locations]
                else:
                    seeds = rng.integers(0, np.iinfo(np.int64).max, size=len(locations))
                    gathered = list(gather_pool.map(
                        lambda item: _gather_offline(
                            item[0][0], item[0][1], np.random.default_rng(int(item[1]))
                        ),
                        zip(locations, seeds),
                    ))
            else:
                gathered = [
                    ds._gather_cosmo_bottleneck(epi, step)
                    for epi, step in locations
                ]
                frames_list = [g.pop("cosmo_frames") for g in gathered]
                bnd_list = [int(g.pop("cosmo_frames_boundary")) for g in gathered]
                lats = _batched_online_wan(wan, frames_list, bnd_list, wan_res=WAN_RES, pool_hw=POOL_HW)
                for g, lt in zip(gathered, lats):
                    g["_latent"] = lt
            gather_seconds = time.perf_counter() - gather_start
            collate_start = time.perf_counter()
            # BUCKETED latent pad (compile once per tier; attention cost tracks actual size).
            lat_arrs = [g["_latent"] for g in gathered]
            lat_bucket = (
                int(expected_latent_bucket)
                if expected_latent_bucket is not None
                else _bucket(max(1, max(a.shape[0] for a in lat_arrs)))
            )
            batch_max_steps = (
                max_steps_for_latent_bucket(
                    lat_bucket,
                    temporal_factor=temporal_factor,
                    tokens_per_chunk=tokens_per_chunk,
                    global_cap=MAX_STEPS,
                )
                if args.length_bucketed_batches
                else MAX_STEPS
            )
            lat, lat_mask = pad_stack(
                lat_arrs,
                fixed_L=lat_bucket,
                require_fit=args.length_bucketed_batches,
            )
            # FIX(bug2): KEEP the masks for recent/action/proprio -- padded slots are NOT
            # valid memory; they must be masked out in the bottleneck's cross-attention,
            # else zero-token pads drown real memory (real recent ~[192..576], not 2048).
            rec, rec_mask = pad_stack([g["cosmo_recent_emb"] for g in gathered], fixed_L=MAX_RECENT)
            # action/proprio length = # buffer steps, VARIES per batch -> pad to constant
            # MAX_STEPS (avoids the recompile bug) but return the real mask.
            act, act_mask = pad_stack(
                [g["cosmo_action_emb"] for g in gathered],
                fixed_L=batch_max_steps,
                require_fit=args.length_bucketed_batches,
            )
            pro, pro_mask = pad_stack(
                [g["cosmo_proprio_emb"] for g in gathered],
                fixed_L=batch_max_steps,
                require_fit=args.length_bucketed_batches,
            )
            # queries + targets: bucket Q too (constant tiers) so it doesn't recompile.
            Qmax = max(g["query_steps"].shape[0] for g in gathered)
            Qmax = min(32, ((Qmax + 7) // 8) * 8)   # round to /8 tier, cap 32 (num_query cap)
            def padq(key, shape_tail, dtype=np.float32):
                out = np.zeros((B, Qmax, *shape_tail), dtype=dtype)
                for i, g in enumerate(gathered):
                    q = g[key][:Qmax]; out[i, :q.shape[0]] = q   # truncate if over bucketed Qmax
                return out
            qsteps = np.zeros((B, Qmax), np.int64)
            query_mask = np.zeros((B, Qmax), bool)   # FIX(bug3): mark real vs padded queries
            for i, g in enumerate(gathered):
                q = g["query_steps"][:Qmax]; qsteps[i, :q.shape[0]] = q; query_mask[i, :q.shape[0]] = True
            # FIX(bug4): stack the per-episode language embedding -> (B, lang_emb_dim).
            # Both the offline gather and dataset._gather_cosmo_bottleneck now return a real
            # vector (zeros fallback == FiLM-identity), so the FiLM path trains in Stage-1.
            lang_dim = int(getattr(cbm, "lang_emb_dim", 2048)) if cbm else 2048
            lang_emb_batch = np.stack([
                (np.asarray(g["cosmo_lang_emb"], np.float32).reshape(-1)
                 if g.get("cosmo_lang_emb") is not None else np.zeros((lang_dim,), np.float32))
                for g in gathered], 0)
            batch = {
                "cosmo_latent_emb": lat,
                "cosmo_mem_mask": lat_mask,
                "cosmo_recent_emb": rec,
                "cosmo_recent_mask": rec_mask,        # FIX(bug2)
                "cosmo_action_emb": act,
                "cosmo_action_mask": act_mask,        # FIX(bug2)
                "cosmo_proprio_emb": pro,
                "cosmo_proprio_mask": pro_mask,       # FIX(bug2)
                "cosmo_lang_emb": lang_emb_batch,     # FIX(bug4): real language (see below)
                "query_steps": qsteps,
                "query_mask": query_mask,             # FIX(bug3)
                "align_target": padq(
                    "align_target", (ALIGN_NP, ALIGN_DIM), dtype=dino_dtype
                ),
                "action_target": padq("action_target", (ACT_DIM,)),
                "proprio_target": padq("proprio_target", (PRO_DIM,)),
            }
            if "reward_target" in gathered[0]:
                # single current-step scalar per sample -> (B,)
                batch["reward_target"] = np.array(
                    [float(g["reward_target"]) for g in gathered], np.float32)
            batch["_host_profile"] = {
                "gather_s": gather_seconds,
                "collate_s": time.perf_counter() - collate_start,
                "latent_bucket": lat_bucket,
                "max_step_tokens": batch_max_steps,
                **episode_cache.stats(),
            }
            yield batch


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    config = OmegaConf.load(args.config)
    from mme_vla_suite.shared.wan_alignment import wan_chunk_alignment_from_config

    wan_chunk_alignment = wan_chunk_alignment_from_config(
        getattr(config, "cosmo_bottleneck_memory", None)
    )
    _configure_visual_stream(config, args.store)
    _configure_compact_alignment(config, args.store)

    # CRITICAL config coupling: decoder must use the SAME memory_token_dim the
    # bottleneck produces (1024 in the wan/modulation variant). Both read a
    # single shared `config.memory_token_dim`.
    logger.info("memory_token_dim=%s", config.memory_token_dim)

    # FSDP mesh (mirror scripts/train.py): shard data + params across GPUs.
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    compute_dtype = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }[args.module_compute_dtype]
    logger.info(
        "Stage-1 module compute dtype=%s with float32 parameters/state",
        args.module_compute_dtype,
    )
    modules = make_stage1_modules(
        config,
        seed=args.seed,
        dtype=compute_dtype,
    )
    if args.resume_stage1_ckpt:
        modules = _load_stage1_checkpoint_into_modules(
            modules,
            args.resume_stage1_ckpt,
            cosmo_bottleneck_memory=getattr(config, "cosmo_bottleneck_memory", None),
        )
    graphdef, state = nnx.split((modules["bottleneck"], modules["decoder"]))

    lr_decay_steps = args.lr_decay_steps or args.num_steps
    tx = build_optimizer(args.lr, args.weight_decay, args.warmup_steps, lr_decay_steps)
    opt_state = tx.init(state)
    step = int(args.start_step)
    if args.resume_full_state:
        state, opt_state, step = _restore_training_state(
            args.resume_full_state, state, opt_state, replicated_sharding
        )
        logger.info("resumed full Stage-1 state from %s at step=%d", args.resume_full_state, step)

    weights = {"align": args.w_align, "action": args.w_action,
               "proprio": args.w_proprio, "reward": args.w_reward}

    def loss_fn(state, batch):
        bmk, dec = nnx.merge(graphdef, state)
        total, parts = composite_alignment_loss({"bottleneck": bmk, "decoder": dec}, batch, weights)
        return total, parts

    @jax.jit
    def train_step(state, opt_state, batch):
        (total, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(state, batch)
        updates, opt_state = tx.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, total, parts

    init_wandb(args, enabled=args.wandb_enabled)
    loader = iter(_prefetch(make_dataloader(args, config), args.prefetch_batches))

    def emit_profile(payload):
        payload = {"step": step, **payload}
        logger.info("PROFILE %s", payload)
        if args.profile_jsonl:
            os.makedirs(os.path.dirname(os.path.abspath(args.profile_jsonl)), exist_ok=True)
            with open(args.profile_jsonl, "a") as handle:
                handle.write(_json.dumps(payload, sort_keys=True) + "\n")

    last_saved_step = None
    blocking_profiled = 0
    throughput_window_start = None
    throughput_window_step = None
    with sharding.set_mesh(mesh):
        while True:
            if step >= args.num_steps:
                break
            loader_wait_start = time.perf_counter()
            batch = next(loader)
            loader_wait_s = time.perf_counter() - loader_wait_start
            host_profile = batch.pop("_host_profile", {})
            host_nbytes = sum(
                value.nbytes for value in batch.values() if isinstance(value, np.ndarray)
            )

            blocking_profile = (
                args.profile_blocking_steps > 0
                and step >= args.profile_warmup_steps
                and blocking_profiled < args.profile_blocking_steps
            )
            transfer_start = time.perf_counter()
            batch = jax.device_put(batch, data_sharding)
            transfer_dispatch_s = time.perf_counter() - transfer_start
            transfer_block_s = None
            if blocking_profile:
                jax.block_until_ready(batch)
                transfer_block_s = time.perf_counter() - transfer_start

            train_start = time.perf_counter()
            state, opt_state, total, parts = train_step(state, opt_state, batch)
            train_dispatch_s = time.perf_counter() - train_start
            train_block_s = None
            if blocking_profile:
                jax.block_until_ready(total)
                train_block_s = time.perf_counter() - train_start
                blocking_profiled += 1
            step += 1

            if blocking_profile:
                emit_profile({
                    **host_profile,
                    "loader_wait_s": loader_wait_s,
                    "host_batch_gib": host_nbytes / (1024 ** 3),
                    "transfer_dispatch_s": transfer_dispatch_s,
                    "transfer_block_s": transfer_block_s,
                    "train_dispatch_s": train_dispatch_s,
                    "train_block_s": train_block_s,
                    "mode": "blocking_components",
                })

            if args.profile_every > 0 and step >= args.profile_warmup_steps:
                if throughput_window_start is None:
                    jax.block_until_ready(total)
                    throughput_window_start = time.perf_counter()
                    throughput_window_step = step
                elif step - throughput_window_step >= args.profile_every:
                    jax.block_until_ready(total)
                    now = time.perf_counter()
                    window_steps = step - throughput_window_step
                    emit_profile({
                        "mode": "overlapped_throughput",
                        "window_steps": window_steps,
                        "window_s": now - throughput_window_start,
                        "seconds_per_step": (now - throughput_window_start) / window_steps,
                    })
                    throughput_window_start = now
                    throughput_window_step = step

            if step % args.log_every == 0:
                import wandb
                total_value = float(total)
                part_values = {k: float(v) for k, v in parts.items()}
                if not np.isfinite(total_value) or not all(
                    np.isfinite(value) for value in part_values.values()
                ):
                    raise FloatingPointError(
                        f"non-finite loss at step={step}: "
                        f"total={total_value}, parts={part_values}"
                    )
                wandb.log(
                    {"loss": total_value, **{f"loss/{k}": v for k, v in part_values.items()}},
                    step=step,
                )
                logger.info("step=%d loss=%.4f", step, total_value)
                logger.info(
                    "PARTS_DIAG %s",
                    {k: round(value, 5) for k, value in part_values.items()},
                )
            if step % args.ckpt_every == 0:
                bmk, dec = nnx.merge(graphdef, state)
                _save_checkpoint(
                    args.checkpoint_dir, step, {"bottleneck": bmk, "decoder": dec},
                    wan_chunk_alignment=wan_chunk_alignment,
                )
                _save_training_state(args.checkpoint_dir, step, state, opt_state)
                last_saved_step = step
                if (
                    args.order_probe_labels
                    and args.order_probe_every > 0
                    and step % args.order_probe_every == 0
                ):
                    from mme_vla_suite.scripts.order_probe import run_order_probe
                    import wandb

                    probe_dir = args.order_probe_out_dir or os.path.join(args.checkpoint_dir, "order_probe")
                    probe_metrics, probe_path = run_order_probe(
                        bmk,
                        config,
                        args.store,
                        args.order_probe_labels,
                        step,
                        probe_dir,
                        folds=args.order_probe_folds,
                        steps_per_episode=args.order_probe_steps_per_episode,
                        seed=args.seed,
                    )
                    wandb.log(probe_metrics, step=step)
                    logger.info("ORDER_PROBE step=%d %s -> %s", step, probe_metrics, probe_path)
    if step > 0 and step != last_saved_step and not args.skip_final_checkpoint:
        bmk, dec = nnx.merge(graphdef, state)
        _save_checkpoint(
            args.checkpoint_dir, step, {"bottleneck": bmk, "decoder": dec},
            wan_chunk_alignment=wan_chunk_alignment,
        )
        _save_training_state(args.checkpoint_dir, step, state, opt_state)


def _module_flat_params(module, prefix):
    """Flatten an nnx module's params to {prefix/a/b/c: array} using flatten_dict (sep='/').

    This MUST match the key format Stage1BottleneckWeightLoader expects: it does
    flatten_dict(stage2_params, sep='/') and looks up 'mem_encoder/<suffix>', remapping
    from 'bottleneck/<suffix>'. So we emit 'bottleneck/<suffix>' with the identical suffix
    (int layer indices stringified). Verified: the 63 bottleneck suffixes == the 63
    mem_encoder suffixes in the Stage-2 tree.
    """
    import flax.traverse_util as _trav
    st = nnx.state(module)
    pure = st.to_pure_dict() if hasattr(st, "to_pure_dict") else nnx.to_pure_dict(st)
    flat = _trav.flatten_dict(pure)  # tuple keys, may contain int layer indices
    return {prefix + "/" + "/".join(str(x) for x in k): np.asarray(v) for k, v in flat.items()}


def _save_checkpoint(checkpoint_dir, step, modules, *, wan_chunk_alignment):
    """Checkpoint bottleneck+decoder params to a .npz whose keys the Stage-2 weight loader
    consumes directly: 'bottleneck/<...>' (-> remapped to mem_encoder/) and 'decoder/<...>'
    (loaded into aux_decoder when Stage-2 enables auxiliary loss). NOT the jax-keystr format
    (that broke the loader's startswith).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"stage1_step{step}.npz")
    flat = {}
    flat.update(_module_flat_params(modules["bottleneck"], "bottleneck"))
    flat.update(_module_flat_params(modules["decoder"], "decoder"))
    flat["metadata/memory_architecture"] = np.asarray(
        str(getattr(modules["bottleneck"], "architecture_name", "perceiver_v1"))
    )
    # Stage-2 training and precompute_mem compare this record with their config.
    from mme_vla_suite.shared.wan_alignment import STAGE1_METADATA_KEY

    flat[STAGE1_METADATA_KEY] = np.asarray(wan_chunk_alignment)
    np.savez(path, **flat)
    logger.info("saved checkpoint -> %s (%d bottleneck + decoder params)", path, len(flat))


def _save_training_state(checkpoint_dir, step, state, opt_state):
    """Save resumable model and optimizer state after ``step`` completed updates."""
    path = os.path.abspath(os.path.join(checkpoint_dir, "full_state", str(step)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    item = {
        "state": state,
        "opt_state": opt_state,
        "step": jnp.asarray(step, dtype=jnp.int32),
    }
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(path, item)
    logger.info("saved resumable training state -> %s", path)
    return path


def _restore_training_state(path, state, opt_state, target_sharding):
    """Restore model, optimizer, and completed-update count into fresh target trees."""
    target = {
        "state": state,
        "opt_state": opt_state,
        "step": jnp.asarray(0, dtype=jnp.int32),
    }
    sharding_tree = jax.tree.map(
        lambda value: target_sharding if isinstance(value, jax.Array) else None,
        target,
    )
    restore_args = checkpoint_utils.construct_restore_args(
        target, sharding_tree
    )
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            os.path.abspath(path),
            args=ocp.args.PyTreeRestore(
                item=target,
                restore_args=restore_args,
            ),
        )
    return restored["state"], restored["opt_state"], int(np.asarray(restored["step"]))


if __name__ == "__main__":
    main()
