from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import functools
import json
import logging
import os
from pathlib import Path
import pickle
import random
import re

import numpy as np
from omegaconf import DictConfig

from mme_vla_suite.shared.mem_buffer import MemoryBuffer
from mme_vla_suite.shared.mem_buffer import MemoryBufferRecurrent
from mme_vla_suite.shared.memory_caps import max_recent_tokens
from mme_vla_suite.shared.visual_streams import load_dino_manifest
from mme_vla_suite.shared.visual_streams import visual_feature_filename
from mme_vla_suite.shared.wan_alignment import NATIVE
from mme_vla_suite.shared.wan_alignment import read_alignment_record
from mme_vla_suite.shared.wan_alignment import require_recorded_alignment
from mme_vla_suite.shared.wan_alignment import require_store_alignment
from mme_vla_suite.shared.wan_alignment import wan_chunk_alignment_from_config
from mme_vla_suite.training.episode_cache import EpisodeStoreCache
from mme_vla_suite.training.key_event_weights import KeyEventWeightLookup

random.seed(0)
logger = logging.getLogger(__name__)


def _next_seeded_generator() -> np.random.Generator:
    """Create a per-sample generator from the DataLoader-seeded NumPy stream."""
    seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    return np.random.default_rng(seed)


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load NumPy 2 pickles in the NumPy 1 Stage-2 runtime."""

    def find_class(self, module, name):
        if module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        return super().find_class(module, name)


def load_vector_file(vector_path: str, step_idx: int) -> tuple[dict, int]:
    with open(vector_path, "rb") as f:
        return np.load(f, allow_pickle=True).item(), step_idx


def late_episode_loss_weight(
    step_idx: int,
    episode_len: int,
    *,
    late_fraction: float = 0.5,
    late_weight: float = 1.0,
) -> float:
    """Return a per-sample objective weight based on episode-relative time."""
    late_weight = float(late_weight)
    late_fraction = float(late_fraction)
    if late_weight <= 1.0:
        return 1.0
    if not 0.0 <= late_fraction <= 1.0:
        raise ValueError(f"late_fraction must be in [0, 1], got {late_fraction}")
    episode_len = max(1, int(episode_len))
    boundary = int(np.floor(episode_len * late_fraction))
    return late_weight if int(step_idx) >= boundary else 1.0


def _visual_feature_stream(cfg) -> str:
    stream = str(getattr(cfg, "visual_feature_stream", "siglip")) if cfg is not None else "siglip"
    visual_feature_filename(stream)
    return stream


@functools.cache
def _precompute_alignment(memory_dir: str) -> str:
    return read_alignment_record(memory_dir)


def resolve_feature_dir(dataset_path: str | Path, history_config) -> Path:
    """Use an optional published feature store without changing policy samples."""
    cfg = getattr(history_config, "cosmo_bottleneck_memory", None)
    feature_store_path = str(getattr(cfg, "feature_store_path", "")).strip()
    root = Path(feature_store_path) if feature_store_path else Path(dataset_path)
    return root / "features"


def pool_square_offline_latents(lat: np.ndarray, pool_hw: int) -> np.ndarray:
    """Pool unpooled 8x8 latents, while preserving stores already pooled on disk."""
    if lat.shape[0] == 0:
        return lat
    side = 8
    pooled_side = int(pool_hw)
    pooled_spatial = pooled_side * pooled_side
    if lat.shape[1] % pooled_spatial == 0 and lat.shape[1] < side * side:
        return lat
    if lat.shape[1] % (side * side) != 0:
        raise AssertionError(f"unexpected spatial layout {lat.shape}")
    if side % pooled_side != 0:
        raise AssertionError(f"pool_hw {pooled_side} must divide {side}")
    chunks, _, latent_dim = lat.shape
    views = lat.shape[1] // (side * side)
    block = side // pooled_side
    grid = lat.reshape(chunks, views, side, side, latent_dim)
    grid = grid.reshape(
        chunks,
        views,
        pooled_side,
        block,
        pooled_side,
        block,
        latent_dim,
    ).mean(axis=(3, 5))
    return grid.reshape(chunks, views * pooled_spatial, latent_dim)


class SampleDataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.stats = json.load(open(os.path.join(self.dataset_path, "meta", "stats.json")))

    def __len__(self):
        if "execution_samples" in self.stats:
            return self.stats["execution_samples"]
        else:
            return self.stats["total_samples"]

    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_path,  "data", f"{idx}.pkl"), "rb") as f:
            return _NumpyCompatUnpickler(f).load()


class RoboMMEDataset:
    def __init__(
        self,
        dataset_path: str,
        data_config: object | None,
        history_config: DictConfig | None,
        action_horizon: int,
        compute_norm_stats: bool = False,
        late_episode_loss_weight_value: float = 1.0,
        late_episode_loss_fraction: float = 0.5,
        key_event_manifest_path: str | None = None,
        key_event_loss_weight: float = 1.0,
        key_event_window_radius: int = 5,
    ):
        self.history_config = history_config

        self.action_horizon = action_horizon
        self.dataset = SampleDataset(dataset_path)
        self.feature_dir = resolve_feature_dir(self.dataset.dataset_path, history_config)
        self._framesamp_fallback_cache = {}
        self._framesamp_pos4 = None
        self.need_aux = False  # default; set True below iff history_config requests aux
        self.late_episode_loss_weight_value = float(late_episode_loss_weight_value)
        self.late_episode_loss_fraction = float(late_episode_loss_fraction)
        self._episode_len_cache = {}
        self._episode_store_cache = None
        self.key_event_weights = None
        if key_event_manifest_path:
            if self.late_episode_loss_weight_value > 1.0:
                raise ValueError(
                    "key-event and late-episode weighting are mutually exclusive"
                )
            if float(key_event_loss_weight) <= 1.0:
                raise ValueError("key_event_loss_weight must be greater than 1")
            episode_count = sum(
                1 for path in self.feature_dir.glob("episode_*") if path.is_dir()
            )
            self.key_event_weights = KeyEventWeightLookup.load(
                key_event_manifest_path,
                expected_store_path=str(Path(dataset_path)),
                expected_episode_count=episode_count,
                expected_radius=int(key_event_window_radius),
                expected_weight=float(key_event_loss_weight),
            )

        if self.history_config is not None:
            self.img_emb_dim = self.history_config.memory_feature.img.input_dim
            self.pos_emb_dim = self.history_config.memory_feature.pos.input_dim
            self.state_emb_dim = self.history_config.memory_feature.state.input_dim
            self.num_views = self.history_config.num_views
            self.streaming_obs_horizon = self.history_config.streaming_obs_horizon

            self.use_time = bool(getattr(self.history_config, "memory_use_time", False))
            # E2E+aux: load reconstruction targets ONLY when the run requests aux (aux_weight>0).
            # Action-only / all other runs skip these (esp. the large align_prefix (Q,64,2048)).
            self.need_aux = float(getattr(self.history_config, "aux_weight", 0.0) or 0.0) > 0.0
            if self.history_config.representation_type == "perceptual":
                self.mem_buffer = MemoryBuffer(
                    num_views=self.num_views,
                    img_emb_dim=self.img_emb_dim,
                    pos_emb_dim=self.pos_emb_dim,
                    state_emb_dim=self.state_emb_dim,
                    token_drop_stride=self.streaming_obs_horizon // 2,
                    use_time=self.use_time,
                )
            elif self.history_config.representation_type == "recurrent":
                self.mem_buffer = MemoryBufferRecurrent(
                    num_views=self.num_views,
                    img_emb_dim=self.img_emb_dim,
                    pos_emb_dim=self.pos_emb_dim,
                    state_emb_dim=self.state_emb_dim,
                    input_obs_horizon=self.streaming_obs_horizon,
                    max_recur_steps=self.history_config.recurrent_memory.max_recur_steps,
                    max_video_steps=self.history_config.recurrent_memory.max_pretraj_steps,
                )
            else:
                # symbolic memory does not need mem_buffer
                self.mem_buffer = None
        else:
            logger.info("=== Do not use history ===")

        self.compute_norm_stats = compute_norm_stats

        if not compute_norm_stats:
            if data_config is None:
                raise ValueError("data_config is required when compute_norm_stats is false")
            self.state_norm_stats = data_config.norm_stats['state']
            self.use_quantiles = data_config.use_quantile_norm

        # --- memory-stream input augmentation (train-only, cosmo_bottleneck) ---
        # Regularizes the bottleneck on the small toy set so it learns a smooth
        # "summarize what was seen" map rather than memorizing exact feature values.
        # Applied ONLY to the encoder INPUT streams (recent/action/proprio); the
        # causal latent backbone and ALL targets stay clean. Disabled during
        # norm-stat computation and whenever the invariant test needs byte-exact
        # offline==online (set aug_enabled=False there).
        aug = getattr(self.history_config, "stream_aug", None) if self.history_config is not None else None
        self.aug_enabled = bool(getattr(aug, "enabled", False)) and not compute_norm_stats
        # (1) additive Gaussian, scale-relative to each stream's own std
        self.aug_gauss_sigma = float(getattr(aug, "gauss_sigma", 0.05)) if aug is not None else 0.05
        # (3) stream dropout: prob of zeroing a WHOLE modality this sample
        self.aug_stream_drop_p = float(getattr(aug, "stream_drop_p", 0.1)) if aug is not None else 0.1
        # (2) OPTIONAL feature dropout: fraction of feature channels zeroed (off by default)
        self.aug_feat_drop_p = float(getattr(aug, "feat_drop_p", 0.0)) if aug is not None else 0.0

        # --- frame-index jitter (#1 stride/interval + #3 start), train-only ---
        # Selects WHICH stored (ds=1) frames form the buffer at each sample, mimicking
        # inference framing variance. The selected frames (strict time order) are the
        # video the ONLINE Wan tokenizer compresses; recent/action/proprio/targets are
        # sliced at the SAME indices. canonical=True -> deterministic inference framing.
        fj = getattr(self.history_config, "frame_jitter", None) if self.history_config is not None else None
        self.fj_enabled = bool(getattr(fj, "enabled", False)) and not compute_norm_stats
        self.fj_start_jitter = int(getattr(fj, "start_jitter", 3)) if fj is not None else 3
        self.fj_stride_choices = tuple(getattr(fj, "stride_choices", (1, 2, 3))) if fj is not None else (1, 2, 3)
        self.fj_interval_jitter = int(getattr(fj, "interval_jitter", 1)) if fj is not None else 1
        self.fj_canonical_downsample = int(getattr(fj, "canonical_downsample", 2)) if fj is not None else 2

    def _sample_frame_indices(self, t, rng, canonical=False):
        """Jittered (or canonical) frame indices covering [s, t] in the ds=1 store.

        #3 start jitter: s ~ U[0, start_jitter].  #1 stride+interval jitter: base
        stride S ~ choice(stride_choices); each gap = S + U[-ij, +ij] (>=1) -> IRREGULAR
        intervals. Current step t is ALWAYS the last index. canonical -> s=0, fixed
        stride=canonical_downsample, no jitter (deterministic inference framing).
        """
        if canonical or not self.fj_enabled:
            step = max(1, self.fj_canonical_downsample)
            idx = list(range(0, t + 1, step))
            if idx[-1] != t:
                idx.append(t)
            return np.array(idx, dtype=np.int64)
        s = int(rng.integers(0, self.fj_start_jitter + 1)) if self.fj_start_jitter > 0 else 0
        s = min(s, t)
        S = int(rng.choice(self.fj_stride_choices))
        ij = self.fj_interval_jitter
        idx = [s]; cur = s
        while cur < t:
            gap = S + (int(rng.integers(-ij, ij + 1)) if ij > 0 else 0)
            gap = max(1, gap)
            cur += gap
            if cur >= t:
                break
            idx.append(cur)
        idx.append(t)
        return np.array(sorted(set(idx)), dtype=np.int64)

    def _augment_stream(self, arr, name):
        """Train-only input noise for one memory stream. name in {recent,action,proprio}.
        (1) additive Gaussian (scale-relative), (3) stream dropout; (2) feature dropout optional.
        Never applied to targets or to the causal latent backbone."""
        if not self.aug_enabled or arr.size == 0:
            return arr
        rng = np.random  # dataset already seeded; per-sample stochasticity is desired
        # (3) stream dropout: zero the whole modality occasionally
        if self.aug_stream_drop_p > 0.0 and rng.random() < self.aug_stream_drop_p:
            return np.zeros_like(arr)
        out = arr
        # (1) additive Gaussian, per-stream std-relative
        if self.aug_gauss_sigma > 0.0:
            scale = float(np.std(arr)) + 1e-6
            out = out + (self.aug_gauss_sigma * scale) * rng.standard_normal(arr.shape).astype(arr.dtype)
        # (2) optional feature (channel) dropout with inverted scaling
        if self.aug_feat_drop_p > 0.0:
            keep = (rng.random(arr.shape[-1]) >= self.aug_feat_drop_p).astype(arr.dtype)
            out = out * keep / max(1e-6, 1.0 - self.aug_feat_drop_p)
        return out.astype(np.float32, copy=False)

    def _get_episode_store_cache(self, cfg):
        cache = getattr(self, "_episode_store_cache", None)
        if cache is None:
            capacity = int(getattr(cfg, "episode_cache_size", 8)) if cfg is not None else 8
            stream = _visual_feature_stream(cfg)
            if stream == "dinov3":
                load_dino_manifest(self.feature_dir.parent)
            cache = EpisodeStoreCache(
                capacity=capacity, visual_feature_stream=stream
            )
            self._episode_store_cache = cache
        return cache

    def _gather_history_feat(self, indices_to_load: list[int], epis_idx: int):
        """Load legacy token files or derive frame-sampling tokens from a compact store."""
        epi_dir = os.path.join(self.feature_dir, f"episode_{epis_idx}")
        first_token = os.path.join(epi_dir, f"token_emb_{indices_to_load[0]}.npy")
        if not os.path.exists(first_token):
            cached = self._framesamp_fallback_cache.get(epis_idx)
            if cached is None:
                recent = np.load(
                    os.path.join(epi_dir, "recent_siglip.npy"), mmap_mode="r"
                )
                proprio = np.load(
                    os.path.join(epi_dir, "action_proprio.npy"), allow_pickle=True
                ).item()["proprio"]
                cached = (recent, proprio)
                self._framesamp_fallback_cache[epis_idx] = cached
                if len(self._framesamp_fallback_cache) > 4:
                    self._framesamp_fallback_cache.pop(
                        next(iter(self._framesamp_fallback_cache))
                    )
            recent, proprio = cached
            max_step = max(indices_to_load)
            required_pos = (max_step + 1) * self.num_views
            if self._framesamp_pos4 is None or self._framesamp_pos4.shape[0] < required_pos:
                import jax.numpy as jnp

                from mme_vla_suite.shared.posemb_3d import PosEmb3D

                self._framesamp_pos4 = np.asarray(
                    PosEmb3D(dim=self.pos_emb_dim)(
                        jnp.arange(max(2048, required_pos)), spatial_size=4
                    )
                )

            history_feats = {}
            for idx in indices_to_load:
                # recent_siglip is (T, V, 64, D); match the legacy 4x4 tokens.
                siglip = np.asarray(recent[idx], dtype=np.float32)
                views, _, dim = siglip.shape
                pooled = siglip.reshape(views, 8, 8, dim).reshape(
                    views, 4, 2, 4, 2, dim
                ).mean(axis=(2, 4))
                start = idx * self.num_views
                history_feats[idx] = {
                    "image_emb_4x4": pooled.reshape(views, 16, dim),
                    "pos_emb_4x4": self._framesamp_pos4[
                        start : start + self.num_views
                    ],
                    "state_emb": np.asarray(proprio[idx], dtype=np.float32),
                }
            return history_feats

        history_feats = {}
        history_paths = []
        for idx in indices_to_load:
            history_feats[idx] = {}
            history_paths.append(
                os.path.join(epi_dir, f"token_emb_{idx}.npy")
            )

        if self.history_config.representation_type == "recurrent":
            # load_vector_fn = load_vector_file_lru4096 if self.use_lru else load_vector_file
            load_vector_fn = load_vector_file
            max_workers = min(48, max(4, len(history_paths)))
        else:
            # load_vector_fn = load_vector_file_lru1024 if self.use_lru else load_vector_file
            load_vector_fn = load_vector_file
            max_workers = min(36, max(4, len(history_paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(load_vector_fn, path, idx): path
                for path, idx in zip(history_paths, indices_to_load)
            }
            for future in as_completed(future_to_path):
                np_dict, idx = future.result()
                history_feats[idx] = np_dict
        return history_feats


    def _gather_cosmo_bottleneck(self, epis_idx, step_idx):
        """Gather one cosmo_bottleneck training sample from the per-episode store.

        Reads every per-episode array via mmap (mmap_mode="r") and slices ONLY the
        rows/chunks needed for root step ``step_idx`` (t) -- never full-loads a stream.
        Returns the batch dict the bottleneck + Perceiver-IO decoder consume.
        """
        t = int(step_idx)
        epi_dir = os.path.join(self.feature_dir, f"episode_{epis_idx}")

        cfg = getattr(self.history_config, "cosmo_bottleneck_memory", None)
        num_query = int(getattr(cfg, "num_query_samples", 32)) if cfg is not None else 32
        # PRECOMPUTE PATH: if a frozen-bottleneck token cache is configured, load a random causal
        # variant's (256, dt) tokens and return early -- skips ALL the heavy raw-stream loading
        # (16MB/sample recent-SigLIP etc). Only valid at Stage-2 (bottleneck frozen).
        _memdir = str(getattr(cfg, "precompute_memory_dir", "")) if cfg is not None else ""
        if _memdir:
            # precompute_mem.py records the alignment of its Stage-1 and store inputs.
            require_recorded_alignment(_precompute_alignment(_memdir), cfg, where=_memdir)
            _mp = os.path.join(_memdir, f"ep{epis_idx}_t{t}.npz")
            if os.path.exists(_mp):
                _md = np.load(_mp)
                _vs = list(_md.files)
                if _vs:
                    _v = _vs[np.random.randint(len(_vs))]
                    _sd = int(getattr(cfg, "siglip_dim", 2048))
                    _ad = int(getattr(cfg, "action_in_dim", 12))
                    _pd = int(getattr(cfg, "proprio_in_dim", 16))
                    _ld_ = int(getattr(cfg, "lang_emb_dim", 2048))
                    # tiny placeholders satisfy the RepackTransform structure; UNUSED because
                    # embed_memory takes the cosmo_mem_tokens bypass (bottleneck is frozen).
                    return {
                        "cosmo_mem_tokens": _md[_v].astype(np.float32),
                        "cosmo_latent_emb": np.zeros((1, 48), np.float32),
                        "cosmo_mem_mask": np.zeros((1,), bool),
                        "cosmo_recent_emb": np.zeros((1, _sd), np.float32),
                        "cosmo_recent_mask": np.zeros((1,), bool),
                        "cosmo_action_emb": np.zeros((1, _ad), np.float32),
                        "cosmo_action_mask": np.zeros((1,), bool),
                        "cosmo_proprio_emb": np.zeros((1, _pd), np.float32),
                        "cosmo_proprio_mask": np.zeros((1,), bool),
                        "cosmo_lang_emb": np.zeros((_ld_,), np.float32),
                    }
        siglip_dim = int(getattr(cfg, "siglip_dim", 2048)) if cfg is not None else 2048
        # FIX(bug4): per-episode pooled language embedding (precomputed into the store by
        # augment_store_lang.py from the GENERIC instruction). Same vector Stage-1 offline /
        # Stage-2 / inference feed -> the bottleneck FiLM path trains consistently. Missing
        # file -> zeros == FiLM-identity (lang_film has zero bias), so this degrades gracefully.
        lang_dim = int(getattr(cfg, "lang_emb_dim", 2048)) if cfg is not None else 2048
        _lang_p = os.path.join(epi_dir, "lang_emb.npy")
        cosmo_lang_emb = (np.load(_lang_p).astype(np.float32).reshape(-1)
                          if os.path.exists(_lang_p) else np.zeros((lang_dim,), np.float32))

        # Preserve per-sample stochasticity while respecting the DataLoader seed.
        rng = _next_seeded_generator()
        act_enc = str(getattr(cfg, "action_encoding", "raw_delta")) if cfg is not None else "raw_delta"
        pro_enc = str(getattr(cfg, "proprio_encoding", "raw_delta")) if cfg is not None else "raw_delta"

        # ---- OFFLINE var8 latents (Stage-2 + Stage-1-offline): read a precomputed jitter
        # variant instead of online Wan. Bottleneck is frozen at Stage-2, so precomputed
        # jitter-augmented latents are in-distribution. Detected by latents_v0.npy presence. ----
        _has_variants = os.path.exists(os.path.join(epi_dir, "latents_v0.npy"))
        cfg_ls = str(getattr(cfg, "latent_source", "")) if cfg is not None else ""
        use_offline_variants = _has_variants and (cfg_ls == "offline")
        if use_offline_variants:
            visual_stream = _visual_feature_stream(cfg)
            episode_store = self._get_episode_store_cache(cfg).get(epi_dir)
            chunk_index = episode_store.chunk_index
            # latent c must hold kept frames c*cf .. c*cf+cf-1 under the SAME alignment the
            # evaluator buffer uses (cosmo_bottleneck_memory.wan_chunk_alignment).
            require_store_alignment(chunk_index, cfg, where=epi_dir)
            variant_chunk_frames = int(chunk_index["chunk_frames"])
            recent_mm = episode_store.recent_visual
            ap = episode_store.action_proprio
            n_variants = int(getattr(cfg, "n_variants", 8))
            pool_hw = int(getattr(cfg, "pool_hw", 0)) or None
            pool_h = int(getattr(cfg, "pool_h", 0)) or None
            pool_w = int(getattr(cfg, "pool_w", 0)) or None
            vi = episode_store.variant_index
            cands = [v for v in range(n_variants)
                     if vi[f"v{v}"]["frame_idx"] and vi[f"v{v}"]["frame_idx"][0] <= t]
            # empty cands fallback: v0 == canonical variant, frame_idx[0]==0, causal for all t.
            v = int(rng.choice(cands)) if cands else 0
            vfi = np.array(vi[f"v{v}"]["frame_idx"], np.int64)
            vfi = vfi[vfi <= t]
            n_vjit = len(vfi)
            lat = episode_store.get_latents(v)                               # (nc_full, V*sp, 48)
            nc = max(0, n_vjit // variant_chunk_frames)
            kept = min(lat.shape[0], nc)
            lat = lat[:kept]
            latent_dim = lat.shape[-1] if lat.ndim == 3 else 48
            if pool_h and pool_w and lat.shape[0] > 0:
                # RECTANGULAR grid pool (RoboDojo: non-square 12x16=192 tok/view). Block-mean the
                # (H,W) grid down to (pool_h,pool_w) OUTPUT tokens/view. Grid dims from wan_h/wan_w
                # (Wan spatial stride 16). robocasa (square, pool_hw) uses the branch below, unchanged.
                ncur = lat.shape[0]
                gh = int(getattr(cfg, "wan_h", 192)) // 16
                gw = int(getattr(cfg, "wan_w", 256)) // 16
                ph = int(pool_h); pw = int(pool_w)
                pooled_spatial = ph * pw
                if lat.shape[1] % pooled_spatial == 0 and lat.shape[1] < gh * gw:
                    pass  # already pooled on disk: (chunks, views*pool_h*pool_w, dim)
                else:
                    spatial = gh * gw; Vv = lat.shape[1] // spatial
                    assert lat.shape[1] % spatial == 0, f"unexpected spatial layout {lat.shape} (grid {gh}x{gw})"
                    assert gh % ph == 0 and gw % pw == 0, f"grid {gh}x{gw} must divide by pool ({ph},{pw})"
                    bh = gh // ph; bw = gw // pw
                    gg = lat.reshape(ncur, Vv, gh, gw, latent_dim)
                    gg = gg.reshape(ncur, Vv, ph, bh, pw, bw, latent_dim).mean(axis=(3, 5))
                    lat = gg.reshape(ncur, Vv * ph * pw, latent_dim)
            elif pool_hw and lat.shape[0] > 0:
                lat = pool_square_offline_latents(lat, int(pool_hw))
            cosmo_latent_emb = (np.ascontiguousarray(lat).reshape(-1, latent_dim).astype(np.float32)
                                if lat.shape[0] > 0 else np.zeros((0, latent_dim), np.float32))
            v_boundary = kept * variant_chunk_frames
            v_rec_idx = vfi[v_boundary:] if v_boundary < n_vjit else vfi[n_vjit:]
            if len(v_rec_idx):
                cosmo_recent_emb = (np.ascontiguousarray(recent_mm[v_rec_idx])
                                    .reshape(-1, siglip_dim).astype(np.float32))
            else:
                cosmo_recent_emb = np.zeros((0, siglip_dim), np.float32)
            action_mem_src = ap.get("action_causal", ap["action"])
            v_action = load_variant_causal_action(
                epi_dir,
                variant=v,
                full_variant_frame_idx=vi[f"v{v}"]["frame_idx"],
                selected_variant_frame_idx=vfi,
                global_causal_actions=action_mem_src,
            )
            v_proprio = np.ascontiguousarray(ap["proprio"][vfi]).astype(np.float32)
            dpv = np.zeros_like(v_proprio); dpv[1:] = v_proprio[1:] - v_proprio[:-1]
            if act_enc == "raw":
                cosmo_action_emb = v_action
            else:
                dav = np.zeros_like(v_action); dav[1:] = v_action[1:] - v_action[:-1]
                cosmo_action_emb = np.concatenate([v_action, dav], axis=-1)
            _pe = str(getattr(cfg, "proprio_encoding", "raw_delta")) if cfg is not None else "raw_delta"
            if _pe == "raw":
                cosmo_proprio_emb = v_proprio                               # (n,16) proprio only
            else:
                cosmo_proprio_emb = np.concatenate([v_proprio, dpv], axis=-1)  # (n,32)

            MAX_RECENT = max_recent_tokens(cfg)
            MAX_STEPS = int(getattr(cfg, "max_step_tokens", 0) or 800)
            LATENT_CAP = int(getattr(cfg, "latent_cap", 0)) or 2048  # robocasa default 2048 (no latent_cap key); RoboDojo yaml sets 2560 (pooled ~2052 max)

            def _fixed_pad(arr, L, D):
                out = np.zeros((L, D), np.float32); msk = np.zeros((L,), bool)
                a = arr[-L:] if arr.shape[0] > L else arr
                if a.shape[0]:
                    out[:a.shape[0]] = a; msk[:a.shape[0]] = True
                return out, msk

            rec_p, rec_m = _fixed_pad(cosmo_recent_emb, MAX_RECENT, siglip_dim)
            _ad = cosmo_action_emb.shape[-1] if cosmo_action_emb.size else (12 if act_enc == "raw" else 24)
            act_p, act_m = _fixed_pad(cosmo_action_emb, MAX_STEPS, _ad)
            _pd = cosmo_proprio_emb.shape[-1] if cosmo_proprio_emb.size else 32
            pro_p, pro_m = _fixed_pad(cosmo_proprio_emb, MAX_STEPS, _pd)
            if cosmo_latent_emb.shape[0] > LATENT_CAP:
                logger.warning(
                    "cosmo_bottleneck offline latents truncated: epi=%d t=%d tokens=%d cap=%d "
                    "(keeping most-recent [-%d:]); consider pool_hw to stay within cap.",
                    int(epis_idx), t, int(cosmo_latent_emb.shape[0]), LATENT_CAP, LATENT_CAP)
            lat_p = np.zeros((LATENT_CAP, cosmo_latent_emb.shape[-1] if cosmo_latent_emb.size else 48), np.float32)
            lat_m = np.zeros((LATENT_CAP,), bool)
            _lc = min(cosmo_latent_emb.shape[0], LATENT_CAP)
            if _lc:
                lat_p[:_lc] = cosmo_latent_emb[-LATENT_CAP:][:_lc]; lat_m[:_lc] = True
            _off = {
                "cosmo_latent_emb": lat_p, "cosmo_mem_mask": lat_m,
                "cosmo_recent_emb": rec_p, "cosmo_recent_mask": rec_m,
                "cosmo_action_emb": act_p, "cosmo_action_mask": act_m,
                "cosmo_proprio_emb": pro_p, "cosmo_proprio_mask": pro_m,
                "cosmo_lang_emb": cosmo_lang_emb,
            }
            # E2E+aux (offline path): build reconstruction targets from the variant's frames (vfi).
            # Only when need_aux -> action-only/other runs keep the original offline dict unchanged.
            if self.need_aux and n_vjit > 0:
                # FIXED query count Qf so every sample has identical shapes (default collate must
                # stack across the batch). Sample Kq<=Qf real queries, pad the rest and mask them.
                Qf = int(num_query)
                Kq = min(Qf, n_vjit)
                q_loc = (np.arange(n_vjit, dtype=np.int64) if Kq >= n_vjit
                         else np.sort(rng.choice(n_vjit, size=Kq, replace=False)).astype(np.int64))
                qfi = vfi[q_loc]                                  # store rows for targets
                # Prefer the fp16 align target (half the disk read + host->device transfer; the
                # target is LayerNorm-normalized under stop_gradient in the loss, so fp16's ~0.016
                # roundtrip error is numerically irrelevant). Falls back to fp32 if not converted.
                if visual_stream == "dinov3":
                    a_al = np.ascontiguousarray(recent_mm[qfi]).reshape(
                        Kq, -1, siglip_dim
                    )
                else:
                    _al_fp16 = os.path.join(epi_dir, "align_prefix_fp16.npy")
                    _al_path = _al_fp16 if os.path.exists(_al_fp16) else os.path.join(epi_dir, "align_prefix.npy")
                    align_mm = np.load(_al_path, mmap_mode="r")
                    a_al = np.ascontiguousarray(align_mm[qfi])
                a_ac = np.ascontiguousarray(ap["action"][qfi]).astype(np.float32)  # (Kq,da)
                a_pr = np.ascontiguousarray(ap["proprio"][qfi]).astype(np.float32) # (Kq,dp)
                # pad to Qf along axis 0 (pad rows are masked out in the loss)
                def _pad0(x):
                    if x.shape[0] >= Qf:
                        return x[:Qf]
                    pad = np.zeros((Qf - x.shape[0], *x.shape[1:]), dtype=x.dtype)
                    return np.concatenate([x, pad], axis=0)
                qs = np.zeros((Qf,), np.int32); qs[:Kq] = q_loc.astype(np.int32)
                qm = np.zeros((Qf,), bool); qm[:Kq] = True
                _off["aux_query_steps"] = qs                      # INT for nnx.Embed (finding #3)
                _off["aux_query_mask"] = qm
                _off["aux_align_target"] = _pad0(a_al)            # (Qf,P,D)
                _off["aux_action_target"] = _pad0(a_ac)           # (Qf,da)
                _off["aux_proprio_target"] = _pad0(a_pr)          # (Qf,dp)
                reward_path = os.path.join(epi_dir, "reward.npy")
                _rw = np.float32(0.0)
                if os.path.exists(reward_path):
                    rd = np.load(reward_path, allow_pickle=True).item()
                    rmean = np.stack([rd[m] for m in ("incremental","forward","backward")], 0).mean(0)
                    _rw = np.float32(np.clip(rmean[int(vfi[-1])], 0.0, 1.0))
                _off["aux_reward_target"] = _rw
            return _off

        if wan_chunk_alignment_from_config(cfg) != NATIVE:
            raise NotImplementedError(
                "wan_chunk_alignment other than 'native' needs offline variant latents "
                "(latent_source: offline, latents_v*.npy re-encoded with "
                "mme_vla_suite.dataset_builder.reencode_wan_latents)"
            )
        # chunk_index is tiny JSON -> load directly.
        with open(os.path.join(epi_dir, "chunk_index.json")) as f:
            chunk_index = json.load(f)
        chunk_frames = int(chunk_index["chunk_frames"])
        latents_online = bool(chunk_index.get("latents_online", False))

        # ------------------------------------------------------------------ #
        # #1/#3 FRAME-INDEX JITTER: choose WHICH stored (ds=1) frames form the
        # buffer, covering [s, t]. These frames, in strict time order, are the
        # video the ONLINE Wan tokenizer compresses; every other stream is
        # sliced at the SAME indices so buffer == a real (jittered) rollout.
        # ------------------------------------------------------------------ #
        frame_idx = self._sample_frame_indices(t, rng)          # (n_jit,) ascending, ends at t
        n_jit = int(frame_idx.shape[0])

        # split jittered frames into consolidated (Wan chunks) vs recent (SigLIP).
        num_chunks = n_jit // chunk_frames                       # finalized complete chunks
        boundary = num_chunks * chunk_frames                     # first recent frame (in jittered order)

        # --- raw frames for ONLINE Wan latents (train step encodes them) ---
        # returned as a per-sample array (variable length) -> custom collate keeps a list.
        frames_mm = np.load(os.path.join(epi_dir, "frames.npy"), mmap_mode="r")  # (T,V,R,R,3) uint8
        cosmo_frames = np.ascontiguousarray(frames_mm[frame_idx])                # (n_jit,V,R,R,3)

        # --- cosmo_recent_emb: precomputed SigLIP at the RECENT jittered frames ---
        visual_stream = _visual_feature_stream(cfg)
        recent_mm = np.load(
            os.path.join(epi_dir, visual_feature_filename(visual_stream)),
            mmap_mode="r",
        )
        recent_idx = frame_idx[boundary:]                        # trailing frames after last chunk
        if recent_idx.size:
            recent_win = np.ascontiguousarray(recent_mm[recent_idx])   # (n_recent,V,64,dim)
            cosmo_recent_emb = recent_win.reshape(-1, siglip_dim).astype(np.float32, copy=False)
        else:
            cosmo_recent_emb = np.zeros((0, siglip_dim), np.float32)

        # --- action / proprio at ALL jittered frames : concat(raw, delta-over-jittered-seq) ---
        ap = np.load(os.path.join(epi_dir, "action_proprio.npy"), allow_pickle=True).item()
        action_mem_src = ap.get("action_causal", ap["action"])
        action = np.ascontiguousarray(action_mem_src[frame_idx]).astype(np.float32)   # causal memory action
        proprio = np.ascontiguousarray(ap["proprio"][frame_idx]).astype(np.float32)  # (n_jit,16)
        # deltas recomputed over the JITTERED sequence (Δ between consecutive kept frames),
        # NOT the store's per-raw-frame da/dp -- jitter changes the effective step spacing.
        # action_encoding controls whether the memory sees action+Δ (raw_delta, 24d) or the
        # raw action only (raw, 12d). proprio always carries its delta (state dynamics matter).
        if act_enc == "raw":
            cosmo_action_emb = action                                # (n_jit,12) action only
        else:  # raw_delta
            da = np.zeros_like(action); da[1:] = action[1:] - action[:-1]
            cosmo_action_emb = np.concatenate([action, da], axis=-1)  # (n_jit,24)
        if pro_enc == "raw":
            cosmo_proprio_emb = proprio                              # (n_jit,16) proprio only
        else:
            dp = np.zeros_like(proprio); dp[1:] = proprio[1:] - proprio[:-1]
            cosmo_proprio_emb = np.concatenate([proprio, dp], axis=-1)  # (n_jit,32)

        # --- Lever-1 sampled queries: subset of the JITTERED frames (predict those steps) ---
        K = min(int(num_query), n_jit)
        if K >= n_jit:
            q_local = np.arange(n_jit, dtype=np.int64)
        else:
            q_local = np.sort(rng.choice(n_jit, size=K, replace=False)).astype(np.int64)
        query_steps = q_local                                    # decoder step-embed index (position in buffer)
        query_frame_idx = frame_idx[q_local]                     # -> store row for the targets

        # --- targets at the queried jittered frames (Stage-1 pretrain reads these directly) ---
        if visual_stream == "dinov3":
            align_target = np.ascontiguousarray(recent_mm[query_frame_idx]).reshape(
                len(query_frame_idx), -1, siglip_dim
            ).astype(np.float32)
        else:
            align_mm = np.load(os.path.join(epi_dir, "align_prefix.npy"), mmap_mode="r")
            align_target = np.ascontiguousarray(align_mm[query_frame_idx]).astype(np.float32)
        action_target = np.ascontiguousarray(ap["action"][query_frame_idx]).astype(np.float32)  # raw 12
        proprio_target = np.ascontiguousarray(ap["proprio"][query_frame_idx]).astype(np.float32)  # raw 16

        # --- train-only input noise (#2, default off) on the ENCODER streams; targets clean ---
        cosmo_recent_emb = self._augment_stream(cosmo_recent_emb, "recent")
        cosmo_action_emb = self._augment_stream(cosmo_action_emb, "action")
        cosmo_proprio_emb = self._augment_stream(cosmo_proprio_emb, "proprio")

        out = {
            # ONLINE-latent path: train step runs Wan on cosmo_frames -> cosmo_latent_emb.
            "cosmo_frames": cosmo_frames,               # (n_jit,V,R,R,3) uint8  [per-sample list]
            "cosmo_frames_boundary": np.int64(boundary),  # frames[:boundary] -> chunks; rest recent
            "cosmo_recent_emb": cosmo_recent_emb,
            "cosmo_action_emb": cosmo_action_emb,
            "cosmo_proprio_emb": cosmo_proprio_emb,
            "cosmo_lang_emb": cosmo_lang_emb,
            "query_steps": query_steps,
            "align_target": align_target,
            "action_target": action_target,
            "proprio_target": proprio_target,
        }
        if not latents_online:
            # (legacy path, ds=2 store with latents.npy) -- keep for back-compat / invariant tests.
            latent_dim = int(getattr(cfg, "cosmos_latent_dim", 48)) if cfg is not None else 48
            latents_mm = np.load(os.path.join(epi_dir, "latents.npy"), mmap_mode="r")
            prefix = (np.ascontiguousarray(latents_mm[:num_chunks]).reshape(-1, latent_dim)
                      if num_chunks > 0 else np.zeros((0, latent_dim), np.float32))
            out["cosmo_latent_emb"] = prefix.astype(np.float32, copy=False)
            out["cosmo_mem_mask"] = np.ones((prefix.shape[0],), bool)

        # --- reward target: SINGLE current-step scalar in (0,1), mean of the 3 modes ---
        # Reward is a GLOBAL readout of task progress at the CURRENT step (frame_idx[-1] == t),
        # NOT a per-step sequence -- only memory can disambiguate current progress. Kept in
        # (0,1) (NOT standardized); decoder sigmoid matches -> bounded MSE.
        reward_path = os.path.join(epi_dir, "reward.npy")
        if os.path.exists(reward_path):
            rd = np.load(reward_path, allow_pickle=True).item()   # {mode: (steps,)}
            modes = np.stack([rd[m] for m in ("incremental", "forward", "backward")], axis=0)  # (3,steps)
            rmean = modes.mean(axis=0)                            # (steps,) unified avg reward in [0,1]
            cur = int(frame_idx[-1])                              # current step == last jittered frame
            out["reward_target"] = np.float32(np.clip(rmean[cur], 0.0, 1.0))  # scalar

        return out

    def prepare_token_drop(self, epis_idx, step_idx):
        token_budget = self.history_config.budget
        kept_indices = json.load(open(os.path.join(self.feature_dir, f"episode_{epis_idx}", "kept_indices.json")))

        return self.mem_buffer.prepare_token_dropping(
            step_idx, token_budget, self._gather_history_feat,
            kept_indices=kept_indices, epis_idx=epis_idx)


    def prepare_frame_sampling(self,  epis_idx,  step_idx):
        token_per_image = self.history_config.token_per_image
        token_budget = self.history_config.budget

        return self.mem_buffer.prepare_frame_sampling(
            step_idx, token_budget, token_per_image, self._gather_history_feat,
            epis_idx=epis_idx)



    def prepare_token_recurrent(self, epis_idx, step_idx, exec_start_idx):
        self.mem_buffer.exec_start_idx = exec_start_idx

        return self.mem_buffer.prepare_token_recurrent(
            step_idx, exec_start_idx, self._gather_history_feat,
            epis_idx=epis_idx)


    def __len__(self):
        return len(self.dataset)


    def _normalize_state(self, state):
        if self.compute_norm_stats:
            return state
        else:
            if self.use_quantiles:
                return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
            else:
                return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _episode_len(self, epis_idx: int) -> int:
        epis_idx = int(epis_idx)
        cached = self._episode_len_cache.get(epis_idx)
        if cached is not None:
            return cached
        epi_dir = self.feature_dir / f"episode_{epis_idx}"
        cfg = getattr(self.history_config, "cosmo_bottleneck_memory", None)
        for name in (
            "frames.npy",
            visual_feature_filename(_visual_feature_stream(cfg)),
            "latents_v0.npy",
        ):
            path = epi_dir / name
            if path.exists():
                arr = np.load(path, mmap_mode="r")
                length = int(arr.shape[0])
                if name.startswith("latents_"):
                    length *= 4
                self._episode_len_cache[epis_idx] = max(1, length)
                return self._episode_len_cache[epis_idx]
        ap_path = epi_dir / "action_proprio.npy"
        if ap_path.exists():
            ap = np.load(ap_path, allow_pickle=True).item()
            length = int(ap["action"].shape[0])
            self._episode_len_cache[epis_idx] = max(1, length)
            return self._episode_len_cache[epis_idx]
        self._episode_len_cache[epis_idx] = 1
        return 1

    def _truncated_gaussian_noise(self, max_val: int, std: float = None) -> int:
        """Generate truncated Gaussian noise centered at 0, clamped to [-max_val, max_val]."""
        if std is None:
            std = max_val / 2.5  # ~95% of samples within range before clamping
        noise = random.gauss(0, std)
        noise = max(-max_val, min(max_val, noise))  # clamp to [-max_val, max_val]
        return int(round(noise))

    def add_grounding_augmentation(self, subgoal: str, noise_range: int = 8) -> str:
        if not subgoal:
            return subgoal

        matches = re.findall(r'at <(\d+), (\d+)>', subgoal)

        if len(matches) == 0 or len(matches) > 1:
            return subgoal

        x, y = matches[0]
        x = int(x)
        y = int(y)
        noise_x = self._truncated_gaussian_noise(noise_range)
        noise_y = self._truncated_gaussian_noise(noise_range)

        new_subgoal = subgoal.replace(f'at <{x}, {y}>', f'at <{x + noise_x}, {y + noise_y}>')
        return new_subgoal


    def __getitem__(self, idx):
        data = self.dataset[idx]

        data["actions"] = data["actions"][:self.action_horizon]
        if data["actions"].shape[0] != self.action_horizon:
            raise ValueError(
                f"sample {idx} has action horizon {data['actions'].shape[0]}, "
                f"expected {self.action_horizon}; rebuild the store for this horizon"
            )

        epis_idx_for_weight = int(data["epis_idx"].item())
        step_idx_for_weight = int(data["step_idx"].item())
        if self.key_event_weights is not None:
            data["sample_loss_weight"] = np.asarray(
                self.key_event_weights.weight_for(
                    epis_idx_for_weight, step_idx_for_weight
                ),
                dtype=np.float32,
            )
        elif self.late_episode_loss_weight_value > 1.0:
            data["sample_loss_weight"] = np.asarray(
                late_episode_loss_weight(
                    step_idx_for_weight,
                    self._episode_len(epis_idx_for_weight),
                    late_fraction=self.late_episode_loss_fraction,
                    late_weight=self.late_episode_loss_weight_value,
                ),
                dtype=np.float32,
            )

        # During online evaluation, the ground-truth subgoal may change earlier than when it was recorded.
        # To make the model robust to this temporal shift and avoid train/test distribution mismatch,
        # we randomly sample from either subgoal or subgoal_online (early change).
        if self.history_config is not None \
            and self.history_config.representation_type == "symbolic" \
            and self.history_config.symbolic_memory.type in ["simple_subgoal", "grounded_subgoal"] \
            and random.random() < 0.5:
            data["simple_subgoal"] = data["simple_subgoal_online"]
            data["grounded_subgoal"] = data["grounded_subgoal_online"]
        data.pop("simple_subgoal_online")
        data.pop("grounded_subgoal_online")

        if self.history_config is not None and self.history_config.representation_type == "symbolic":
            data["grounded_subgoal"] = self.add_grounding_augmentation(data["grounded_subgoal"], noise_range=8)
            data["simple_subgoal"] = self.add_grounding_augmentation(data["simple_subgoal"], noise_range=8)


        if self.history_config is not None:
            # use history
            epis_idx = data["epis_idx"].item()
            step_idx = data["step_idx"].item()
            exec_start_idx = data["exec_start_idx"].item()

            if self.history_config.representation_type == "perceptual":
                if self.history_config.perceptual_memory.type == "token_dropping":
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask, # >=64
                    ) = self.prepare_token_drop(epis_idx, step_idx)
                else:
                    fs_out = self.prepare_frame_sampling(epis_idx, step_idx)
                    if self.use_time:
                        (static_img_emb, static_pos_emb, static_state_emb,
                         static_mask, static_time) = fs_out
                    else:
                        (static_img_emb, static_pos_emb, static_state_emb,
                         static_mask) = fs_out
                        static_time = None

                data["static_image_emb"] = np.asarray(static_img_emb, dtype=np.float32)
                data["static_pos_emb"] = np.asarray(static_pos_emb, dtype=np.float32)
                data["static_state_emb"] = np.asarray(self._normalize_state(static_state_emb), dtype=np.float32)
                data["static_mask"] = static_mask
                if static_time is not None:
                    data["static_time"] = np.asarray(static_time, dtype=np.float32)  # (budget, 1) elapsed

            elif self.history_config.representation_type == "recurrent":
                recur_img_emb, recur_pos_emb, recur_state_emb, recur_mask = (
                    self.prepare_token_recurrent(
                        epis_idx, step_idx, exec_start_idx)
                )

                data["recur_image_emb"] = np.asarray(recur_img_emb, dtype=np.float32)
                data["recur_pos_emb"] = np.asarray(recur_pos_emb, dtype=np.float32)
                data["recur_state_emb"] = np.asarray(self._normalize_state(recur_state_emb), dtype=np.float32)
                data["recur_mask"] = recur_mask  # 1 ~ max_exec_steps

            elif self.history_config.representation_type == "symbolic":
                pass
            elif self.history_config.representation_type == "cosmo_bottleneck":
                cosmo = self._gather_cosmo_bottleneck(epis_idx, step_idx)
                # The gather always emits reconstruction targets (Stage-1 pretrain needs them via the
                # direct-gather path). For the VLA dataloader: only KEEP them (renamed aux_*) when the
                # run requests aux; otherwise DROP so action-only training carries no extra labels.
                _tgt = {"query_steps", "align_target", "action_target", "proprio_target", "reward_target"}
                for k, v in cosmo.items():
                    if k in _tgt:
                        if self.need_aux:
                            data["aux_" + k] = v
                        # else: drop (action-only)
                    else:
                        data[k] = v
                if self.need_aux and "aux_query_steps" not in data:
                    # online path: query_steps comes from the gather's top-level key -> rename + INT dtype
                    q = cosmo.get("query_steps")
                    if q is not None:
                        data["aux_query_steps"] = np.asarray(q, np.int32)   # INT for nnx.Embed (finding #3)
                        data["aux_query_mask"] = np.ones((np.asarray(q).shape[0],), bool)
            else:
                raise ValueError(
                    f"Not supported representation type: {self.history_config.representation_type}"
                )


        for key in [
            "static_image_emb",
            "static_pos_emb",
            "static_state_emb",
            "static_mask",
            "static_time",

            "recur_image_emb",
            "recur_pos_emb",
            "recur_state_emb",
            "recur_mask",

            "cosmo_latent_emb",
            "cosmo_mem_mask",
            "cosmo_recent_emb",
            "cosmo_action_emb",
            "cosmo_proprio_emb",
            "cosmo_lang_emb",
            "query_steps",
            "align_target",
            "action_target",
            "proprio_target",
            "reward_target",

            "simple_subgoal",
            "grounded_subgoal",
            "prompt",
            "sample_loss_weight",
        ]:
            if key not in data:
                data[key] = None

        return data


def load_variant_causal_action(
    episode_dir: str,
    *,
    variant: int,
    full_variant_frame_idx: np.ndarray,
    selected_variant_frame_idx: np.ndarray,
    global_causal_actions: np.ndarray,
) -> np.ndarray:
    """Return the causal action timeline aligned to an offline latent variant."""
    full_idx = np.asarray(full_variant_frame_idx, dtype=np.int64)
    selected_idx = np.asarray(selected_variant_frame_idx, dtype=np.int64)
    if selected_idx.size > full_idx.size or not np.array_equal(selected_idx, full_idx[: selected_idx.size]):
        raise ValueError(f"selected frame indices are not a prefix of variant v{variant}")

    local_path = os.path.join(episode_dir, f"action_causal_v{variant}.npy")
    if os.path.exists(local_path):
        local_actions = np.load(local_path, mmap_mode="r")
        if local_actions.shape[0] != full_idx.size:
            raise ValueError(
                f"{os.path.basename(local_path)} length {local_actions.shape[0]} "
                f"does not match variant_index length {full_idx.size}"
            )
        return np.ascontiguousarray(local_actions[: selected_idx.size]).astype(np.float32)

    return np.ascontiguousarray(global_causal_actions[selected_idx]).astype(np.float32)
