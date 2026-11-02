"""Online dual-stream memory buffer for consolidated video latents and recent visual tokens.

DROP-TERMINAL: only finalized (non-terminal) Cosmos chunks are consolidated; the newest
complete-but-provisional chunk stays in the recent-SigLIP window until a newer chunk exists.

A chunk of chunk_frames downsampled frames becomes FINALIZED as soon as at least one
frame of the NEXT chunk has arrived (that makes the earlier chunk non-terminal, hence
bit-stable). Concretely, after appending a frame, while len(_recent_frames) > chunk_frames
we pop the oldest chunk_frames frames and append the finalized latent for that chunk to
_consolidated. CAUSAL: the finalized latent is the c-th chunk of a causal encode over the
GROWING PREFIX of every downsampled frame seen so far (matching the offline encode-once store
builder), NOT the popped chunk encoded standalone -- standalone re-encoding discards the
strictly-causal temporal context and gives the wrong latent for every non-leading chunk.
The recent visual window therefore spans every frame
since the last finalized boundary (up to 2*chunk_frames-1 frames: the most recent complete
provisional chunk plus the current partial chunk).

Memory action alignment (the memory_action_alignment argument; only the RMBench launcher sets it).
``add_buffer`` receives, for raw step s, the action that was executed from raw step s-1 (the
evaluators pair each new observation with the action that produced it; step 0 has none).
  * "causal_raw_previous" (DEFAULT, released evaluator behavior): kept frame r stores the
    action supplied with step r, i.e. the action executed from raw step r-1. This is the
    canonical action_causal stream sampled at the kept frames.
  * "previous_kept_frame": kept frame f_j stores the action executed at the previous kept frame
    f_{j-1}, and zeros for the first kept frame. This is paper Eq. A.1 and the rule of the
    offline variant files (action_causal_v*.npy, shared/variant_actions.py). With
    downsample == 1 the two modes are identical.

Action encoding modes (config-selectable via action_encoding):
  * "raw_delta" (DEFAULT, backward compatible): cosmo_action_emb = concat(action, Δaction),
    width = 2*action_dim (=24 for the 12-dim robocasa action). Δ[i]=action[i]-action[i-1].
  * "geometric" (Cosmos-3 style): reframe the 12-dim robocasa action
    [eef_pos(3), eef_rot_rotvec(3), gripper(1), base_motion(4), control_mode(1)] into a single
    15-dim geometric vector [eef_pos_delta(3), eef_rot_6d(6), gripper_abs(1), base(4), mode(1)].
    The 3-dim eef_rot rotation-vector DELTA is converted to a continuous 6D rotation
    representation (Zhou et al. 2019 = first two columns of the rotation matrix). Grasp and
    control_mode pass through as absolute/current values. Unlike raw_delta, geometric mode does
    NOT append a separate Δ concat: the relative pose already encodes motion (Cosmos-3 scheme),
    so the width is 15, not 30.
"""
import os
import time

import einops
import jax
import numpy as np

from mme_vla_suite.shared.data_utils import pool_tokens_to_size
from openpi.shared import image_tools


_DINOV3_ENCODER_CACHE = {}
# benchmarks/rmbench/eval/run_local.py repeats these names (it does not import mme_vla_suite).
# "causal_raw_previous" has the same meaning as the RoboCasa MemoryActionStream mode of that
# name (benchmarks/robocasa/eval/eval_standard_3view.py). "previous_kept_frame" differs from
# the RoboCasa "stage2_selected_previous" mode, which delays the action by one raw step.
MEMORY_ACTION_ALIGNMENTS = ("causal_raw_previous", "previous_kept_frame")


def rotvec_to_6d(rotvec):
    """Convert an axis-angle rotation vector (3,) to a continuous 6D rotation (Zhou 2019).

    rotvec -> R (3x3) via Rodrigues exp-map -> first two columns of R, laid out
    column-major as [col0(3), col1(3)] = R[:, :2].reshape(-1, order="F").
    Identity (theta~0) returns [1,0,0, 0,1,0].
    """
    rotvec = np.asarray(rotvec, np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        R = np.eye(3, dtype=np.float64)
    else:
        k = rotvec / theta
        K = np.array([[0.0, -k[2], k[1]],
                      [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]], dtype=np.float64)
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R[:, :2].reshape(-1, order="F").astype(np.float32)


class CosmoMemoryBuffer:
    def __init__(self, cosmo, siglip_fn, chunk_frames=8, downsample=2, num_views=1,
                 state_dim=16, action_dim=12, latent_dim=16, max_steps=2048,
                 action_encoding="raw_delta", proprio_encoding="raw_delta", pool_hw=None,
                 pool_h=None, pool_w=None, wan_h=0, wan_w=0,
                 memory_action_alignment="causal_raw_previous"):
        self.cosmo = cosmo
        self.siglip_fn = siglip_fn
        self.chunk_frames = chunk_frames
        self.downsample = downsample
        self.num_views = num_views
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_steps = max_steps
        if action_encoding not in ("raw_delta", "geometric", "raw"):
            raise ValueError(f"unknown action_encoding: {action_encoding!r}")
        self.action_encoding = action_encoding
        if proprio_encoding not in ("raw_delta", "raw"):
            raise ValueError(f"unknown proprio_encoding: {proprio_encoding!r}")
        self.proprio_encoding = proprio_encoding
        if memory_action_alignment not in MEMORY_ACTION_ALIGNMENTS:
            raise ValueError(
                f"unknown memory action alignment {memory_action_alignment!r}; "
                f"expected one of {MEMORY_ACTION_ALIGNMENTS}"
            )
        self.memory_action_alignment = memory_action_alignment
        # FIX(eval parity): pool consolidated Wan latents to pool_h x pool_w per view, EXACTLY as
        # training dataset._gather_cosmo_bottleneck does. robocasa: square 8x8 grid, pool_hw=2 ->
        # 4 tok/view. RoboDojo: NON-square 12x16 grid (192x256 Wan input), pool_h=pool_w=2 -> also
        # 4 tok/view but the grid factoring differs. pool_h/pool_w + wan_h/wan_w make it grid-agnostic;
        # falling back to pool_hw (square, spatial=64) keeps robocasa's behavior byte-identical.
        self.pool_hw = int(pool_hw) if pool_hw else None
        self.pool_h = int(pool_h) if pool_h else None
        self.pool_w = int(pool_w) if pool_w else None
        self.wan_h = int(wan_h) if wan_h else 0
        self.wan_w = int(wan_w) if wan_w else 0
        # OPTION B: per-frame recent-SigLIP token cache (encode each frame ONCE).
        # Toggled by env RECENT_SIGLIP_CACHE=1. Default off -> Option A (unchanged).
        self._recent_cache = os.environ.get("RECENT_SIGLIP_CACHE", "0") == "1"
        # OPTIONAL train-parity: commit ALL complete chunks (incl. terminal) so the Wan-latent
        # count == offline training, and the recent stream sees only n%%cf frames (like training).
        # Safe for Wan2.2 (terminal chunk already causally finalized). Default OFF (drop-terminal).
        self._consolidate_all = os.environ.get("EVAL_CONSOLIDATE_ALL", "0") == "1"
        self._visual_stream = os.environ.get("EVAL_VISUAL_PROFILE", "siglip")
        if self._visual_stream not in ("siglip", "dinov3"):
            raise ValueError(f"unsupported EVAL_VISUAL_PROFILE={self._visual_stream!r}")
        self._visual_dim = 768 if self._visual_stream == "dinov3" else 2048
        self._dinov3_encoder = None
        # RECENT_SIGLIP_VERIFY=1: on each _encode_recent, also run the A path and
        # compare (cosine / max_abs_diff) -> append to the timing/verify log. Slow.
        self._recent_verify = os.environ.get("RECENT_SIGLIP_VERIFY", "0") == "1"
        self._timing_log = os.environ.get("MME_TIMING_LOG", "") or None
        self._verify_log = os.environ.get("MME_VERIFY_LOG", "") or None
        self.clear()

    def clear(self):
        self._consolidated = []      # finalized chunk latents, each (spatial_tokens, latent_dim)
        self._delta_pending_chunk_indices = []
        self._recent_frames = []     # downsampled frames since last FINALIZED boundary
        self._all_ds_frames = []
        self._all_ds_frames_hi = []   # DUAL-RES: 256px frames for recent-SigLIP (train parity)     # ALL downsampled frames (kept so finalized chunks can be
        # OPTION B cache: per-frame recent-SigLIP tokens, aligned 1:1 with
        # _all_ds_frames_hi. entry i = (V, tokens, dim) float32 for absolute frame i.
        # keyed by absolute index -> gathering [_rec_start:] stays correct across chunk finalize.
        self._all_ds_frames_hi_tokens = []
                                     # re-encoded over the GROWING PREFIX -- causal consolidation)
        self._states = []            # every step state
        self._actions = []           # memory action of every kept frame
        self._last_kept_step = None  # raw step of the newest kept frame
        self._action_from_last_kept = None  # action executed from that kept frame
        self._lang_emb = None
        self._seen = 0
        # O(T) streaming cache: if the frontend can make a streamer (Wan), use it to encode
        # ONLY the new chunk each finalization (verified byte-exact vs full re-encode) instead
        # of the O(T^2) full-prefix path. `_stream_all` holds ALL chunks the streamer has
        # emitted (incl. the currently-terminal one); `_consolidated` exposes drop-terminal
        # (== offline store's finalized_encode == encode[:-1]).
        self._streamer = self.cosmo.make_streamer(self.num_views) if hasattr(self.cosmo, "make_streamer") else None
        self._stream_all = []        # every chunk the streamer emitted (terminal included)
        self._pushed_ds = 0          # how many _all_ds_frames already pushed to the streamer

    def set_language(self, lang_emb):
        self._lang_emb = None if lang_emb is None else np.asarray(lang_emb, np.float32)

    def add_buffer(self, images, states, actions, step_idx_list, images_siglip=None):
        _t_ab0 = time.perf_counter()
        old_consolidated_count = len(self._consolidated)
        # images: (t,v,h,w,3) uint8 ; keep num_views views ; downsample by step index
        for i, raw_step in enumerate(step_idx_list):
            step = int(raw_step)
            if self._last_kept_step is not None and step == self._last_kept_step + 1:
                # actions[i] was executed from the newest kept frame.
                self._action_from_last_kept = np.asarray(actions[i], np.float32)
            # FIX(eval parity): training indexes action/proprio at the DOWNSAMPLED frame_idx
            # (every -th step), NOT every step. Gate states/actions by the SAME
            # downsample as frames so all three streams share the frame_idx cadence.
            if step % self.downsample != 0:
                continue
            self._states.append(np.asarray(states[i], np.float32))
            self._actions.append(self._memory_action(step, actions[i]))
            self._last_kept_step = step
            self._action_from_last_kept = None
            self._all_ds_frames.append(np.asarray(images[i, :self.num_views]))  # (V,h,w,3)
            if images_siglip is not None:
                _hi = np.asarray(images_siglip[i, :self.num_views])
            else:
                _hi = np.asarray(images[i, :self.num_views])
            self._all_ds_frames_hi.append(_hi)
            # OPTION B: encode THIS frame's recent-SigLIP tokens once, now, and cache.
            if self._recent_cache:
                self._all_ds_frames_hi_tokens.append(self._encode_one_frame(_hi))
        # CAUSAL CONSOLIDATION driven off the frontend's OWN finalized-chunk count (single
        # source of truth -> no cadence/finalized_encode off-by-one). Both Wan and Cosmos are
        # strictly causal, so chunk c's latent is the c-th chunk of a causal encode over the
        # growing prefix -- byte-for-byte the offline store builder's encode-once path. We
        # commit any chunks that finalized_encode now reports beyond what we've committed.
        # Gate on floor(frames/cf) > committed (necessary for a new chunk under BOTH the real
        # frontend's warmup+terminal drop accounting and the plain fake accounting) so we only
        # re-encode when a new chunk could exist. NOTE O(T^2): re-encodes the whole prefix;
        # a stateful causal cache (tracked separately) restores O(T).
        n_ds = len(self._all_ds_frames)
        if self._streamer is not None:
            # O(T) path: push only the NEW downsampled frames; streamer emits new chunks.
            if n_ds > self._pushed_ds:
                new = np.stack(self._all_ds_frames[self._pushed_ds:], 0)   # (k, V, h, w, 3)
                self._pushed_ds = n_ds
                self._stream_all.extend(self._streamer.push(new))
            # Drop-terminal commits all but the currently-terminal chunk. Consolidate-all
            # follows the fixed floor(n/chunk_frames) split used by the training stores.
            if self._consolidate_all:
                n_finalized = min(
                    len(self._stream_all), n_ds // self.chunk_frames
                )
            else:
                n_finalized = max(0, len(self._stream_all) - 1)
            for c in range(len(self._consolidated), n_finalized):
                self._consolidated.append(np.asarray(self._stream_all[c], np.float32))
        elif n_ds // self.chunk_frames > len(self._consolidated):
            # O(T^2) fallback (Cosmos / test fakes without a streamer): full-prefix re-encode.
            finalized = self._encode_all_chunks() if self._consolidate_all else self._finalized_encode_all()  # (num_finalized, V*spatial, latent)
            if self._consolidate_all:
                finalized = finalized[: n_ds // self.chunk_frames]
            for c in range(len(self._consolidated), int(finalized.shape[0])):
                self._consolidated.append(np.asarray(finalized[c], np.float32))
        # recent window = downsampled frames after the last finalized chunk boundary
        _rec_start = len(self._consolidated) * self.chunk_frames
        # Wan uses lo-res frames (self._all_ds_frames); the recent stream uses 256px frames for
        # train/eval resolution parity. Same indices -> same frames, different resolution.
        self._recent_frames = self._all_ds_frames_hi[_rec_start:]
        self._seen = len(self._states)
        if len(self._consolidated) > old_consolidated_count:
            self._delta_pending_chunk_indices.extend(
                range(old_consolidated_count, len(self._consolidated))
            )
        if self._timing_log is not None:
            _ms = (time.perf_counter() - _t_ab0) * 1000.0
            self._log_timing("add_buffer", _ms, n_ds=len(self._all_ds_frames),
                             rec=len(self._recent_frames))

    def _memory_action(self, step, supplied):
        """Return the memory action of the kept frame at raw ``step``."""
        supplied = np.asarray(supplied, np.float32)
        if self.memory_action_alignment == "causal_raw_previous":
            return supplied
        if self._last_kept_step is None:
            return np.zeros_like(supplied)  # Eq. A.1: a^mem_0 = 0
        if self._action_from_last_kept is None:
            raise RuntimeError(
                "previous_kept_frame alignment needs the action executed from kept "
                f"step {self._last_kept_step}, but step {self._last_kept_step + 1} was "
                f"never buffered before kept step {step}"
            )
        return self._action_from_last_kept

    def _encode_all_chunks(self):
        """Like _finalized_encode_all but KEEPS the terminal chunk (EVAL_CONSOLIDATE_ALL).
        Wan2.2 terminal chunk is already causal/stable, so encode_multiview's full complete-chunk
        set == offline store's chunk set (which also kept all complete chunks)."""
        frames = np.stack(self._all_ds_frames, 0)
        V = frames.shape[1]
        if V > 1 and hasattr(self.cosmo, "encode_multiview"):
            return self.cosmo.encode_multiview(frames)
        return self.cosmo.encode(frames[:, 0])

    def _finalized_encode_all(self):
        """Causally encode the FULL growing prefix of downsampled frames seen so far and
        return per-chunk latents: (num_chunks, V*spatial, latent). Chunk c is bit-stable.

        For real frontends (WanFrontend/CosmoFrontend) this calls finalized_encode_multiview /
        finalized_encode -- literally the SAME code the OFFLINE store builder uses -- so the
        online consolidated chunks equal the offline encode-once chunks byte-for-byte. The
        buffer only ever indexes chunk c once it has been popped (drop-terminal), and the pop
        cadence matches finalized_encode's chunk count, so index c is always in range.

        Because it re-encodes the whole prefix on every finalization, consolidating a T-frame
        episode is O(T^2) VAE calls; acceptable for correctness (see buffer docstring / report).
        num_views==1 with a frontend exposing only encode() falls back to the single-view path."""
        frames = np.stack(self._all_ds_frames, 0)             # (t, V, h, w, 3)
        V = frames.shape[1]
        # Prefer the frontend's own causal drop-terminal methods when present so the online
        # path is literally the same code as the offline store builder.
        if V > 1 and hasattr(self.cosmo, "finalized_encode_multiview"):
            return self.cosmo.finalized_encode_multiview(frames)
        if hasattr(self.cosmo, "finalized_encode"):
            return self.cosmo.finalized_encode(frames[:, 0])
        # Backward-compat fallback for encoders exposing only encode()/encode_multiview()
        # (e.g. test fakes). encode() returns ALL complete chunks with no drop-terminal, so we
        # drop the terminal (provisional) chunk here to match the real frontends' finalized_encode
        # accounting (finalized count = complete_chunks - 1). This keeps the consolidation cadence
        # identical whether driven by a real frontend or a fake, eliminating the off-by-one that
        # caused the IndexError regression.
        raw = (self.cosmo.encode_multiview(frames) if (V > 1 and hasattr(self.cosmo, "encode_multiview"))
               else self.cosmo.encode(frames[:, 0]))
        return raw[:-1] if raw.shape[0] > 0 else raw

    def _log_timing(self, name, ms, **kw):
        try:
            extra = " ".join(f"{k}={v}" for k, v in kw.items())
            with open(self._timing_log, "a") as f:
                f.write(f"{name} {ms:.3f} {extra}\n")
        except Exception:
            pass

    def _encode_one_frame(self, frame_v):
        """OPTION B: encode a single downsampled high-resolution frame.

        frame_v: (V,h,w,3) uint8. Returns (V, visual_tokens, dim) float32 -- i.e. one
        row-block of the A-path (t,V,tokens,dim) tensor for this frame. Pooling/resize/
        normalization IDENTICAL to _encode_recent (the A path), just batched over t=1 so
        the cached gather reproduces A byte-for-byte."""
        emb = self._encode_visual_frames(frame_v[None, ...])
        return emb[0]                                         # (V, tokens, dim)

    def _get_dinov3_encoder(self):
        if self._dinov3_encoder is not None:
            return self._dinov3_encoder

        import torch  # noqa: PLC0415
        from transformers import AutoImageProcessor  # noqa: PLC0415
        from transformers import AutoModel  # noqa: PLC0415

        model_path = os.environ.get("DINOV3_MODEL_PATH", "")
        if not model_path:
            raise RuntimeError(
                "DINOV3_MODEL_PATH is required for online DINOv3 encoding"
            )
        cache_key = os.path.realpath(model_path)
        cached = _DINOV3_ENCODER_CACHE.get(cache_key)
        if cached is not None:
            self._dinov3_encoder = cached
            return cached
        processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16
        ).to("cuda").eval()
        patch_size = int(getattr(model.config, "patch_size", 0))
        register_tokens = int(getattr(model.config, "num_register_tokens", 0))
        if patch_size <= 0 or int(getattr(model.config, "hidden_size", 0)) != 768:
            raise ValueError("unexpected staged DINOv3 model geometry")

        def encode(frames):
            frames = np.asarray(frames)
            if (
                frames.ndim != 5
                or frames.shape[-1] != 3
                or frames.dtype != np.uint8
            ):
                raise ValueError(
                    "DINOv3 recent encoder expects uint8 (T,V,H,W,3), got "
                    f"{frames.shape} {frames.dtype}"
                )
            t, v = frames.shape[:2]
            flat = frames.reshape((-1, *frames.shape[2:]))
            resize = {"height": 256, "width": 256}
            cpu_inputs = processor(
                images=list(flat),
                return_tensors="pt",
                size=resize,
                crop_size=resize,
            )
            inputs = {key: value.to("cuda") for key, value in cpu_inputs.items()}
            image_h, image_w = inputs["pixel_values"].shape[-2:]
            if (int(image_h), int(image_w)) != (256, 256):
                raise ValueError(
                    f"DINOv3 processor produced {image_h}x{image_w}, expected 256x256"
                )
            if image_h % patch_size or image_w % patch_size:
                raise ValueError(
                    f"DINOv3 input {image_h}x{image_w} is incompatible with "
                    f"patch size {patch_size}"
                )
            with torch.inference_mode():
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state.float()
                grid_h, grid_w = image_h // patch_size, image_w // patch_size
                prefix = 1 + register_tokens
                expected_tokens = prefix + grid_h * grid_w
                if hidden.shape[1] != expected_tokens:
                    raise ValueError(
                        f"DINOv3 sequence length {hidden.shape[1]} != "
                        f"expected {expected_tokens}"
                    )
                patches = hidden[:, prefix:].reshape(
                    hidden.shape[0], grid_h, grid_w, hidden.shape[-1]
                )
                if grid_h % 8 or grid_w % 8:
                    raise ValueError(
                        f"DINOv3 patch grid {grid_h}x{grid_w} is not divisible by 8x8"
                    )
                pooled = patches.reshape(
                    hidden.shape[0],
                    8,
                    grid_h // 8,
                    8,
                    grid_w // 8,
                    hidden.shape[-1],
                ).mean(dim=(2, 4))
            encoded = (
                pooled.reshape(pooled.shape[0], 64, self._visual_dim)
                .cpu()
                .numpy()
                .reshape(t, v, 64, self._visual_dim)
            )
            if encoded.shape != (t, v, 64, self._visual_dim):
                raise AssertionError(
                    f"unexpected DINOv3 output shape: {encoded.shape}"
                )
            return encoded

        self._dinov3_encoder = encode
        _DINOV3_ENCODER_CACHE[cache_key] = encode
        return encode

    def warmup_visual_encoder(self):
        """Load DINOv3 once and execute the exact online preprocessing path."""
        if self._visual_stream != "dinov3":
            return {"visual_stream": self._visual_stream, "warmed": False}
        frames = np.zeros(
            (1, self.num_views, 256, 256, 3),
            dtype=np.uint8,
        )
        encoded = self._get_dinov3_encoder()(frames)
        return {
            "visual_stream": self._visual_stream,
            "warmed": True,
            "shape": list(encoded.shape),
        }

    def _encode_visual_frames(self, frames):
        if self._visual_stream == "dinov3":
            return self._get_dinov3_encoder()(frames)

        x = np.asarray(frames)
        num_views = x.shape[1]
        xj = x.astype(np.float32) / 255.0 * 2.0 - 1.0
        xj = einops.rearrange(xj, "t v h w c -> (t v) h w c")
        xj = image_tools.resize_with_pad(xj, 224, 224)
        xj = einops.rearrange(
            xj, "(t v) h w c -> t v h w c", v=num_views
        )
        emb = np.asarray(jax.device_get(self.siglip_fn(xj)))
        return np.asarray(jax.device_get(pool_tokens_to_size(emb, 64)))

    def _encode_recent_A(self):
        """OPTION A path: re-encode the ENTIRE recent window each call (original)."""
        if not self._recent_frames:
            return np.zeros((0, self._visual_dim), np.float32)
        # _recent_frames entries are (V,h,w,3)
        x = np.stack(self._recent_frames, 0)                  # (t, V, h, w, 3)
        emb = self._encode_visual_frames(x)
        return emb.reshape(-1, emb.shape[-1])                 # (t*V*siglip_tokens, dim)

    def _encode_recent_B(self):
        """OPTION B path: gather CACHED per-frame tokens for the current recent window
        (_rec_start: onward, keyed by absolute frame index) and concatenate -- NO re-encode."""
        _rec_start = len(self._consolidated) * self.chunk_frames
        toks = self._all_ds_frames_hi_tokens[_rec_start:]     # each (V, tokens, dim)
        if not toks:
            return np.zeros((0, self._visual_dim), np.float32)
        emb = np.stack(toks, 0)                               # (t, V, tokens, dim)
        return emb.reshape(-1, emb.shape[-1])                 # (t*V*tokens, dim) -- SAME order as A

    def _encode_recent(self):
        _t0 = time.perf_counter()
        if self._recent_cache:
            out = self._encode_recent_B()
            if self._recent_verify:
                a = self._encode_recent_A()
                self._log_verify(a, out)
        else:
            out = self._encode_recent_A()
        if self._timing_log is not None:
            self._log_timing("recent_encode", (time.perf_counter() - _t0) * 1000.0,
                             mode=("B" if self._recent_cache else "A"),
                             rows=out.shape[0], rec=len(self._recent_frames))
        return out

    def _log_verify(self, a, b):
        try:
            a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
            if a.shape != b.shape:
                msg = f"SHAPE_MISMATCH a={a.shape} b={b.shape}"
            else:
                mad = float(np.max(np.abs(a - b))) if a.size else 0.0
                fa = a.reshape(-1); fb = b.reshape(-1)
                denom = (np.linalg.norm(fa) * np.linalg.norm(fb))
                cos = float(fa @ fb / denom) if denom > 0 else 1.0
                msg = f"rows={a.shape[0]} max_abs_diff={mad:.3e} cosine={cos:.10f}"
            tgt = self._verify_log or self._timing_log
            if tgt is not None:
                with open(tgt, "a") as f:
                    f.write("VERIFY " + msg + "\n")
        except Exception:
            pass

    @staticmethod
    def _with_delta(arr):
        if arr.shape[0] == 0:
            return np.zeros((0, arr.shape[1] * 2), np.float32)
        delta = np.zeros_like(arr)
        delta[1:] = arr[1:] - arr[:-1]
        return np.concatenate([arr, delta], axis=1).astype(np.float32)

    @staticmethod
    def _geometric_actions(arr):
        """Reframe robocasa 12-dim actions into Cosmos-3-style 15-dim geometric vectors.

        Input layout per step: [eef_pos(3), eef_rot_rotvec(3), gripper(1), base(4), mode(1)].
        Output layout per step: [eef_pos_delta(3), eef_rot_6d(6), gripper_abs(1), base(4), mode(1)].
        No separate Δ concat: the relative pose already encodes motion (Cosmos-3 scheme).
        """
        if arr.shape[0] == 0:
            return np.zeros((0, 15), np.float32)
        out = np.zeros((arr.shape[0], 15), np.float32)
        for i in range(arr.shape[0]):
            a = arr[i]
            out[i, 0:3] = a[0:3]                       # eef_pos delta passthrough
            out[i, 3:9] = rotvec_to_6d(a[3:6])         # rotvec -> 6D rotation
            out[i, 9] = a[6]                           # gripper absolute
            out[i, 10:14] = a[7:11]                    # base_motion passthrough
            out[i, 14] = a[11]                         # control_mode passthrough
        return out

    def _pool_latents(self, chunks):
        """Pool each finalized chunk's Wan latents to (pool_h x pool_w) per view, EXACTLY as
        training dataset._gather_cosmo_bottleneck. GRID-AGNOSTIC:
          * RoboDojo: NON-square 12x16 grid (wan 192x256 // stride 16) = 192 tok/view, 3 views ->
            V*192. pool_h=pool_w=2 -> block-mean 12x16 -> 2x2 = 4 tok/view = 12 tok/chunk.
          * robocasa: square 8x8=64 tok/view, pool_hw=2 -> 4 tok/view (byte-identical to old path
            since gh=gw=8, ph=pw=pool_hw).
        chunks: list of (V*spatial, d). Returns (num_chunks*V*ph*pw, d) flattened."""
        ph = self.pool_h or self.pool_hw
        pw = self.pool_w or self.pool_hw
        if not chunks or not (ph and pw):
            return (np.concatenate(chunks, 0) if chunks
                    else np.zeros((0, self.latent_dim), np.float32))
        lat = np.stack(chunks, 0)                       # (nc, V*spatial, d)
        nc = lat.shape[0]; d = lat.shape[-1]
        Vv = int(self.num_views) if self.num_views else (lat.shape[1] // 64)
        spatial = lat.shape[1] // Vv
        assert lat.shape[1] == Vv * spatial, f"latent width {lat.shape[1]} not V({Vv})*spatial"
        if self.wan_h and self.wan_w:
            gh, gw = self.wan_h // 16, self.wan_w // 16      # RoboDojo 192x256 -> 12x16
        else:
            gh = gw = int(round(spatial ** 0.5))            # square fallback (robocasa 8x8=64)
        assert gh * gw == spatial, f"grid {gh}x{gw} != spatial {spatial}"
        assert gh % ph == 0 and gw % pw == 0, f"pool {ph}x{pw} must divide grid {gh}x{gw}"
        bh, bw = gh // ph, gw // pw
        gg = lat.reshape(nc, Vv, gh, gw, d)
        gg = gg.reshape(nc, Vv, ph, bh, pw, bw, d).mean(axis=(3, 5))
        return gg.reshape(nc * Vv * ph * pw, d).astype(np.float32)

    def get_history_feats(self, step_idx):
        cosmo = self._pool_latents(self._consolidated)
        recent = self._encode_recent()
        states = np.stack(self._states, 0) if self._states else np.zeros((0, self.state_dim), np.float32)
        actions = np.stack(self._actions, 0) if self._actions else np.zeros((0, self.action_dim), np.float32)
        if self.action_encoding == "geometric":
            action_emb = self._geometric_actions(actions)
        elif self.action_encoding == "raw":
            action_emb = actions.astype(np.float32)          # action only, no delta (12d)
        else:
            action_emb = self._with_delta(actions)           # raw_delta (24d)
        return {
            "cosmo_latent_emb": cosmo.astype(np.float32),
            "cosmo_recent_emb": recent.astype(np.float32),
            "cosmo_mem_mask": np.ones((cosmo.shape[0],), bool),
            "cosmo_proprio_emb": (states.astype(np.float32) if self.proprio_encoding == "raw"
                                  else self._with_delta(states)),
            "cosmo_action_emb": action_emb,
            "cosmo_lang_emb": self._lang_emb,
            "cosmo_finalized_steps": np.int32(
                len(self._consolidated) * self.chunk_frames
            ),
        }

    def drain_delta_updates(self):
        """Return each newly finalized chunk with its aligned encoded trajectories."""
        if not self._delta_pending_chunk_indices:
            return []

        states = (
            np.stack(self._states, 0)
            if self._states
            else np.zeros((0, self.state_dim), np.float32)
        )
        actions = (
            np.stack(self._actions, 0)
            if self._actions
            else np.zeros((0, self.action_dim), np.float32)
        )
        if self.action_encoding == "geometric":
            action_emb = self._geometric_actions(actions)
        elif self.action_encoding == "raw":
            action_emb = actions.astype(np.float32)
        else:
            action_emb = self._with_delta(actions)
        proprio_emb = (
            states.astype(np.float32)
            if self.proprio_encoding == "raw"
            else self._with_delta(states)
        )

        updates = []
        for chunk_index in self._delta_pending_chunk_indices:
            start = chunk_index * self.chunk_frames
            end = start + self.chunk_frames
            if end > len(action_emb) or end > len(proprio_emb):
                raise RuntimeError(
                    "finalized chunk is not aligned with action/proprio history: "
                    f"chunk={chunk_index} range=[{start},{end}) "
                    f"actions={len(action_emb)} proprio={len(proprio_emb)}"
                )
            updates.append(
                {
                    "cosmo_chunk": self._pool_latents(
                        [self._consolidated[chunk_index]]
                    ),
                    "action_chunk": np.ascontiguousarray(
                        action_emb[start:end]
                    ).astype(np.float32),
                    "proprio_chunk": np.ascontiguousarray(
                        proprio_emb[start:end]
                    ).astype(np.float32),
                    "chunk_index": np.int32(chunk_index),
                }
            )
        self._delta_pending_chunk_indices.clear()
        return updates
