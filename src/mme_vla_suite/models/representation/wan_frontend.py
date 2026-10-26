"""Frozen, causal Wan2.2 video VAE tokenizer wrapper (PyTorch). No JAX imports here.

Thin wrapper around the Wan2.2-TI2V-5B video VAE (``diffusers.AutoencoderKLWan``)
that turns a clip of robot camera frames into continuous latent tokens for the
downstream JAX bottleneck. The VAE is fully frozen (no grad) and run under
``torch.no_grad()``. Mirrors the ``CosmoFrontend`` interface so the two frontends
are config-swappable.

Discovered / measured properties (Wan2.2 VAE, empirically verified on B200):
  * ``diffusers.AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    subfolder="vae")`` -- Apache-2.0, ~2.8GB fp32 safetensors (VAE subfolder only).
  * Architecture uses ``WanCausalConv3d`` -> strictly TEMPORALLY CAUSAL.
  * ``encode(x)`` takes ``x`` of layout (B, 3, T, H, W) in range [-1, 1] and
    returns a DiagonalGaussianDistribution; we take ``.mode()`` (the mean) for a
    deterministic latent of layout (B, 48, t', h', w').
  * Temporal factor 4: for a clip of T frames the encoder emits
    ``t' = 1 + (T - 1) // 4`` temporal latents. Every latent (including the
    leading one) is a genuine, causally-stable chunk -- there is NO separate
    'warm-up' latent to drop (unlike Cosmos). For T a multiple of 4 this gives
    ``num_chunks == T // temporal_factor`` complete chunks.
  * 16x spatial: at 224px h'=w'=14 (196 tokens); at 128px h'=w'=8 (64 tokens).
  * latent_dim (z_dim) = 48. Continuous latents. fully deterministic in fp32.
  * UNLIKE Cosmos: the terminal latent is NOT provisional -- appending future
    frames leaves ALL prior latents (incl. the last) bit-identical (measured
    max_abs_diff 0.0 across clips of 5..40 frames). encode-once-and-slice is
    therefore exactly valid. We still expose finalized_encode == encode[:-1] to
    match the CosmoFrontend drop-terminal contract used by the offline store /
    online buffer, but note the dropped terminal chunk here is already finalized.
"""
from __future__ import annotations

import numpy as np
import torch

from mme_vla_suite.shared.wan_alignment import NATIVE
from mme_vla_suite.shared.wan_alignment import REPLICATE_FIRST_FRAME
from mme_vla_suite.shared.wan_alignment import WAN_CHUNK_ALIGNMENTS
from mme_vla_suite.shared.wan_alignment import resolve_wan_chunk_alignment

_HF_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
_HF_REVISION = "b8fff7315c768468a5333511427288870b2e9635"
_TEMPORAL_FACTOR = 4


class WanFrontend:
    """Frozen causal Wan2.2 video VAE encoder: (T,H,W,3) uint8 -> latent tokens."""

    chunk_alignments = WAN_CHUNK_ALIGNMENTS

    def __init__(
        self,
        variant: str = "Wan2.2-TI2V-5B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,   # bf16 default: 1.32x faster, ~1e-2 noise (accepted)
        compile_encode: bool = False,          # torch.compile the VAE encode: +1.7x on top of bf16
    ) -> None:
        self.variant = variant
        self.temporal_factor = _TEMPORAL_FACTOR
        self.device = device
        self.dtype = dtype

        from diffusers import AutoencoderKLWan

        self._model = AutoencoderKLWan.from_pretrained(
            _HF_REPO,
            subfolder="vae",
            revision=_HF_REVISION,
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        if compile_encode:
            # VAE _encode has a dynamic chunk loop + feat_cache -> fullgraph=False.
            self._model.encode = torch.compile(self._model.encode, mode="reduce-overhead", fullgraph=False)
        self.latent_dim = int(self._model.config.z_dim)  # 48

    @torch.no_grad()
    def encode(self, frames_uint8: np.ndarray, alignment: str = NATIVE) -> np.ndarray:
        """(T,H,W,3) uint8 -> (num_chunks, spatial_tokens, latent_dim) float32.

        native: 1 + (T-1)//temporal_factor latents; latent 0 holds frame 0 only and latent
        c >= 1 holds frames 4c-3..4c. replicate_first_frame: one copy of frame 0 is encoded
        first and its latent is dropped, so T//temporal_factor latents remain and latent c
        holds frames 4c..4c+3 (see mme_vla_suite.shared.wan_alignment).
        """
        alignment = resolve_wan_chunk_alignment(alignment)
        frames = np.ascontiguousarray(np.asarray(frames_uint8))
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected frames (T,H,W,3), got {frames.shape}")
        if alignment == REPLICATE_FIRST_FRAME:
            if frames.shape[0] == 0:
                raise ValueError("cannot encode an empty clip")
            frames = np.ascontiguousarray(np.concatenate([frames[:1], frames], axis=0))

        x = torch.from_numpy(frames).to(self.device, self.dtype)
        x = x.div(255.0).mul_(2.0).sub_(1.0)
        x = x.permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,H,W)

        out = self._model.encode(x)
        latent = out.latent_dist.mode()  # (1,C,t',h',w') deterministic mean

        # The Wan2.2 VAE emits t' = 1 + (T-1)//temporal_factor temporal latents,
        # every one of which is a genuine, causally-stable chunk (verified
        # max_abs_diff == 0.0 across growing prefixes). There is NO separate
        # warm-up latent to drop here (unlike Cosmos): the leading latent IS the
        # first real chunk. For clips whose length is a multiple of
        # temporal_factor this yields num_chunks == T // temporal_factor.

        latent = latent[0]  # (C, nc, h', w')
        c, nc, h, w = latent.shape
        latent = latent.permute(1, 2, 3, 0).reshape(nc, h * w, c)  # (nc, hw, C)
        if alignment == REPLICATE_FIRST_FRAME:
            latent = latent[1:]  # the replicated frame's own leading latent
        return latent.float().cpu().numpy()

    @torch.no_grad()
    def finalized_encode(self, frames_uint8: np.ndarray) -> np.ndarray:
        """encode() minus the terminal chunk, matching the CosmoFrontend contract.

        NOTE: for the Wan2.2 VAE the terminal chunk is already finalized (causal,
        stable under future appends). We still drop it so this frontend is a
        drop-in replacement for CosmoFrontend under the offline-store / online-
        buffer drop-terminal protocol.
        """
        lat = self.encode(frames_uint8)
        return lat[:-1] if lat.shape[0] > 0 else lat

    def encode_multiview(self, frames_v, alignment: str = NATIVE):
        """frames_v: (T, V, H, W, 3) uint8 -> (num_chunks, V*spatial_tokens, latent_dim) float32.
        Each view encoded independently (frozen per-view tokenizer), latents concatenated on the
        token axis. Chunk count is identical across views (same T), so alignment is exact."""
        assert frames_v.ndim == 5 and frames_v.shape[-1] == 3, f"expected (T,V,H,W,3), got {frames_v.shape}"
        V = frames_v.shape[1]
        per_view = [
            self.encode(np.ascontiguousarray(frames_v[:, v]), alignment=alignment)
            for v in range(V)
        ]
        nc = per_view[0].shape[0]
        assert all(p.shape[0] == nc for p in per_view), "view chunk counts differ"
        return np.concatenate(per_view, axis=1)

    def finalized_encode_multiview(self, frames_v):
        """Like encode_multiview but drops each view's provisional terminal chunk BEFORE concat,
        so finalized chunk counts stay aligned across views."""
        assert frames_v.ndim == 5 and frames_v.shape[-1] == 3, f"expected (T,V,H,W,3), got {frames_v.shape}"
        V = frames_v.shape[1]
        per_view = [self.finalized_encode(np.ascontiguousarray(frames_v[:, v])) for v in range(V)]
        nc = per_view[0].shape[0]
        assert all(p.shape[0] == nc for p in per_view), "view chunk counts differ"
        return np.concatenate(per_view, axis=1) if nc > 0 else per_view[0]

    @torch.no_grad()
    def finalized_encode_multiview_batched(self, frames_gv, pool_hw=None, resize_to=None):
        """BATCHED finalized encode over a minibatch of EQUAL-length clips.

        frames_gv: (G, T, V, H, W, 3) uint8 -> (G, num_chunks, V*spatial, latent) float32.
        All G items share T (caller buckets by length), so one VAE forward handles G*V views
        at once (VAE encode supports a leading batch dim), amortizing GPU launch overhead vs
        G separate calls. Drops each view's provisional terminal chunk (== finalized_encode).

        resize_to: if set (e.g. 128), the frames are uploaded at their native res and
        downsampled ON GPU (F.interpolate area) -- avoids the CPU cv2 double-loop and the
        larger CPU->GPU transfer. pool_hw: adaptive-avg-pool each chunk grid to pool_hw^2.
        """
        frames_gv = np.asarray(frames_gv)
        assert frames_gv.ndim == 6 and frames_gv.shape[-1] == 3, f"expected (G,T,V,H,W,3), got {frames_gv.shape}"
        G, T, V, H, W, _ = frames_gv.shape
        # upload uint8, cast, normalize on GPU; (G,T,V,H,W,3) -> (G*V,3,T,H,W)
        x = torch.from_numpy(np.ascontiguousarray(frames_gv)).to(self.device)
        x = x.to(self.dtype).div(255.0).mul_(2.0).sub_(1.0)
        x = x.permute(0, 2, 5, 1, 3, 4).reshape(G * V, 3, T, H, W)   # (G*V,3,T,H,W)
        if resize_to and (H != resize_to or W != resize_to):
            # GPU downsample (area interp) fused before the VAE, instead of CPU cv2.
            gv3 = x.shape[0]
            xr = x.permute(0, 2, 1, 3, 4).reshape(gv3 * T, 3, H, W)  # (gv*T,3,H,W)
            xr = torch.nn.functional.interpolate(xr, size=(resize_to, resize_to), mode="area")
            x = xr.reshape(gv3, T, 3, resize_to, resize_to).permute(0, 2, 1, 3, 4)
        out = self._model.encode(x)
        lat = out.latent_dist.mode()                                # (G*V, C, nc, h', w')
        gv, C, nc, h, w = lat.shape
        if pool_hw and (h > pool_hw or w > pool_hw):
            # adaptive avg-pool the (h,w) grid of every (view, chunk) to pool_hw x pool_hw.
            lp = lat.permute(0, 2, 1, 3, 4).reshape(gv * nc, C, h, w)   # (gv*nc, C, h, w)
            lp = torch.nn.functional.adaptive_avg_pool2d(lp, (pool_hw, pool_hw))
            h = w = pool_hw
            lat = lp.reshape(gv, nc, C, h, w).permute(0, 2, 1, 3, 4)    # (gv, C, nc, h, w)
        lat = lat.permute(0, 2, 3, 4, 1).reshape(G, V, nc, h * w, C)  # (G,V,nc,hw,C)
        lat = lat[:, :, :-1] if nc > 0 else lat                     # drop terminal chunk
        nc2 = lat.shape[2]
        # concat views on token axis: (G, nc, V*hw, C)
        lat = lat.permute(0, 2, 1, 3, 4).reshape(G, nc2, V * h * w, C)
        return lat.float().cpu().numpy()

    @torch.no_grad()
    def streaming_encode(self, frames_uint8, chunk_frames=_TEMPORAL_FACTOR):
        """Incrementally encode FINALIZED chunks, mirroring the online buffer.

        Because the Wan2.2 VAE is strictly causal (every chunk is stable under
        future appends), independently or incrementally encoding growing prefixes
        reproduces the full pass exactly. Output length equals finalized_encode.
        """
        assert chunk_frames == self.temporal_factor
        frames = np.asarray(frames_uint8)
        total = frames.shape[0]
        # encode() emits 1 + (T - 1) // temporal_factor latents: latent 0 holds
        # frame 0 only. finalized_encode drops the terminal latent on top of that.
        num_full = self.encode(frames).shape[0]
        num_finalized = num_full - 1
        if num_finalized <= 0:
            return np.empty((0, 0, 0), dtype=np.float32)

        committed: list[np.ndarray | None] = [None] * num_finalized
        # Grow the prefix one temporal chunk at a time and commit each chunk once
        # it is no longer terminal. (For the Wan2.2 VAE every chunk is already
        # stable, so this equals encode-once-and-slice; kept for parity with the
        # CosmoFrontend online-buffer protocol.)
        n_prefixes = total // chunk_frames
        for k in range(n_prefixes + 1):
            prefix_len = (k + 1) * chunk_frames
            if prefix_len > total:
                break
            lat = self.encode(frames[:prefix_len])
            ncur = lat.shape[0]
            for c in range(ncur - 1):
                if c < num_finalized and committed[c] is None:
                    committed[c] = lat[c]
        return np.stack([c for c in committed], axis=0)

    def parameters(self):
        """Yield the underlying frozen VAE's parameters."""
        return self._model.parameters()

    def make_streamer(self, num_views: int = 1, alignment: str = NATIVE):
        """Create a WanStreamingEncoder sharing this frozen VAE (O(T) inference cache)."""
        return WanStreamingEncoder(self, num_views=num_views, alignment=alignment)


class WanStreamingEncoder:
    """O(T) causal streaming encoder for INFERENCE (replaces O(T^2) full-prefix re-encode).

    Wan's VAE (`_encode`) already streams internally: it feeds [frame0],[f1:5],[f5:9],...
    through `encoder(group, feat_cache=_enc_feat_map, feat_idx=_enc_conv_idx)` where the
    WanCausalConv3d layers carry causal temporal state in `_enc_feat_map`. Stock code clears
    that state every call. We PERSIST it across control steps and feed only the NEW group,
    so each finalization encodes ONE group (~9ms/view) instead of the whole prefix.

    VERIFIED byte-exact (max_abs_diff 0.0, fp32) vs the full-clip encode -- because we use the
    VAE's own causal-conv cache, and quant_conv is a 1x1 pointwise conv (per-chunk == full).

    One streamer holds V INDEPENDENT view streams. Feed downsampled frames as they arrive via
    push(); it returns any newly-finalized chunk latents. NEVER clears mid-episode; reset()
    on episode boundary.
    """

    def __init__(self, frontend: "WanFrontend", num_views: int = 1, alignment: str = NATIVE):
        self.fe = frontend
        self.alignment = resolve_wan_chunk_alignment(alignment)
        self.model = frontend._model
        self.device = frontend.device
        self.dtype = frontend.dtype
        self.latent_dim = frontend.latent_dim
        self.tf = frontend.temporal_factor
        self.num_views = int(num_views)
        self.reset()

    def reset(self):
        # per-view persisted causal state + pending (not-yet-grouped) frames + group counter.
        self._feat_map = [None] * self.num_views     # each: the VAE's _enc_feat_map list
        self._pending = [[] for _ in range(self.num_views)]   # list of (H,W,3) uint8 per view
        self._iter = [0] * self.num_views            # groups fed so far (0 -> next is frame0)
        self._needs_leading_copy = self.alignment == REPLICATE_FIRST_FRAME
        try:
            self.model.clear_cache()
        except Exception:
            pass

    def _frames_to_tensor(self, frames):
        # frames: list of (H,W,3) uint8 -> (1,3,t,H,W) in [-1,1]
        x = np.stack(frames, 0)                                # (t,H,W,3)
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device, self.dtype)
        t = t.div(255.0).mul_(2.0).sub_(1.0)
        return t.permute(3, 0, 1, 2).unsqueeze(0)              # (1,3,t,H,W)

    @torch.no_grad()
    def _encode_group_view(self, v, group_frames):
        """Feed ONE group (1 frame for group 0, else `tf` frames) through the VAE encoder,
        carrying view v's persisted causal state. Returns the group's chunk latent
        (spatial_tokens, latent_dim) float32, or None for the leading warm-up-only step."""
        from diffusers.models.autoencoders.autoencoder_kl_wan import patchify
        m = self.model
        x = self._frames_to_tensor(group_frames)               # (1,3,g,H,W)
        if m.config.patch_size is not None:
            x = patchify(x, patch_size=m.config.patch_size)
        # install this view's persisted feat_map; conv_idx resets per group (per _encode loop).
        if self._feat_map[v] is None:
            m.clear_cache(); self._feat_map[v] = m._enc_feat_map
        m._enc_feat_map = self._feat_map[v]
        m._enc_conv_idx = [0]
        out = m.encoder(x, feat_cache=m._enc_feat_map, feat_idx=m._enc_conv_idx)
        self._feat_map[v] = m._enc_feat_map                    # persist updated state
        enc = m.quant_conv(out)                                # (1, 2*z, g', h', w') mean+logvar
        # take the MODE (mean) == first z_dim channels, matching WanFrontend.encode's
        # latent_dist.mode(). quant_conv is 1x1 pointwise so per-group == full-clip.
        enc = enc[:, : self.latent_dim]                        # (1, z, g', h', w')
        enc = enc[0]                                           # (z, g', h', w')
        c, gp, h, w = enc.shape
        # one temporal latent per group; take it -> (h*w, z)
        lat = enc.permute(1, 2, 3, 0).reshape(gp, h * w, c)[-1]  # (h*w, z)
        return lat.float().cpu().numpy()

    @torch.no_grad()
    def push(self, new_frames_v):
        """new_frames_v: (n, V, H, W, 3) uint8 (downsampled frames, strict time order).
        Returns list of newly-finalized chunk latents, each (V*spatial, latent_dim), in order.
        A chunk finalizes when a view has a complete group (group0=1 frame, groupN=tf frames)."""
        new_frames_v = np.asarray(new_frames_v)
        n = new_frames_v.shape[0]
        for v in range(self.num_views):
            for i in range(n):
                self._pending[v].append(new_frames_v[i, v])
        if self._needs_leading_copy and n > 0:
            # replicate_first_frame: Wan's one-frame leading group is a copy of frame 0, so
            # every emitted chunk holds four kept frames. Its latent is dropped below.
            for v in range(self.num_views):
                self._pending[v].insert(0, np.array(self._pending[v][0], copy=True))
            self._needs_leading_copy = False
        # emit as many complete groups as are available (all views advance in lockstep since
        # they share the same frame timeline -> identical group availability).
        new_chunks = []
        while True:
            need = 1 if self._iter[0] == 0 else self.tf
            if len(self._pending[0]) < need:
                break
            leading_copy = self.alignment == REPLICATE_FIRST_FRAME and self._iter[0] == 0
            per_view_lat = []
            for v in range(self.num_views):
                grp = self._pending[v][:need]
                self._pending[v] = self._pending[v][need:]
                per_view_lat.append(self._encode_group_view(v, grp))
                self._iter[v] += 1
            if leading_copy:
                continue  # causal state is kept; the replicated frame's latent is dropped
            new_chunks.append(np.concatenate(per_view_lat, axis=0))  # (V*spatial, latent)
        return new_chunks
