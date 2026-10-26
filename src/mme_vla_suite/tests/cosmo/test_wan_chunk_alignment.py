"""Wan2.2 chunk alignment between kept memory frames and consolidated latents.

The fake VAE below keeps the real AutoencoderKLWan contract that the frontend, the streamers
and the buffer rely on: ``_encode`` feeds frame 0 alone, then groups of four frames, with a
causal cache carried in ``_enc_feat_map``. Each latent records the first and last frame ID of
its group, the number of frames the causal cache has seen, and the group size, so a test can
read back which kept frames each latent holds. The real WanFrontend / WanStreamingEncoder /
BatchedWanStreamingEncoder / CosmoMemoryBuffer code runs on top of it.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mme_vla_suite.models.representation import wan_frontend as wf
from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
from mme_vla_suite.shared.wan_alignment import ALIGNMENT_RECORD
from mme_vla_suite.shared.wan_alignment import NATIVE
from mme_vla_suite.shared.wan_alignment import REPLICATE_FIRST_FRAME
from mme_vla_suite.shared.wan_alignment import STAGE1_METADATA_KEY
from mme_vla_suite.shared.wan_alignment import expected_stored_chunks
from mme_vla_suite.shared.wan_alignment import latent_frame_ranges
from mme_vla_suite.shared.wan_alignment import read_alignment_record
from mme_vla_suite.shared.wan_alignment import record_training_alignment
from mme_vla_suite.shared.wan_alignment import require_feature_store_alignment
from mme_vla_suite.shared.wan_alignment import require_store_alignment
from mme_vla_suite.shared.wan_alignment import write_alignment_record

ROOT = Path(__file__).resolve().parents[4]
ADAPTER_DIR = ROOT / "benchmarks/rmbench/eval/runtime_overlay/adapter"
VIEW_OFFSET = 70   # image value = frame id + VIEW_OFFSET * view (uint8, ids stay < 70)
HW = 32            # 2x2 Wan grid (16x spatial factor)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeWanVAE:
    """Wan grouping + causal cache. Latent channels: [first id, last id, frames seen, group size]."""

    def __init__(self):
        self.config = SimpleNamespace(patch_size=None, z_dim=4)
        self.clear_cache()

    def clear_cache(self):
        self._enc_feat_map = [None]
        self._enc_conv_idx = [0]

    def encoder(self, x, feat_cache=None, feat_idx=None):
        batch, _, group, height, width = x.shape
        ids = torch.round((x[:, 0, :, 0, 0] + 1.0) / 2.0 * 255.0)      # (B, group)
        seen = feat_cache[0] if feat_cache[0] is not None else torch.zeros(batch)
        seen = seen + group
        feat_cache[0] = seen                                             # causal state
        feat_idx[0] += 1
        feats = torch.stack(
            [ids[:, 0], ids[:, -1], seen, torch.full((batch,), float(group))], dim=1
        )
        mean = feats[:, :, None, None, None].expand(batch, 4, 1, height // 16, width // 16)
        return torch.cat([mean, torch.zeros_like(mean)], dim=1)          # mean + logvar

    def quant_conv(self, out):
        return out

    def encode(self, x):
        # Same loop as diffusers AutoencoderKLWan._encode.
        self.clear_cache()
        iters = 1 + (x.shape[2] - 1) // 4
        outs = []
        for i in range(iters):
            self._enc_conv_idx = [0]
            group = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1): 1 + 4 * i]
            outs.append(self.encoder(group, feat_cache=self._enc_feat_map,
                                     feat_idx=self._enc_conv_idx))
        mean = self.quant_conv(torch.cat(outs, dim=2))[:, : self.config.z_dim]
        self.clear_cache()
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: mean))


def _frontend():
    fe = wf.WanFrontend.__new__(wf.WanFrontend)
    fe.variant = "fake-wan"
    fe.temporal_factor = 4
    fe.device = "cpu"
    fe.dtype = torch.float32
    fe._model = _FakeWanVAE()  # noqa: SLF001
    fe.latent_dim = 4
    return fe


def _clip(frame_ids, num_views):
    ids = np.asarray(frame_ids, np.int64)
    out = np.zeros((len(ids), num_views, HW, HW, 3), np.uint8)
    for v in range(num_views):
        out[:, v] = (ids + VIEW_OFFSET * v)[:, None, None, None]
    return out


def _decode(latent, num_views, view=0):
    """(V*spatial, 4) chunk latent -> (first id, last id, frames seen, group size) of one view."""
    rows = np.asarray(latent).reshape(num_views, -1, 4)[view]
    assert np.allclose(rows, rows[0])
    first, last, seen, group = rows[0]
    return int(first) - VIEW_OFFSET * view, int(last) - VIEW_OFFSET * view, int(seen), int(group)


# --------------------------------------------------------------------------- frontend


def test_native_latent_c_holds_kept_frames_4c_minus_3_to_4c():
    lat = _frontend().encode_multiview(_clip(range(12), 2))
    assert [_decode(c, 2)[:2] for c in lat] == [(0, 0), (1, 4), (5, 8)]
    assert [_decode(c, 2, view=1)[:2] for c in lat] == [(0, 0), (1, 4), (5, 8)]
    assert [(r.start, r.stop - 1) for r in latent_frame_ranges(3, 4, NATIVE)] == [(0, 0), (1, 4), (5, 8)]


@pytest.mark.parametrize("num_frames", [3, 4, 7, 12, 13, 21])
def test_replicate_first_frame_latent_c_holds_kept_frames_4c_to_4c_plus_3(num_frames):
    lat = _frontend().encode_multiview(_clip(range(num_frames), 2), alignment=REPLICATE_FIRST_FRAME)
    assert lat.shape[0] == num_frames // 4 == expected_stored_chunks(num_frames, 4, REPLICATE_FIRST_FRAME)
    got = [_decode(c, 2)[:2] for c in lat]
    want = [(r.start, r.stop - 1) for r in latent_frame_ranges(len(lat), 4, REPLICATE_FIRST_FRAME)]
    assert got == want == [(4 * c, 4 * c + 3) for c in range(num_frames // 4)]
    # every latent is a full group of four kept frames on top of the dropped frame-0 copy
    assert [_decode(c, 2)[2:] for c in lat] == [(4 * c + 5, 4) for c in range(len(lat))]


def test_unknown_alignment_is_rejected():
    with pytest.raises(ValueError, match="unknown wan_chunk_alignment"):
        _frontend().encode(_clip(range(5), 1)[:, 0], alignment="shifted")


# --------------------------------------------------------------------------- streamers


def _push_irregular(streamer, clip):
    out, i = [], 0
    for size in (1, 2, 5, 1, 3, 4, 7, 2, 100):
        if i >= len(clip):
            break
        out.extend(streamer.push(clip[i: i + size]))
        i += size
    return out


@pytest.mark.parametrize("alignment", [NATIVE, REPLICATE_FIRST_FRAME])
def test_streamers_match_full_encode(alignment):
    gdo = _load_module(
        "gated_delta_eval_optimizations_under_test",
        ROOT / "benchmarks/rmbench/eval/runtime_overlay/adapter/gated_delta_eval_optimizations.py",
    )
    fe = _frontend()
    clip = _clip(range(23), 3)
    full = fe.encode_multiview(clip, alignment=alignment)
    for streamer in (fe.make_streamer(3, alignment=alignment),
                     gdo.BatchedWanStreamingEncoder(fe, num_views=3, alignment=alignment)):
        chunks = _push_irregular(streamer, clip)
        assert len(chunks) == len(full)
        np.testing.assert_array_equal(np.stack(chunks), full)
        streamer.reset()
        again = _push_irregular(streamer, clip)
        np.testing.assert_array_equal(np.stack(again), full)


# --------------------------------------------------------------------------- online buffer


def _run_buffer(monkeypatch, alignment, steps):
    monkeypatch.setenv("EVAL_CONSOLIDATE_ALL", "1")     # benchmarks/rmbench/eval/run_local.py
    monkeypatch.setenv("EVAL_VISUAL_PROFILE", "siglip")
    buf = CosmoMemoryBuffer(
        cosmo=_frontend(), siglip_fn=lambda x: np.zeros((x.shape[0], x.shape[1], 1, 8), np.float32),
        chunk_frames=4, downsample=2, num_views=2, state_dim=16, action_dim=12, latent_dim=4,
        action_encoding="raw", proprio_encoding="raw", wan_chunk_alignment=alignment,
    )
    buf.set_language(None)
    report = []
    for step in range(steps):   # one env step per call, as the RMBench adapter feeds it
        state = np.zeros((1, 16), np.float32)
        state[0, 0] = step
        buf.add_buffer(images=_clip([step], 2), states=state,
                       actions=np.zeros((1, 12), np.float32), step_idx_list=[step])
        kept = list(range(0, step + 1, 2))
        latent_steps = set()
        for chunk in buf._consolidated:  # noqa: SLF001
            first, last, _, _ = _decode(chunk, 2)
            latent_steps.update(range(first, last + 1, 2))
        recent_steps = [int(f[0, 0, 0, 0]) for f in buf._recent_frames]  # noqa: SLF001
        events = [
            (sorted(range(_decode(u["cosmo_chunk"], 2)[0], _decode(u["cosmo_chunk"], 2)[1] + 1, 2)),
             [int(s) for s in u["proprio_chunk"][:, 0]])
            for u in buf.drain_delta_updates()
        ]
        report.append({"step": step, "kept": kept, "latent": latent_steps,
                       "recent": recent_steps, "events": events})
    return buf, report


def test_native_buffer_leaves_kept_frames_out_of_both_visual_streams(monkeypatch):
    _, report = _run_buffer(monkeypatch, NATIVE, steps=41)
    last = report[-1]   # 21 kept frames (steps 0..40), 5 chunks, recent window from kept frame 20
    missing = sorted(set(last["kept"]) - last["latent"] - set(last["recent"]))
    assert missing == [34, 36, 38]          # kept frames 17..19
    video, rows = [e for r in report for e in r["events"]][1]
    assert (video, rows) == ([2, 4, 6, 8], [8, 10, 12, 14])   # event c=1 pairs other frames


def test_aligned_buffer_puts_every_kept_frame_in_one_stream_and_pairs_events(monkeypatch):
    buf, report = _run_buffer(monkeypatch, REPLICATE_FIRST_FRAME, steps=41)
    assert buf.wan_chunk_alignment == REPLICATE_FIRST_FRAME
    for rec in report:
        assert not (rec["latent"] & set(rec["recent"])), rec["step"]
        assert sorted(rec["latent"] | set(rec["recent"])) == rec["kept"], rec["step"]
    events = [e for r in report for e in r["events"]]
    assert len(events) == 5
    for c, (video, rows) in enumerate(events):
        assert video == rows == [8 * c + 2 * k for k in range(4)]


def test_buffer_refuses_alignment_the_frontend_cannot_encode(monkeypatch):
    monkeypatch.setenv("EVAL_VISUAL_PROFILE", "siglip")

    class _NoAlignmentFrontend:
        def encode(self, frames):
            raise AssertionError("unused")

    with pytest.raises(ValueError, match="needs a frontend"):
        CosmoMemoryBuffer(cosmo=_NoAlignmentFrontend(), siglip_fn=None, chunk_frames=4,
                          wan_chunk_alignment=REPLICATE_FIRST_FRAME)
    with pytest.raises(ValueError, match="unknown wan_chunk_alignment"):
        CosmoMemoryBuffer(cosmo=_frontend(), siglip_fn=None, chunk_frames=4,
                          wan_chunk_alignment="shifted")


# --------------------------------------------------------------------------- stores


VARIANTS = {
    0: list(range(0, 21, 2)),
    1: [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19],
    2: [0, 2, 3, 4, 7, 9, 12, 14, 16, 18, 20],
    3: list(range(0, 21, 3)),
    4: [0, 2, 5, 6, 9, 10, 13, 14, 17, 18],
    5: list(range(21)),
    6: [0, 4, 8, 12, 16, 20],
    7: [0, 1, 2],
}


def _write_native_store(root: Path) -> Path:
    ep = root / "features" / "episode_0"
    ep.mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta" / "dino_store_manifest.json").write_text(json.dumps({
        "episodes": 1,
        "extractor": {"model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                      "feature_dim": 768, "tokens_per_view": 64},
        "replacement_streams": {"recent": {"file": "dinov3.npy", "shape": ["T", "V", 64, 768]},
                                "alignment": {"file": "dinov3.npy", "shape": ["T", "V*P", 768]}},
    }))
    num_frames = 21
    np.save(ep / "frames.npy", _clip(range(num_frames), 3))
    dino = np.broadcast_to(np.arange(num_frames, dtype=np.float16)[:, None, None, None], (num_frames, 3, 64, 768))
    np.save(ep / "dinov3.npy", np.ascontiguousarray(dino))
    action = np.stack([np.arange(num_frames), -np.arange(num_frames)], 1).astype(np.float32)
    np.save(ep / "action_proprio.npy",
            {"action": action, "proprio": np.zeros((num_frames, 3), np.float32)}, allow_pickle=True)
    (ep / "chunk_index.json").write_text(json.dumps({"chunk_frames": 4, "num_views": 3}))
    index = {}
    for v, frame_idx in VARIANTS.items():
        sampled = action[frame_idx]
        causal = np.zeros_like(sampled)
        causal[1:] = sampled[:-1]
        np.save(ep / f"action_causal_v{v}.npy", causal)
        n_chunks = expected_stored_chunks(len(frame_idx), 4, NATIVE)
        np.save(ep / f"latents_v{v}.npy", np.zeros((n_chunks, 3, 4), np.float32))
        index[f"v{v}"] = {"frame_idx": frame_idx, "n_frames": len(frame_idx), "n_chunks": n_chunks}
    (ep / "variant_index.json").write_text(json.dumps(index))
    # Source checksums of files that the re-encode tool rewrites.
    (ep / "variants.complete.json").write_text(json.dumps({"outputs": {"variant_index.json": {}}}))
    (ep / "base.complete.json").write_text(json.dumps({"outputs": {"chunk_index.json": {}}}))
    (ep / "checkpoint.complete.json").write_text("{}")
    (root / "meta" / "stage1_compact.json").write_text(json.dumps({"compact_files": {
        f"features/episode_0/{name}": {"bytes": 0, "dtype": "float32", "sha256": "0" * 64, "shape": [0]}
        for name in ("latents_v0.npy", "frames.npy")
    }}))
    return ep


def _reencode(tmp_path):
    tool = _load_module("reencode_wan_latents_under_test",
                        ROOT / "src/mme_vla_suite/dataset_builder/reencode_wan_latents.py")
    src_ep = _write_native_store(tmp_path / "src")
    before = {p.name: p.read_bytes() for p in src_ep.iterdir()}
    tool.reencode_store(tmp_path / "src", tmp_path / "out", _frontend(),
                        alignment=REPLICATE_FIRST_FRAME, num_views=3, wan_h=HW, wan_w=HW,
                        pool_h=1, pool_w=1)
    assert {p.name: p.read_bytes() for p in src_ep.iterdir()} == before   # source untouched
    return src_ep, tmp_path / "out" / "features" / "episode_0"


def test_reencoded_store_holds_aligned_latents_and_validates(tmp_path):
    validate = _load_module("validate_store_under_test", ROOT / "scripts/reproduce/validate_store.py")
    src_ep, out_ep = _reencode(tmp_path)
    index = json.loads((out_ep / "variant_index.json").read_text())
    assert json.loads((out_ep / "chunk_index.json").read_text())["wan_chunk_alignment"] == REPLICATE_FIRST_FRAME
    for v, frame_idx in VARIANTS.items():
        lat = np.load(out_ep / f"latents_v{v}.npy")
        assert lat.shape == (len(frame_idx) // 4, 3, 4)
        assert index[f"v{v}"]["n_chunks"] == len(frame_idx) // 4
        for c in range(lat.shape[0]):
            for view in range(3):
                first, last, _, group = _decode(lat[c], 3, view)
                assert (first, last, group) == (frame_idx[4 * c], frame_idx[4 * c + 3], 4)
    for name in ("frames.npy", "dinov3.npy", "action_proprio.npy", "action_causal_v0.npy",
                 "checkpoint.complete.json"):
        assert (out_ep / name).is_symlink()
    assert not (out_ep / "variants.complete.json").exists()
    assert not (out_ep / "base.complete.json").exists()
    compact_path = out_ep.parents[1] / "meta" / "stage1_compact.json"
    assert not compact_path.is_symlink()
    compact = json.loads(compact_path.read_text())["compact_files"]
    latents_entry = compact["features/episode_0/latents_v0.npy"]
    latents_file = out_ep / "latents_v0.npy"
    assert latents_entry["sha256"] == hashlib.sha256(latents_file.read_bytes()).hexdigest()
    assert latents_entry["bytes"] == latents_file.stat().st_size
    assert latents_entry["shape"] == list(np.load(latents_file).shape)
    assert compact["features/episode_0/frames.npy"]["sha256"] == "0" * 64   # linked: unchanged

    validate.validate_episode(out_ep, 2, 3, REPLICATE_FIRST_FRAME)
    validate.validate_episode(src_ep, 2, 3)
    with pytest.raises(ValueError, match="wan_chunk_alignment='native'"):
        validate.validate_episode(src_ep, 2, 3, REPLICATE_FIRST_FRAME)
    # Without the recorded field the validator applies native accounting: v1 has 12 kept
    # frames, so 3 aligned chunks where native stores hold (12 - 1) // 4 = 2.
    chunk_index = json.loads((out_ep / "chunk_index.json").read_text())
    chunk_index.pop("wan_chunk_alignment")
    (out_ep / "chunk_index.json").write_text(json.dumps(chunk_index))
    with pytest.raises(ValueError, match="v1 n_chunks=3"):
        validate.validate_episode(out_ep, 2, 3)


def _loader(feature_dir, alignment):
    from mme_vla_suite.training.dataset import RoboMMEDataset

    cfg = SimpleNamespace(
        visual_feature_stream="dinov3", latent_source="offline", n_variants=1,
        num_query_samples=2, pool_hw=0, pool_h=1, pool_w=1, wan_h=HW, wan_w=HW,
        siglip_dim=768, lang_emb_dim=2048, action_encoding="raw", proprio_encoding="raw",
        latent_cap=64, max_recent_tokens=768 * 3 * 64, wan_chunk_alignment=alignment,
    )
    ds = RoboMMEDataset.__new__(RoboMMEDataset)
    ds.feature_dir = feature_dir
    ds.history_config = SimpleNamespace(cosmo_bottleneck_memory=cfg)
    ds._episode_store_cache = None  # noqa: SLF001
    ds.need_aux = False
    return ds, cfg


def _gather(ds, t):
    return ds._gather_cosmo_bottleneck(epis_idx=0, step_idx=t)  # noqa: SLF001


def test_stage2_loader_reads_aligned_store_without_gaps(tmp_path):
    src_ep, out_ep = _reencode(tmp_path)
    ds, cfg = _loader(out_ep.parent, REPLICATE_FIRST_FRAME)
    assert require_feature_store_alignment(out_ep.parent.parent, cfg) == 1
    for t in range(21):
        sample = _gather(ds, t)
        kept = [f for f in VARIANTS[0] if f <= t]
        n_lat = int(np.asarray(sample["cosmo_mem_mask"]).sum())
        lat = np.asarray(sample["cosmo_latent_emb"])[:n_lat].reshape(-1, 3, 4)
        latent_frames = [f for c in lat for f in VARIANTS[0] if _decode(c, 3)[0] <= f <= _decode(c, 3)[1]]
        n_rec = int(np.asarray(sample["cosmo_recent_mask"]).sum())
        recent = np.asarray(sample["cosmo_recent_emb"])[:n_rec].reshape(-1, 3 * 64, 768)
        recent_frames = [int(r[0, 0]) for r in recent]
        assert latent_frames + recent_frames == kept, t

    ds_native, _ = _loader(src_ep.parent, REPLICATE_FIRST_FRAME)
    with pytest.raises(ValueError, match="re-encode the store"):
        _gather(ds_native, 20)
    ds_mixed, _ = _loader(out_ep.parent, None)
    with pytest.raises(ValueError, match="wan_chunk_alignment='replicate_first_frame'"):
        _gather(ds_mixed, 20)


def test_store_guard_treats_missing_field_as_native():
    require_store_alignment({"chunk_frames": 4}, SimpleNamespace(), where="ep")
    require_store_alignment({"chunk_frames": 4}, None, where="ep")
    with pytest.raises(ValueError, match="re-encode"):
        require_store_alignment({"chunk_frames": 4},
                                SimpleNamespace(wan_chunk_alignment=REPLICATE_FIRST_FRAME), where="ep")


def test_precomputed_tokens_need_the_requested_alignment(tmp_path):
    cache = tmp_path / "cache_native"
    cache.mkdir()
    np.savez(cache / "ep0_t20.npz", v0=np.ones((2, 3), np.float16))
    ds, cfg = _loader(tmp_path, REPLICATE_FIRST_FRAME)
    cfg.precompute_memory_dir = str(cache)
    with pytest.raises(ValueError, match="records wan_chunk_alignment='native'"):
        _gather(ds, 20)

    aligned = tmp_path / "cache_aligned"
    aligned.mkdir()
    np.savez(aligned / "ep0_t20.npz", v0=np.ones((2, 3), np.float16))
    write_alignment_record(aligned, REPLICATE_FIRST_FRAME)
    cfg.precompute_memory_dir = str(aligned)
    sample = _gather(ds, 20)
    assert sample["cosmo_mem_tokens"].shape == (2, 3)


def _stage1_ckpt(path, alignment):
    values = {"bottleneck/w": np.zeros(2, np.float32)}
    if alignment is not None:
        values[STAGE1_METADATA_KEY] = np.asarray(alignment)
    np.savez(path, **values)
    return path


def test_training_records_alignment_and_refuses_another_stage1(tmp_path):
    aligned_cfg = SimpleNamespace(wan_chunk_alignment=REPLICATE_FIRST_FRAME)
    released = _stage1_ckpt(tmp_path / "released.npz", None)
    aligned = _stage1_ckpt(tmp_path / "aligned.npz", REPLICATE_FIRST_FRAME)
    assert read_alignment_record(tmp_path) == NATIVE
    with pytest.raises(ValueError, match="records wan_chunk_alignment='native'"):
        record_training_alignment(tmp_path, aligned_cfg, stage1_ckpt=released)
    assert record_training_alignment(tmp_path, aligned_cfg, stage1_ckpt=aligned) == REPLICATE_FIRST_FRAME
    assert read_alignment_record(tmp_path) == REPLICATE_FIRST_FRAME
    assert record_training_alignment(tmp_path, None, stage1_ckpt=released) == NATIVE


def _stage1_modules():
    from flax import nnx

    return {"bottleneck": nnx.Linear(2, 2, rngs=nnx.Rngs(0)),
            "decoder": nnx.Linear(2, 2, rngs=nnx.Rngs(1))}


def test_stage1_checkpoint_records_alignment(tmp_path):
    from mme_vla_suite.scripts import pretrain_bottleneck

    pretrain_bottleneck._save_checkpoint(  # noqa: SLF001
        tmp_path, 7, _stage1_modules(), wan_chunk_alignment=REPLICATE_FIRST_FRAME
    )
    with np.load(tmp_path / "stage1_step7.npz") as npz:
        assert str(npz[STAGE1_METADATA_KEY]) == REPLICATE_FIRST_FRAME


def test_stage1_resume_needs_the_recorded_alignment(tmp_path):
    from mme_vla_suite.scripts import pretrain_bottleneck

    resume = pretrain_bottleneck._load_stage1_checkpoint_into_modules  # noqa: SLF001
    aligned_cfg = SimpleNamespace(wan_chunk_alignment=REPLICATE_FIRST_FRAME)
    pretrain_bottleneck._save_checkpoint(  # noqa: SLF001
        tmp_path, 1, _stage1_modules(), wan_chunk_alignment=REPLICATE_FIRST_FRAME
    )
    aligned = tmp_path / "stage1_step1.npz"
    resume(_stage1_modules(), aligned, cosmo_bottleneck_memory=aligned_cfg)
    with pytest.raises(ValueError, match="records wan_chunk_alignment='replicate_first_frame'"):
        resume(_stage1_modules(), aligned, cosmo_bottleneck_memory=None)
    with np.load(aligned) as npz:   # a released checkpoint has no record: native
        values = {key: npz[key] for key in npz.files if key != STAGE1_METADATA_KEY}
    np.savez(tmp_path / "released.npz", **values)
    resume(_stage1_modules(), tmp_path / "released.npz", cosmo_bottleneck_memory=None)
    with pytest.raises(ValueError, match="records wan_chunk_alignment='native'"):
        resume(_stage1_modules(), tmp_path / "released.npz", cosmo_bottleneck_memory=aligned_cfg)


def test_alignment_record_reads_stay_complete_during_rewrites(tmp_path):
    write_alignment_record(tmp_path, REPLICATE_FIRST_FRAME)
    stop = threading.Event()

    def rewrite():   # another precompute shard writes the same value
        while not stop.is_set():
            write_alignment_record(tmp_path, REPLICATE_FIRST_FRAME)

    writer = threading.Thread(target=rewrite)
    writer.start()
    try:
        reads = {read_alignment_record(tmp_path) for _ in range(2000)}
    finally:
        stop.set()
        writer.join()
    assert reads == {REPLICATE_FIRST_FRAME}
    assert [path.name for path in tmp_path.iterdir()] == [ALIGNMENT_RECORD]


def _load_policy(monkeypatch, tmp_path, *, record, requested):
    """Run create_trained_policy up to the data config; return the history config it used."""
    from omegaconf import OmegaConf

    from mme_vla_suite.policies import policy_config

    class _GuardPassedError(Exception):
        pass

    def _stop(*args, **kwargs):
        raise _GuardPassedError

    run_dir = tmp_path / "run"
    (run_dir / "1000").mkdir(parents=True)
    if record is not None:
        write_alignment_record(run_dir, record)
    history = OmegaConf.create({"cosmo_bottleneck_memory": {"latent_source": "offline"}})
    monkeypatch.setattr("mme_vla_suite.policies.policy_config._model.restore_params",
                        lambda *args, **kwargs: {})
    train_config = SimpleNamespace(
        model=SimpleNamespace(history_config="history.yaml",
                              load=lambda params: SimpleNamespace(history_config=history)),
        data=SimpleNamespace(create=_stop),
        assets_dirs=tmp_path,
    )
    with pytest.raises(_GuardPassedError):
        policy_config.create_trained_policy(train_config, run_dir / "1000",
                                            wan_chunk_alignment=requested)
    return history


def test_policy_load_applies_the_recorded_alignment(monkeypatch, tmp_path):
    history = _load_policy(monkeypatch, tmp_path, record=REPLICATE_FIRST_FRAME,
                           requested=REPLICATE_FIRST_FRAME)
    assert history.cosmo_bottleneck_memory.wan_chunk_alignment == REPLICATE_FIRST_FRAME
    _load_policy(monkeypatch, tmp_path / "released", record=None, requested=None)


@pytest.mark.parametrize(("record", "requested"), [(REPLICATE_FIRST_FRAME, None),
                                                   (REPLICATE_FIRST_FRAME, NATIVE),
                                                   (None, REPLICATE_FIRST_FRAME)])
def test_policy_load_refuses_another_alignment(monkeypatch, tmp_path, record, requested):
    monkeypatch.setenv("IMPRINT_PREFER_REQUESTED_HISTORY_CONFIG", "1")
    with pytest.raises(ValueError, match="records wan_chunk_alignment"):
        _load_policy(monkeypatch, tmp_path, record=record, requested=requested)


def _adapter(monkeypatch, name):
    # serve_robomme_tcp.py runs from the adapter folder and imports the adapter by name.
    monkeypatch.syspath_prepend(str(ADAPTER_DIR))
    return importlib.import_module(name)


def _get_model(monkeypatch, tmp_path, **settings):
    """Run the RMBench adapter get_model with a stand-in rmbench_policy."""
    from mme_vla_suite.training import config as training_config

    robomme_tcp_policy = _adapter(monkeypatch, "robomme_tcp_policy")
    for name in (
        "MEMFT_DETERMINISTIC_INFERENCE",
        "MEMFT_EXPECTED_MODEL_TYPE",
        "RMBENCH_TRACE_DIR",
        "RMBENCH_FULL_TRACE_DIR",
        "RMBENCH_DETAILED_PROFILE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMORY_LANGUAGE_SOURCE", "live")
    model = SimpleNamespace(model_type=SimpleNamespace(value="pi05"), pi05=True, use_history=True)
    monkeypatch.setattr(training_config, "get_config", lambda name: SimpleNamespace(model=model))
    create_kwargs = {}

    def create_policy(*args, **kwargs):
        create_kwargs.update(kwargs)
        return SimpleNamespace()

    fake_policy_module = ModuleType("rmbench_policy")
    fake_policy_module.create_policy = create_policy
    monkeypatch.setitem(sys.modules, "rmbench_policy", fake_policy_module)
    (tmp_path / "_CHECKPOINT_METADATA").write_text("{}")
    usr_args = {
        "checkpoint": str(tmp_path),
        "config_name": "imprint_rmbench_recurrent_memft",
        "history_config": "history.yaml",
        "prediction_horizon": 50,
        **settings,
    }
    return robomme_tcp_policy.get_model(usr_args), create_kwargs


def test_get_model_passes_and_records_wan_chunk_alignment(monkeypatch, tmp_path):
    server, create_kwargs = _get_model(monkeypatch, tmp_path, wan_chunk_alignment=REPLICATE_FIRST_FRAME)

    assert create_kwargs["wan_chunk_alignment"] == REPLICATE_FIRST_FRAME
    assert server.get_runtime_provenance()["wan_chunk_alignment"] == REPLICATE_FIRST_FRAME


def test_get_model_without_wan_chunk_alignment_uses_native(monkeypatch, tmp_path):
    _, create_kwargs = _get_model(monkeypatch, tmp_path)

    assert create_kwargs["wan_chunk_alignment"] == NATIVE


def test_get_model_rejects_unknown_wan_chunk_alignment(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="unknown wan_chunk_alignment"):
        _get_model(monkeypatch, tmp_path, wan_chunk_alignment="shifted")


def test_rmbench_create_policy_passes_the_alignment_to_the_policy_load(monkeypatch, tmp_path):
    rmbench_policy = _adapter(monkeypatch, "rmbench_policy")
    load_kwargs = {}
    monkeypatch.setattr(rmbench_policy.mme_policy_config, "create_trained_policy",
                        lambda train_config, checkpoint_dir, **kwargs: load_kwargs.update(kwargs))
    rmbench_policy.create_policy(
        tmp_path / "30000",
        config_name="imprint_rmbench_recurrent_memft",
        history_config="cosmo-wan-gated-delta-rmbench-all9-dinov3-memft-optimized.yaml",
        wan_chunk_alignment=REPLICATE_FIRST_FRAME,
    )

    assert load_kwargs["wan_chunk_alignment"] == REPLICATE_FIRST_FRAME
