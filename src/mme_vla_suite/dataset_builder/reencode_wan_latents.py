#!/usr/bin/env python3
"""Build a derived offline store whose Wan latents use another chunk alignment.

The released var8 stores hold ``latents_v*.npy`` encoded with the native Wan2.2 alignment
(latent 0 = kept frame 0 alone, latent c >= 1 = kept frames 4c-3 .. 4c, terminal latent
dropped). The memory buffer, the loaders and the recurrent event builder read chunk c as kept
frames 4c .. 4c+3. This tool re-encodes every variant with
``WanFrontend.encode_multiview(frames, alignment="replicate_first_frame")`` so that latent c
holds exactly kept frames 4c .. 4c+3 (see ``mme_vla_suite.shared.wan_alignment``).

Only the RMBench store layout is supported: ``frames.npy`` at ``wan_h x wan_w`` in each
episode and latents pooled per view on disk.

The source store is not changed. The output store links every other file to the source
(absolute real paths, as ``assemble_var8_base.py`` does) and writes new
``latents_v*.npy``, ``variant_index.json`` (``n_chunks``) and ``chunk_index.json``
(``wan_chunk_alignment``) files. It writes ``meta/stage1_compact.json`` with new entries for
the rewritten files, and it does not link the episode completion markers in ``NOT_LINKED``,
which record checksums of the source files. Latents are pooled per view to
``pool_h x pool_w`` with the same block mean as ``CosmoMemoryBuffer._pool_latents``. Set
``cosmo_bottleneck_memory.wan_chunk_alignment`` to the same value for training and evaluation.

Example::

    python -m mme_vla_suite.dataset_builder.reencode_wan_latents \\
        --src /path/to/store/stage1 --out /path/to/store/stage1_aligned \\
        --num-views 3 --wan-h 256 --wan-w 256 --pool-h 2 --pool-w 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from mme_vla_suite.shared.wan_alignment import NATIVE
from mme_vla_suite.shared.wan_alignment import REPLICATE_FIRST_FRAME
from mme_vla_suite.shared.wan_alignment import expected_stored_chunks
from mme_vla_suite.shared.wan_alignment import resolve_wan_chunk_alignment

REWRITTEN = ("variant_index.json", "chunk_index.json")
# Source completion markers with checksums of rewritten files; no loader reads them.
NOT_LINKED = ("base.complete.json", "variants.complete.json")
COMPACT_MANIFEST = "stage1_compact.json"
COMPLETE_MARKER = "wan_alignment.complete.json"


def _link(src: Path, dst: Path) -> None:
    src = Path(os.path.realpath(src))
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def _atomic_write_bytes(path: Path, write) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _write_json(path: Path, value) -> None:
    _atomic_write_bytes(path, lambda h: h.write((json.dumps(value, indent=1) + "\n").encode()))


def _write_npy(path: Path, array: np.ndarray) -> None:
    _atomic_write_bytes(path, lambda h: np.save(h, array, allow_pickle=False))


def pool_view_grid(latents: np.ndarray, *, num_views: int, grid_h: int, grid_w: int,
                   pool_h: int, pool_w: int) -> np.ndarray:
    """(nc, V*grid_h*grid_w, d) -> (nc, V*pool_h*pool_w, d) block mean per view."""
    nc, tokens, dim = latents.shape
    if tokens != num_views * grid_h * grid_w:
        raise ValueError(f"latent tokens {tokens} != {num_views}*{grid_h}*{grid_w}")
    if grid_h % pool_h or grid_w % pool_w:
        raise ValueError(f"pool {pool_h}x{pool_w} must divide grid {grid_h}x{grid_w}")
    bh, bw = grid_h // pool_h, grid_w // pool_w
    grid = latents.reshape(nc, num_views, pool_h, bh, pool_w, bw, dim).mean(axis=(3, 5))
    return grid.reshape(nc, num_views * pool_h * pool_w, dim).astype(np.float32)


def reencode_episode(src_dir: Path, out_dir: Path, frontend, *, alignment: str, num_views: int,
                     wan_h: int, wan_w: int, pool_h: int, pool_w: int) -> dict:
    """Write one derived episode directory. Returns a short summary."""
    if (out_dir / COMPLETE_MARKER).is_file():
        return {"episode": src_dir.name, "skipped": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_index = json.loads((src_dir / "chunk_index.json").read_text())
    stored = resolve_wan_chunk_alignment(chunk_index.get("wan_chunk_alignment"))
    if stored != NATIVE:
        raise ValueError(f"{src_dir}: source latents already use {stored!r}")
    chunk_frames = int(chunk_index["chunk_frames"])
    if chunk_frames != int(frontend.temporal_factor):
        raise ValueError(
            f"{src_dir}: chunk_frames={chunk_frames} != Wan temporal factor "
            f"{frontend.temporal_factor}"
        )
    variant_index = json.loads((src_dir / "variant_index.json").read_text())
    frames = np.load(src_dir / "frames.npy", mmap_mode="r", allow_pickle=False)
    if frames.ndim != 5 or frames.shape[1] < num_views or frames.shape[-1] != 3:
        raise ValueError(f"{src_dir}: unexpected frames shape {frames.shape}")
    if tuple(frames.shape[2:4]) != (wan_h, wan_w):
        raise ValueError(
            f"{src_dir}: frames are {frames.shape[2]}x{frames.shape[3]}, Wan input is "
            f"{wan_h}x{wan_w}; resizing is not implemented"
        )

    for entry in sorted(src_dir.iterdir()):
        name = entry.name
        if name.startswith("latents_v") and name.endswith(".npy"):
            continue
        if name in REWRITTEN or name in NOT_LINKED or name == COMPLETE_MARKER:
            continue
        _link(entry, out_dir / name)

    grid_h, grid_w = wan_h // 16, wan_w // 16
    new_index = {}
    for key in sorted(variant_index, key=lambda k: int(k[1:])):
        meta = dict(variant_index[key])
        frame_idx = np.asarray(meta["frame_idx"], np.int64)
        expected = expected_stored_chunks(frame_idx.size, chunk_frames, alignment)
        if frame_idx.size:
            clip = np.ascontiguousarray(frames[frame_idx][:, :num_views])
            latents = np.asarray(frontend.encode_multiview(clip, alignment=alignment), np.float32)
        else:
            latents = np.zeros((0, num_views * grid_h * grid_w, int(frontend.latent_dim)), np.float32)
        if latents.shape[0] != expected:
            raise RuntimeError(
                f"{src_dir} {key}: frontend returned {latents.shape[0]} chunks, expected {expected}"
            )
        pooled = (pool_view_grid(latents, num_views=num_views, grid_h=grid_h, grid_w=grid_w,
                                 pool_h=pool_h, pool_w=pool_w)
                  if latents.shape[0] else
                  np.zeros((0, num_views * pool_h * pool_w, latents.shape[-1]), np.float32))
        src_latents = np.load(src_dir / f"latents_{key}.npy", mmap_mode="r", allow_pickle=False)
        if src_latents.ndim != 3 or src_latents.shape[1:] != pooled.shape[1:]:
            raise ValueError(
                f"{src_dir} {key}: source latents {src_latents.shape} and re-encoded "
                f"{pooled.shape} have different token layouts"
            )
        _write_npy(out_dir / f"latents_{key}.npy", pooled)
        meta["n_chunks"] = int(pooled.shape[0])
        new_index[key] = meta

    chunk_index = dict(chunk_index)
    chunk_index["wan_chunk_alignment"] = alignment
    _write_json(out_dir / "variant_index.json", new_index)
    _write_json(out_dir / "chunk_index.json", chunk_index)
    _write_json(out_dir / COMPLETE_MARKER, {"wan_chunk_alignment": alignment,
                                            "source": str(src_dir.resolve())})
    return {"episode": src_dir.name, "skipped": False,
            "n_chunks": {k: v["n_chunks"] for k, v in new_index.items()}}


def reencode_store(src: Path, out: Path, frontend, *, alignment: str = REPLICATE_FIRST_FRAME,
                   num_views: int, wan_h: int, wan_w: int, pool_h: int, pool_w: int,
                   episodes: list[int] | None = None, provenance: dict | None = None) -> list[dict]:
    """Re-encode ``src`` into the derived store ``out``. ``src`` is only read."""
    alignment = resolve_wan_chunk_alignment(alignment)
    if alignment == NATIVE:
        raise ValueError("the native alignment is the source layout; nothing to re-encode")
    src = Path(src).resolve()
    out = Path(out).resolve()
    if out == src or src in out.parents:
        raise ValueError("--out must be outside the source store")
    src_episodes = sorted((src / "features").glob("episode_*"),
                          key=lambda p: int(p.name.rsplit("_", 1)[1]))
    if episodes is not None:
        wanted = {int(e) for e in episodes}
        src_episodes = [p for p in src_episodes if int(p.name.rsplit("_", 1)[1]) in wanted]
    if not src_episodes:
        raise FileNotFoundError(f"no episodes selected under {src / 'features'}")

    (out / "features").mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        if entry.name in ("features", "meta"):
            continue
        _link(entry, out / entry.name)
    if (src / "meta").is_dir():
        (out / "meta").mkdir(exist_ok=True)
        for entry in sorted((src / "meta").iterdir()):
            if entry.name != COMPACT_MANIFEST:
                _link(entry, out / "meta" / entry.name)
    _write_json(out / "meta" / "wan_chunk_alignment.json", {
        "wan_chunk_alignment": alignment,
        "source_store": str(src),
        "num_views": num_views, "wan_h": wan_h, "wan_w": wan_w,
        "pool_h": pool_h, "pool_w": pool_w,
        **(provenance or {}),
    })

    results = []
    for src_dir in src_episodes:
        result = reencode_episode(src_dir, out / "features" / src_dir.name, frontend,
                                  alignment=alignment, num_views=num_views, wan_h=wan_h,
                                  wan_w=wan_w, pool_h=pool_h, pool_w=pool_w)
        print(json.dumps(result), flush=True)
        results.append(result)
    _write_compact_manifest(src, out)
    return results


def _write_compact_manifest(src: Path, out: Path) -> None:
    """Copy ``meta/stage1_compact.json`` with entries that describe the derived store."""
    path = src / "meta" / COMPACT_MANIFEST
    if not path.is_file():
        return
    manifest = json.loads(path.read_text())
    files = {}
    for rel, source_entry in manifest.get("compact_files", {}).items():
        target = out / rel
        if not target.exists():
            continue  # an episode that --episodes left out
        entry = dict(source_entry)
        if not target.is_symlink():  # a file that this tool wrote
            entry.update(bytes=target.stat().st_size,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            if "shape" in entry:
                array = np.load(target, mmap_mode="r", allow_pickle=False)
                entry.update(dtype=str(array.dtype), shape=list(array.shape))
        files[rel] = entry
    _write_json(out / "meta" / COMPACT_MANIFEST, dict(manifest, compact_files=files))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alignment", default=REPLICATE_FIRST_FRAME)
    parser.add_argument("--num-views", type=int, required=True)
    parser.add_argument("--wan-h", type=int, required=True)
    parser.add_argument("--wan-w", type=int, required=True)
    parser.add_argument("--pool-h", type=int, required=True)
    parser.add_argument("--pool-w", type=int, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16",
                        help="bfloat16 matches the evaluator's WanFrontend default.")
    args = parser.parse_args(argv)

    import torch

    from mme_vla_suite.models.representation import wan_frontend

    frontend = wan_frontend.WanFrontend(device=args.device, dtype=getattr(torch, args.dtype))
    reencode_store(
        args.src, args.out, frontend, alignment=args.alignment, num_views=args.num_views,
        wan_h=args.wan_h, wan_w=args.wan_w, pool_h=args.pool_h, pool_w=args.pool_w,
        episodes=args.episodes,
        provenance={"wan_repo": wan_frontend._HF_REPO,  # noqa: SLF001
                    "wan_revision": wan_frontend._HF_REVISION,  # noqa: SLF001
                    "dtype": args.dtype, "device": args.device},
    )


if __name__ == "__main__":
    main()
