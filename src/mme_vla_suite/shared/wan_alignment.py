"""Temporal alignment between kept memory frames and Wan2.2 latent chunks.

The Wan2.2 VAE is causal with temporal factor 4. It encodes the first input frame alone and
then groups of four input frames (1-4, 5-8, ...), one latent per group. The memory buffer, the
Stage-1/Stage-2 loaders, and the recurrent event builder all treat finalized chunk ``c`` as
kept frames ``4c .. 4c+3`` and start the recent visual window at kept frame ``4k`` after ``k``
chunks.

``native`` (default; the released stores and checkpoints)
    The kept frames are encoded as they are. Latent 0 holds kept frame 0 only and latent
    ``c >= 1`` holds kept frames ``4c-3 .. 4c``. After ``k >= 1`` finalized chunks, kept frames
    ``4k-3 .. 4k-1`` are in neither visual stream, and recurrent chunk event ``c`` pairs the
    video of kept frames ``4c-3 .. 4c`` with the actions and states of kept frames
    ``4c .. 4c+3``.

``replicate_first_frame``
    One copy of kept frame 0 is encoded before the kept frames and its latent is dropped.
    Latent ``c`` then holds exactly kept frames ``4c .. 4c+3``, which is the accounting that
    every consumer already uses (paper App. A.1.1). Stores must be re-encoded with
    ``dataset_builder/reencode_wan_latents.py`` and the memory must be retrained.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import uuid

NATIVE = "native"
REPLICATE_FIRST_FRAME = "replicate_first_frame"
WAN_CHUNK_ALIGNMENTS = (NATIVE, REPLICATE_FIRST_FRAME)
# A Stage-2 run directory and a precompute_mem output directory record their alignment in
# ALIGNMENT_RECORD; a Stage-1 .npz checkpoint records it under STAGE1_METADATA_KEY. A missing
# record means NATIVE (the released checkpoints and caches).
ALIGNMENT_RECORD = "wan_chunk_alignment.txt"
STAGE1_METADATA_KEY = "metadata/wan_chunk_alignment"


def resolve_wan_chunk_alignment(value) -> str:
    """Validate an alignment name. ``None`` means the native (released) alignment."""
    alignment = NATIVE if value is None else str(value).strip()
    if alignment not in WAN_CHUNK_ALIGNMENTS:
        raise ValueError(
            f"unknown wan_chunk_alignment {alignment!r}; expected one of {WAN_CHUNK_ALIGNMENTS}"
        )
    return alignment


def wan_chunk_alignment_from_config(cosmo_bottleneck_memory) -> str:
    """Return the alignment requested by a ``cosmo_bottleneck_memory`` config section."""
    return resolve_wan_chunk_alignment(
        getattr(cosmo_bottleneck_memory, "wan_chunk_alignment", None)
        if cosmo_bottleneck_memory is not None
        else None
    )


def store_wan_chunk_alignment(chunk_index: Mapping) -> str:
    """Return the alignment recorded in an episode ``chunk_index.json`` (absent: native)."""
    return resolve_wan_chunk_alignment(chunk_index.get("wan_chunk_alignment"))


def require_store_alignment(chunk_index: Mapping, cosmo_bottleneck_memory, *, where) -> None:
    """Refuse offline latents encoded with a different alignment than the config requests."""
    stored = store_wan_chunk_alignment(chunk_index)
    requested = wan_chunk_alignment_from_config(cosmo_bottleneck_memory)
    if stored != requested:
        raise ValueError(
            f"{where}: offline Wan latents use wan_chunk_alignment={stored!r}, but the "
            f"history config requests {requested!r}; re-encode the store with "
            "mme_vla_suite.dataset_builder.reencode_wan_latents or change the config"
        )


def read_alignment_record(directory) -> str:
    """Return the alignment recorded in ``directory`` (no record: native)."""
    path = Path(directory) / ALIGNMENT_RECORD
    return resolve_wan_chunk_alignment(path.read_text().strip() if path.is_file() else None)


def write_alignment_record(directory, alignment) -> None:
    """Replace the record atomically, so a concurrent reader sees a complete record."""
    text = resolve_wan_chunk_alignment(alignment) + "\n"
    tmp = Path(directory) / f".{ALIGNMENT_RECORD}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(text)
        os.replace(tmp, Path(directory) / ALIGNMENT_RECORD)
    finally:
        tmp.unlink(missing_ok=True)


def stage1_checkpoint_alignment(npz) -> str:
    """Return the alignment recorded in a loaded Stage-1 ``.npz`` (no record: native)."""
    if STAGE1_METADATA_KEY not in npz.files:
        return NATIVE
    return resolve_wan_chunk_alignment(str(npz[STAGE1_METADATA_KEY].item()))


def require_recorded_alignment(recorded, cosmo_bottleneck_memory, *, where) -> None:
    """Refuse a trained memory or a token cache whose alignment differs from the config."""
    recorded = resolve_wan_chunk_alignment(recorded)
    requested = wan_chunk_alignment_from_config(cosmo_bottleneck_memory)
    if recorded != requested:
        raise ValueError(
            f"{where} records wan_chunk_alignment={recorded!r}, but the history config "
            f"requests {requested!r}; set cosmo_bottleneck_memory.wan_chunk_alignment "
            "(RMBench evaluation: run_local.py --wan-chunk-alignment) to the recorded value, "
            "or rebuild this input with the requested alignment"
        )


def record_training_alignment(run_dir, cosmo_bottleneck_memory, *, stage1_ckpt=None) -> str:
    """Write the Stage-2 alignment record. Refuse a Stage-1 checkpoint with another alignment."""
    alignment = wan_chunk_alignment_from_config(cosmo_bottleneck_memory)
    if stage1_ckpt is not None:
        import numpy as np

        with np.load(stage1_ckpt) as npz:
            require_recorded_alignment(
                stage1_checkpoint_alignment(npz), cosmo_bottleneck_memory, where=stage1_ckpt
            )
    write_alignment_record(run_dir, alignment)
    return alignment


def require_feature_store_alignment(feature_store, cosmo_bottleneck_memory) -> int:
    """Check every ``features/episode_*/chunk_index.json`` of a store. Return the count."""
    paths = sorted(Path(feature_store).glob("features/episode_*/chunk_index.json"))
    for path in paths:
        require_store_alignment(json.loads(path.read_text()), cosmo_bottleneck_memory, where=path)
    return len(paths)


def expected_stored_chunks(num_frames: int, chunk_frames: int, alignment: str) -> int:
    """Number of offline latent chunks a store holds for a variant of ``num_frames`` frames.

    Native stores keep ``finalized_encode`` output (the terminal Wan latent is dropped).
    Replicated-first-frame stores keep every complete group of ``chunk_frames`` kept frames.
    """
    alignment = resolve_wan_chunk_alignment(alignment)
    num_frames = int(num_frames)
    if num_frames <= 0:
        return 0
    if alignment == NATIVE:
        return (num_frames - 1) // int(chunk_frames)
    return num_frames // int(chunk_frames)


def latent_frame_ranges(num_chunks: int, chunk_frames: int, alignment: str) -> list[range]:
    """Kept-frame indices that each latent chunk contains."""
    alignment = resolve_wan_chunk_alignment(alignment)
    cf = int(chunk_frames)
    if alignment == REPLICATE_FIRST_FRAME:
        return [range(cf * c, cf * c + cf) for c in range(int(num_chunks))]
    return [range(1) if c == 0 else range(cf * c - cf + 1, cf * c + 1) for c in range(int(num_chunks))]
