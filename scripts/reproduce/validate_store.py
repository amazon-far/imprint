#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Mirrors mme_vla_suite.shared.wan_alignment (kept inline: this script imports numpy only).
WAN_CHUNK_ALIGNMENTS = ("native", "replicate_first_frame")


def expected_stored_chunks(num_frames: int, chunk_frames: int, alignment: str) -> int:
    """native: finalized_encode drops the terminal Wan latent -> (n - 1) // cf chunks.
    replicate_first_frame: every complete group of cf kept frames -> n // cf chunks."""
    if alignment == "native":
        return (num_frames - 1) // chunk_frames
    return num_frames // chunk_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-store", "--store", dest="feature_store", type=Path, required=True)
    parser.add_argument("--policy-store", type=Path)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--proprio-dim", type=int, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--wan-chunk-alignment",
        choices=WAN_CHUNK_ALIGNMENTS,
        default=None,
        help="Require this Wan chunk alignment in every checked chunk_index.json "
        "(default: accept the recorded value; absent means native).",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    value = np.load(path, allow_pickle=True)
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def validate_episode(
    path: Path,
    action_dim: int,
    proprio_dim: int,
    wan_chunk_alignment: str | None = None,
) -> None:
    dino = np.load(path / "dinov3.npy", mmap_mode="r")
    if dino.ndim != 4 or tuple(dino.shape[1:]) != (3, 64, 768):
        raise ValueError(f"{path}: unexpected DINO shape {dino.shape}")

    payload = load_payload(path / "action_proprio.npy")
    action = np.asarray(payload["action"])
    proprio = np.asarray(payload["proprio"])
    if action.ndim != 2 or action.shape[-1] != action_dim:
        raise ValueError(f"{path}: unexpected action shape {action.shape}")
    if proprio.ndim != 2 or proprio.shape[-1] != proprio_dim:
        raise ValueError(f"{path}: unexpected proprio shape {proprio.shape}")
    if dino.shape[0] != action.shape[0] or proprio.shape[0] != action.shape[0]:
        raise ValueError(f"{path}: raw stream lengths do not match")

    variants = sorted(path.glob("action_causal_v*.npy"))
    if variants:
        variant_index_path = path / "variant_index.json"
        chunk_index_path = path / "chunk_index.json"
        if not variant_index_path.is_file() or not chunk_index_path.is_file():
            raise FileNotFoundError(f"{path}: missing variant or chunk index")
        variant_index = json.loads(variant_index_path.read_text())
        chunk_index = json.loads(chunk_index_path.read_text())
        chunk_frames = int(chunk_index["chunk_frames"])
        if chunk_frames < 1:
            raise ValueError(f"{chunk_index_path}: chunk_frames must be positive")
        alignment = str(chunk_index.get("wan_chunk_alignment", "native"))
        if alignment not in WAN_CHUNK_ALIGNMENTS:
            raise ValueError(f"{chunk_index_path}: unknown wan_chunk_alignment {alignment!r}")
        if wan_chunk_alignment is not None and alignment != wan_chunk_alignment:
            raise ValueError(
                f"{chunk_index_path}: wan_chunk_alignment={alignment!r}, "
                f"expected {wan_chunk_alignment!r}"
            )
        if len(variants) != 8:
            raise ValueError(f"{path}: expected eight causal-action variants")

        for variant_path in variants:
            variant = int(variant_path.stem.rsplit("v", 1)[1])
            variant_metadata = variant_index[f"v{variant}"]
            frame_idx = np.asarray(
                variant_metadata["frame_idx"],
                dtype=np.int64,
            )
            if (
                frame_idx.ndim != 1
                or frame_idx.size == 0
                or frame_idx[0] != 0
                or np.any(np.diff(frame_idx) <= 0)
                or frame_idx[-1] >= action.shape[0]
            ):
                raise ValueError(f"{variant_index_path}: invalid frame_idx for v{variant}")

            causal = np.asarray(np.load(variant_path))
            if causal.shape != (frame_idx.size, action_dim):
                raise ValueError(
                    f"{variant_path}: {causal.shape} != "
                    f"{(frame_idx.size, action_dim)}"
                )
            sampled = np.ascontiguousarray(action[frame_idx])
            expected = np.zeros_like(sampled)
            expected[1:] = sampled[:-1]
            if not np.allclose(causal, expected):
                raise ValueError(
                    f"{variant_path}: causal actions do not match sampled variant"
                )

            latent_path = path / f"latents_v{variant}.npy"
            if not latent_path.is_file():
                raise FileNotFoundError(latent_path)
            latents = np.load(latent_path, mmap_mode="r")
            metadata_frames = int(variant_metadata["n_frames"])
            metadata_chunks = int(variant_metadata["n_chunks"])
            if metadata_frames != frame_idx.size:
                raise ValueError(
                    f"{variant_index_path}: v{variant} n_frames={metadata_frames} "
                    f"but frame_idx has {frame_idx.size} entries"
                )
            expected_chunks = expected_stored_chunks(frame_idx.size, chunk_frames, alignment)
            if metadata_chunks != expected_chunks:
                raise ValueError(
                    f"{variant_index_path}: v{variant} n_chunks={metadata_chunks}, "
                    f"expected {expected_chunks} {alignment} chunks"
                )
            if latents.ndim != 3 or latents.shape[0] != expected_chunks:
                raise ValueError(
                    f"{latent_path}: expected {expected_chunks} chunks, "
                    f"found shape {latents.shape}"
                )
    elif "action_causal" in payload:
        causal = np.asarray(payload["action_causal"])
        if causal.shape != action.shape:
            raise ValueError(f"{path}: causal action shape mismatch")
        if not np.allclose(causal[0], 0.0) or not np.allclose(causal[1:], action[:-1]):
            raise ValueError(f"{path}: causal action shift is invalid")
    else:
        raise ValueError(f"{path}: no causal action stream")

    if len(list(path.glob("latents_v*.npy"))) != 8:
        raise ValueError(f"{path}: expected eight offline latent variants")


def main() -> None:
    args = parse_args()
    feature_store = args.feature_store.expanduser().resolve()
    policy_store = (args.policy_store or feature_store).expanduser().resolve()
    episodes = sorted(
        (feature_store / "features").glob("episode_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )
    if len(episodes) != args.episodes:
        raise ValueError(f"expected {args.episodes} feature episodes, found {len(episodes)}")

    manifest = feature_store / "meta/dino_store_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    dino_manifest = json.loads(manifest.read_text())
    if int(dino_manifest["episodes"]) != args.episodes:
        raise ValueError("DINO manifest episode count does not match")

    selected = episodes if args.full else episodes[: min(3, len(episodes))]
    for episode in selected:
        validate_episode(
            episode, args.action_dim, args.proprio_dim, args.wan_chunk_alignment
        )

    stats = policy_store / "meta/stats.json"
    data = policy_store / "data"
    if not stats.is_file() or not data.is_dir():
        raise FileNotFoundError(f"policy store is incomplete: {policy_store}")
    sample_count = json.loads(stats.read_text()).get("execution_samples")
    if sample_count is None:
        sample_count = json.loads(stats.read_text()).get("total_samples")
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError(f"invalid policy sample count in {stats}")

    print(
        "STORE_VALID "
        f"episodes={len(episodes)} checked={len(selected)} "
        f"policy_samples={sample_count} feature_store={feature_store} "
        f"policy_store={policy_store}"
    )


if __name__ == "__main__":
    main()
