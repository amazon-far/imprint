#!/usr/bin/env python3
"""Download, verify, and extract the paper RoboCasa replay assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


REPOSITORY = "Amazon-FAR/imprint_datasets"
REPOSITORY_REVISION = "eb1a5bca3eea7085f3b0a7d89e26ef2ce242e7ae"
ARCHIVE_PATH = (
    "simulation/robocasa/replay/v1/"
    "robocasa-replay-assets-v1.tar"
)
ARCHIVE_SHA256 = (
    "ce73132d4bcad7476c730db9bbf0c007346d229d4b0e72ea2e607cb123121166"
)
RETRIEVE_MANIFEST_SHA256 = (
    "1bc4b06030712c15c717d9e0f8036d950994d358854713f5bac727b43f7e4061"
)
SCENE_FILES = frozenset({"ep_meta.json", "model.xml.gz", "states.npz"})
WASH_COUNTS = {
    "MemWashAndReturnLeft": 25,
    "MemWashAndReturnRight": 25,
    "MemWashAndReturnSameLocation": 51,
}
RECORDED_DATASETS = {
    "pan_transfer/lerobot": 510,
    "rinse_cutting_board/lerobot": 107,
    (
        "robocasa365/v1.0/pretrain/composite/"
        "OrganizeCondiments/20250731/lerobot"
    ): 105,
    (
        "robocasa365/v1.0/pretrain/atomic/"
        "PickPlaceCounterToSink/20250819/lerobot"
    ): 108,
    (
        "robocasa365/v1.0/pretrain/composite/"
        "PlaceEqualIceCubes/20250801/lerobot"
    ): 104,
    (
        "robocasa365/v1.0/pretrain/composite/"
        "ScrubCuttingBoard/20250805/lerobot"
    ): 103,
    (
        "robocasa365/v1.0/pretrain/composite/"
        "FryingPanAdjustment/20250714/lerobot"
    ): 108,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Empty destination for the extracted replay asset root.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded archive instead of Hugging Face.",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(max_workers: int) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is required when --archive is omitted"
        ) from error
    snapshot = Path(
        snapshot_download(
            repo_id=REPOSITORY,
            repo_type="dataset",
            revision=REPOSITORY_REVISION,
            allow_patterns=[ARCHIVE_PATH],
            max_workers=max_workers,
        )
    )
    return snapshot / ARCHIVE_PATH


def safe_extract(archive: Path, output: Path) -> None:
    output_resolved = output.resolve()
    with tarfile.open(archive, "r") as source:
        members = source.getmembers()
        for member in members:
            member_path = Path(member.name)
            destination = (output / member_path).resolve()
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or not destination.is_relative_to(output_resolved)
                or member.issym()
                or member.islnk()
                or member.isdev()
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
        source.extractall(
            output,
            members=members,
            filter="fully_trusted",
        )


def validate_episode_inventory(
    dataset: Path,
    expected_count: int,
) -> None:
    extras = dataset / "extras"
    episodes = sorted(extras.glob("episode_*"))
    if len(episodes) != expected_count:
        raise ValueError(
            f"{dataset}: expected {expected_count} episodes, "
            f"found {len(episodes)}"
        )
    for episode in episodes:
        if not episode.is_dir():
            raise ValueError(f"invalid episode path: {episode}")
        missing = [
            name for name in SCENE_FILES if not (episode / name).is_file()
        ]
        if missing:
            raise ValueError(f"{episode}: missing {missing}")


def validate_retrieve(root: Path) -> None:
    manifest_path = root / "retrieve/retrieve_combined350_lineage.json"
    if sha256(manifest_path) != RETRIEVE_MANIFEST_SHA256:
        raise ValueError("Retrieve lineage manifest digest is invalid")
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("Retrieve lineage manifest must contain 100 rows")
    source_root = root / "retrieve/sources"
    expected_hashes = {
        row["source_file"]: row["source_file_sha256"] for row in rows
    }
    for relative, expected in expected_hashes.items():
        source = source_root / relative
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Retrieve source digest is invalid: {source}")


def validate_sweeten(root: Path) -> None:
    sweeten = root / "sweeten"
    expected = {
        "extras",
        "comparisons",
        "recovery_report.json",
        "RECOVERY_COMPLETE.json",
    }
    if {path.name for path in sweeten.iterdir()} != expected:
        raise ValueError("Sweeten recovered root has an invalid inventory")
    episodes = sorted((sweeten / "extras").glob("episode_*"))
    if len(episodes) != 50:
        raise ValueError(
            f"Sweeten recovered root must contain 50 episodes, "
            f"found {len(episodes)}"
        )
    expected_files = SCENE_FILES | {"recovery.json"}
    for episode in episodes:
        if {path.name for path in episode.iterdir()} != expected_files:
            raise ValueError(f"Sweeten episode inventory is invalid: {episode}")


def validate(root: Path) -> None:
    validate_retrieve(root)
    for environment, count in WASH_COUNTS.items():
        validate_episode_inventory(
            root / "wash" / environment / "lerobot",
            count,
        )
    validate_sweeten(root)
    for relative, count in RECORDED_DATASETS.items():
        validate_episode_inventory(root / relative, count)


def main() -> None:
    args = parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output: {output}")

    archive = (
        args.archive.expanduser().resolve()
        if args.archive is not None
        else download_archive(args.max_workers)
    )
    actual = sha256(archive)
    if actual != ARCHIVE_SHA256:
        raise SystemExit(
            f"archive SHA256 mismatch: expected {ARCHIVE_SHA256}, got {actual}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        safe_extract(archive, temporary)
        validate(temporary)
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"ROBOCASA_REPLAY_ASSETS_READY root={output} "
        f"archive_sha256={actual}"
    )


if __name__ == "__main__":
    main()
