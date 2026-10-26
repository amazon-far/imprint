from __future__ import annotations

import importlib.util
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "setup_robocasa_replay_assets",
    ROOT / "scripts/reproduce/setup_robocasa_replay_assets.py",
)
assert SPEC is not None and SPEC.loader is not None
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    source = tmp_path / "payload.txt"
    source.write_text("payload")
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as output:
        output.add(source, arcname="../payload.txt")

    with pytest.raises(ValueError, match="unsafe archive member"):
        SETUP.safe_extract(archive, tmp_path / "output")


def test_safe_extract_accepts_regular_files(tmp_path: Path) -> None:
    source = tmp_path / "payload.txt"
    source.write_text("payload")
    archive = tmp_path / "safe.tar"
    with tarfile.open(archive, "w") as output:
        output.add(source, arcname="nested/payload.txt")
    destination = tmp_path / "output"
    destination.mkdir()

    SETUP.safe_extract(archive, destination)

    assert (destination / "nested/payload.txt").read_text() == "payload"
