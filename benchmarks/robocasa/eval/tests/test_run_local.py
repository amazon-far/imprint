from __future__ import annotations

import argparse
from pathlib import Path
import sys

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from run_local import REPLAY_ASSET_LAYOUT  # noqa: E402
from run_local import replay_asset_paths  # noqa: E402
from run_local import task_asset_paths  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    values = {
        "replay_assets_root": None,
        "memory_data_root": None,
        "memory_store": None,
        "wash_scene_root": None,
        "retrieve_lineage_manifest": None,
        "retrieve_source_root": None,
        "sweeten_recovered_root": None,
        "pan_transfer_dataset": None,
        "rinse_cutting_board_dataset": None,
        "new5_scene_root": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_replay_root_maps_every_public_asset() -> None:
    root = Path("/tmp/replay")
    resolved_root = root.resolve()

    assert replay_asset_paths(root) == {
        name: resolved_root / relative
        for name, relative in REPLAY_ASSET_LAYOUT.items()
    }


def test_explicit_task_path_overrides_replay_bundle(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "replay"
    override = tmp_path / "custom-wash"

    paths = task_asset_paths(
        _args(replay_assets_root=replay, wash_scene_root=override),
        EVAL_ROOT,
    )

    assert paths["wash-scene-root"] == override.resolve()
    assert paths["retrieve-source-root"] == (
        replay / "retrieve/sources"
    ).resolve()


def test_repository_manifest_is_default_without_bundle() -> None:
    paths = task_asset_paths(_args(), EVAL_ROOT)

    assert paths == {
        "retrieve-lineage-manifest": (
            EVAL_ROOT / "retrieve_combined350_lineage.json"
        )
    }
