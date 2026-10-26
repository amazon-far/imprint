# IMPRINT RoboCasa Replay Assets v1

This archive contains the immutable recorded-scene inputs used by the
paper-compatible evaluator for all 12 customized RoboCasa task groups.

## Layout

```text
retrieve/
  retrieve_combined350_lineage.json
  sources/
wash/
sweeten/
pan_transfer/lerobot/
rinse_cutting_board/lerobot/
robocasa365/
```

The Retrieve manifest and source HDF5 files are checksum-pinned by the
evaluator. Other task roots contain recorded `ep_meta.json`, `model.xml.gz`,
and `states.npz` payloads. Sweeten also includes its recovery metadata and
comparison evidence.

The public artifact is pinned by:

- Dataset revision:
  `6f2e15fe0d28f85a5dfe0b2fbdf49fa1267dab75`
- Archive SHA256:
  `ce73132d4bcad7476c730db9bbf0c007346d229d4b0e72ea2e607cb123121166`
- Extracted regular files: 3,989

Dataset and simulator assets remain subject to the source terms documented by
the public dataset card; this bundle does not relicense third-party assets.

Install the archive through the IMPRINT repository:

```bash
python scripts/reproduce/setup_robocasa_replay_assets.py \
  --output artifacts/replay/robocasa-v1
```

The command prints `ROBOCASA_REPLAY_ASSETS_READY` only after archive,
inventory, and Retrieve source-hash validation succeeds.

Then pass the extracted root to the coordinated evaluator:

```bash
python benchmarks/robocasa/eval/run_local.py \
  ... \
  --replay-assets-root artifacts/replay/robocasa-v1
```
