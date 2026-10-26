---
pretty_name: IMPRINT RoboCasa 600 Training Store
tags:
  - robotics
  - robocasa
  - dinov3
  - vision-language-action
---

# IMPRINT RoboCasa 600 Training Store

This public repository contains the immutable shared store used by the
canonical IMPRINT customized RoboCasa experiments:

- 600 demonstrations
- 12 task groups
- horizon-20 policy samples
- three-view pooled DINOv3 features
- eight history variants
- strictly causal variant-matched actions

The release is packaged as deterministic tar shards. Download the repository,
then reconstruct the `store/` root with:

```bash
python scripts/reproduce/extract_dataset_store.py \
  --archive-root /path/to/download \
  --output artifacts/datasets/robocasa
```

See the manifests for per-shard SHA256 checksums. The exact patched simulator
source is under `simulation/robocasa/environment/` in
`Amazon-FAR/imprint_datasets`.

The recorded evaluation scenes are published separately from the training
store under `simulation/robocasa/replay/v1/` at immutable dataset revision
`eb1a5bca3eea7085f3b0a7d89e26ef2ce242e7ae`. Install and verify them with:

```bash
python scripts/reproduce/setup_robocasa_replay_assets.py \
  --output artifacts/replay/robocasa-v1
```

The installer verifies archive SHA256
`ce73132d4bcad7476c730db9bbf0c007346d229d4b0e72ea2e607cb123121166`
before extracting any files.

The corresponding code, launch configurations, and validation tools are
published in the IMPRINT GitHub repository.
