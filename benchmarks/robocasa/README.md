# Customized RoboCasa Benchmark

This evaluator targets the patched RoboCasa 1.0.1 replay environment used by
the paper. It covers 12 task groups:

- Retrieve fruit
- Retrieve oil
- Wash and return, left/right
- Wash and return, same location
- Sweeten hot chocolate
- Pan transfer
- Rinse cutting board
- Organize condiments
- Pick/place counter to sink
- Place equal ice cubes
- Scrub cutting board
- Frying-pan adjustment

## Required Environment

Use the archived customized RoboCasa environment, not an unmodified upstream
installation. The required changes include:

- memory-task environment registrations
- legacy scene-index mapping
- the `framed_window` compatibility alias
- recorded-scene XML/state replay
- transient disposable-scene retry
- task-specific terminal success trackers

Download and verify the exact source environment:

```bash
mkdir -p artifacts/environments/robocasa
hf download Amazon-FAR/imprint_datasets \
  simulation/robocasa/environment/robocasa-memory-setup.tar.gz \
  --repo-type dataset \
  --local-dir artifacts/downloads/imprint_datasets
bash scripts/reproduce/setup_robocasa_eval.sh \
  artifacts/downloads/imprint_datasets/simulation/robocasa/environment/robocasa-memory-setup.tar.gz \
  .external/robocasa-memory-setup
```

The setup downloads the official RoboCasa bundles and the pinned legacy
RoboCasa 0.2 scene/assets required by the migrated memory tasks.

Set these paths:

```bash
export ROBOCASA_ROOT=/path/to/patched-robocasa
export ROBOCASA_REPLAY_ASSETS=/path/to/replay-assets
export IMPRINT_CHECKPOINT=/path/to/stage2/checkpoint
export DINOV3_MODEL_PATH=/path/to/dinov3-vitb16
```

Download and validate the exact replay scenes for all 12 task groups:

```bash
python scripts/reproduce/setup_robocasa_replay_assets.py \
  --output artifacts/replay/robocasa-v1
export ROBOCASA_REPLAY_ASSETS=$PWD/artifacts/replay/robocasa-v1
```

This includes the checksum-pinned Retrieve manifest and its six HDF5 source
datasets, Wash scenes, the recovered Sweeten scenes, Pan Transfer and Rinse
Cutting Board scenes, and the five RoboCasa365 task bundles. The installer
uses immutable dataset revision
`eb1a5bca3eea7085f3b0a7d89e26ef2ce242e7ae`, verifies archive SHA256
`ce73132d4bcad7476c730db9bbf0c007346d229d4b0e72ea2e607cb123121166`,
and prints `ROBOCASA_REPLAY_ASSETS_READY` only after all inventories and
Retrieve source hashes pass.

If the preserved policy environment predates native DINOv3 support in
Transformers, set `DINOV3_RUNTIME_OVERLAY` to a compatible Python package
overlay and pass `--dinov3-runtime-overlay "$DINOV3_RUNTIME_OVERLAY"`.
Omit that argument in environments that already import DINOv3.

The coordinated launcher enforces recorded-state Retrieve replay, disables
synthetic Wash exploration, and aligns memory actions causally to the previous
raw action. The customized simulator must include
`scripts/replay/replay_memory_demos.py`; use `--memory-replay-helper` to point
to a nonstandard location.

Run a one-trial task-group smoke. The launcher starts and stops the memory
policy automatically:

```bash
python benchmarks/robocasa/eval/run_local.py \
  --robocasa-root "$ROBOCASA_ROOT" \
  --checkpoint "$IMPRINT_CHECKPOINT" \
  --dinov3-model "$DINOV3_MODEL_PATH" \
  --architecture full_attention \
  --task-group organize_condiments \
  --pilot-trials 1 \
  --replay-assets-root "$ROBOCASA_REPLAY_ASSETS" \
  --sim-python "$ROBOCASA_ROOT/.venv-mem/bin/python" \
  --output outputs/robocasa/organize-condiments-smoke
```

`ROBOCASA_REPLAY_ASSETS` is also used as the default for
`--replay-assets-root`. Task-specific `ROBOCASA_*` variables and command-line
flags remain available as overrides.

## Paper-Compatible Reporting

The supported launcher writes `paper_metrics.json` beside the evaluator logs.
Raw simulator success is retained. Sweeten Hot Chocolate uses the paper's
recovered placement-completion rule,
`stable_placement_count == goal_count`; all other task groups use raw success.

Apply the same reporting rule to an existing result directory with:

```bash
python benchmarks/robocasa/eval/postprocess_results.py \
  outputs/robocasa/sweeten/sweeten_hot_chocolate/group_stats.json \
  --output outputs/robocasa/sweeten/paper_metrics.json
```
