# IMPRINT

IMPRINT is a policy-agnostic long-horizon memory module for
vision-language-action policies. This repository contains the code, paper
configurations, training stores, checkpoints, and evaluation harnesses for
RMBench and a customized RoboCasa benchmark.

> **Simulation-only release.** This repository intentionally excludes the
> IMPRINT real-world training and evaluation pipeline, real-world datasets and
> checkpoints, private storage locations, and machine-specific paths. Vendored
> upstream backbone packages may retain generic robotics utilities and
> documentation; they are not IMPRINT real-world experiment code or artifacts.

IMPRINT provides two interchangeable memory encoders:

| Encoder | How history is processed | Best use |
| --- | --- | --- |
| **Full attention** | Memory queries attend over the retained history | Reference implementation |
| **Recurrent** | A gated delta memory updates a fixed-size state incrementally | Long histories and lower update cost |

Both variants use the same two-stage recipe: bootstrap the memory encoder with
reconstruction objectives, then adapt it with a VLA policy.

## 🧩 Policy Backbones

The default branch contains the Pi0.5 implementation. Native ports use
separate branches so their dependency stacks and action-expert integrations
remain reproducible.

| Branch | Policy backbone | Contents |
| --- | --- | --- |
| [`main`](https://github.com/amazon-far/imprint/tree/main) | Pi0.5 | Full-attention and recurrent IMPRINT, RMBench, and RoboCasa |
| [`pi0`](https://github.com/amazon-far/imprint/tree/pi0) | Pi0 | Native Pi0 action-expert integration and paper configuration |
| [`molmoact2`](https://github.com/amazon-far/imprint/tree/molmoact2) | MolmoAct2 | Native PyTorch/LeRobot integration, training, and RMBench evaluation |

## 🧭 Choose a Workflow

| Goal | Start here | Environment |
| --- | --- | --- |
| Train Stage 1 or Stage 2 | [Training](#-train-imprint) | Root Python 3.11 environment |
| Evaluate on RMBench | [RMBench evaluation](#-evaluate-on-rmbench) | Root policy env + separate RMBench env |
| Evaluate on RoboCasa | [RoboCasa evaluation](#-evaluate-on-robocasa) | Root policy env + patched RoboCasa 1.0.1 env |
| Inspect published artifacts | [Release artifacts](#-release-artifacts) | Hugging Face CLI |
| Check an installation | [Validation](#-validate-the-installation) | Root environment |

Do not install either simulator into the root environment. The policy and
simulator intentionally run as separate processes so their dependency versions
cannot conflict.

## 📦 Release Artifacts

The canonical public release is hosted by the
[`Amazon-FAR`](https://huggingface.co/Amazon-FAR) Hugging Face organization.

| Benchmark | Policy | Memory encoder | Stage 1 | Stage 2 | Path in `Amazon-FAR/imprint_models` |
| --- | --- | --- | ---: | ---: | --- |
| RMBench | Pi0.5 | Full attention | 160k | 100k | `simulation/rmbench/full_attention/` |
| RMBench | Pi0.5 | Recurrent | 160k | 100k | `simulation/rmbench/recurrent/` |
| RMBench | Pi0 | Full attention | 160k | 100k | `simulation/rmbench/pi0/` |
| RMBench | MolmoAct2 | Full attention | 160k | 100k | `simulation/rmbench/molmoact2/` |
| RoboCasa | Pi0.5 | Full attention | 160k selected | 150k | `simulation/robocasa/full_attention/` |
| RoboCasa | Pi0.5 | Recurrent | 160k selected | 150k | `simulation/robocasa/recurrent/` |

All checkpoints are in
[`Amazon-FAR/imprint_models`](https://huggingface.co/Amazon-FAR/imprint_models).
Datasets and simulator assets are in
[`Amazon-FAR/imprint_datasets`](https://huggingface.co/datasets/Amazon-FAR/imprint_datasets):

| Contents | Path |
| --- | --- |
| RMBench 450-episode training stores | `simulation/rmbench/450/` |
| RoboCasa 600-episode training store | `simulation/robocasa/600/` |
| RoboCasa recorded replay scenes | [`simulation/robocasa/replay/v1/`](https://huggingface.co/datasets/Amazon-FAR/imprint_datasets/tree/6f2e15fe0d28f85a5dfe0b2fbdf49fa1267dab75/simulation/robocasa/replay/v1) |
| MolmoAct2 evaluation metadata and language embeddings | `simulation/rmbench/molmoact2_eval_assets/` |
| Patched RoboCasa evaluation environment | `simulation/robocasa/environment/` |

Machine-readable file counts, byte counts, revisions, and layouts are recorded
in [`artifacts/checkpoints/manifest.yaml`](artifacts/checkpoints/manifest.yaml),
[`artifacts/datasets/manifest.yaml`](artifacts/datasets/manifest.yaml), and
[`artifacts/environments/manifest.yaml`](artifacts/environments/manifest.yaml).

Upstream runtime assets:

- DINOv3 ViT-B/16 runtime:
  [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- WAN2.2 VAE:
  [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)

The training stores are hundreds of gigabytes. For evaluation, download only a
model:

```bash
bash scripts/reproduce/download_release.sh \
  rmbench full_attention --model-only
```

For training, download and extract the corresponding store:

```bash
bash scripts/reproduce/download_release.sh \
  rmbench full_attention --dataset-only
```

Omit the final flag, or use `--all`, to download both.

## 🛠️ Install the Policy and Training Environment

### Requirements

- Linux x86_64
- Python 3.11
- `uv`
- Hugging Face CLI (`hf`)
- NVIDIA driver compatible with CUDA 12.8
- Four 80 GB GPUs for the paper training configurations

RMBench evaluation needs CuRobo. The setup helper compiles it when `nvcc` is
available, or installs a compatible wheel supplied through
`IMPRINT_CUROBO_WHEEL`. Its SAPIEN renderer also requires the NVIDIA
Vulkan/GLX driver inside the container; a compute-only CUDA mount is
insufficient.

```bash
git clone https://github.com/amazon-far/imprint.git
cd imprint
export UV_CACHE_DIR=$PWD/.uv-cache
export OPENPI_DATA_HOME=$PWD/.openpi-cache
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --python 3.11 --frozen
source .venv/bin/activate
python -c "import jax; print(jax.devices())"
```

The repository-local uv and OpenPI caches avoid permission conflicts on shared
GPU machines. You may point either variable at another writable location.
The Stage-1 and Stage-2 launchers otherwise default `OPENPI_DATA_HOME` to a
cache inside the selected output root.

Stage 2 also needs the Pi0.5 base parameters. Obtain them using the OpenPI
checkpoint instructions and record the local `params` directory:

```bash
export IMPRINT_PI05_PARAMS=/path/to/pi05_base/params
```

Evaluation computes recent-frame DINOv3 features online. Stage the exact
ViT-B/16 model once:

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --local-dir artifacts/models/dinov3-vitb16
export DINOV3_MODEL_PATH=$PWD/artifacts/models/dinov3-vitb16
```

The policy also uses the frozen WAN2.2 VAE for causal video tokenization. Stage
the evaluated VAE files from the pinned upstream revision:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  vae/config.json vae/diffusion_pytorch_model.safetensors \
  --revision b8fff7315c768468a5333511427288870b2e9635 \
  --local-dir artifacts/models/wan22-ti2v-5b-local
```

The root lock pins `diffusers==0.38.0`, which is required by this VAE config.

## 🧠 Train IMPRINT

Paper configurations live under:

```text
configs/paper/<rmbench|robocasa>/<full_attention|recurrent>/experiment.env
```

The launchers reject nonempty output directories and write the effective
history configuration beside the checkpoints.

Deterministic GPU execution and seeded Stage-2 history sampling are enabled by
default. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact
flags and the distinction between fresh retraining and the historical paper
checkpoints.

W&B logging is enabled by default. Authenticate before launching, or disable
logging for an offline reproduction:

```bash
export WANDB_API_KEY=...
# Or:
export IMPRINT_WANDB_ENABLED=0
```

### Stage 1: bootstrap memory

RMBench uses separate Stage-1 and Stage-2 store roots:

```bash
export IMPRINT_STAGE1_STORE=$PWD/artifacts/datasets/rmbench/stage1
export IMPRINT_OUTPUT_DIR=/path/to/new/stage1-output

bash scripts/reproduce/train_stage1.sh \
  configs/paper/rmbench/full_attention/experiment.env
```

For RoboCasa, use the shared extracted store:

```bash
export IMPRINT_STAGE1_STORE=$PWD/artifacts/datasets/robocasa/store
export IMPRINT_OUTPUT_DIR=/path/to/new/stage1-output

bash scripts/reproduce/train_stage1.sh \
  configs/paper/robocasa/recurrent/experiment.env
```

### Stage 2: adapt the VLA policy

```bash
export IMPRINT_STAGE1_STORE=$PWD/artifacts/datasets/rmbench/stage1
export IMPRINT_STAGE2_STORE=$PWD/artifacts/datasets/rmbench/stage2
export IMPRINT_STAGE1_CHECKPOINT=/path/to/stage1_step160000.npz
export IMPRINT_PI05_PARAMS=/path/to/pi05_base/params
export IMPRINT_OUTPUT_DIR=/path/to/new/stage2-output

bash scripts/reproduce/train_stage2.sh \
  configs/paper/rmbench/full_attention/experiment.env
```

Use the same benchmark and encoder in both stages. See
[docs/TRAINING.md](docs/TRAINING.md) for the objectives, caps, freeze recipes,
and output layout.

## 🎮 Evaluate on RMBench

### 1. Create the simulator environment

RMBench is distributed as `RoboTwin-Mem` inside the EventVLA repository. Clone
it next to IMPRINT, create its dedicated Python environment, and download its
task assets:

```bash
git clone https://github.com/InternRobotics/EventVLA.git \
  .external/EventVLA
git -C .external/EventVLA checkout \
  a4e02f741f78ab01eb76f6a0bc6685f72f60ed24

export RMBENCH_ROOT=$PWD/.external/EventVLA/RoboTwin-Mem

bash scripts/reproduce/setup_rmbench_eval.sh \
  "$RMBENCH_ROOT" \
  .external/rmbench-env

python scripts/reproduce/download_rmbench_assets.py \
  "$RMBENCH_ROOT" \
  --max-workers 16

export RMBENCH_PYTHON=$PWD/.external/rmbench-env/.venv/bin/python
```

The setup helper explicitly installs `script/requirements.txt` before applying
the upstream SAPIEN/MPLib patches and installing CuRobo v0.7.8. CuRobo builds
for compute capabilities `8.9` and `9.0+PTX` by default; set
`TORCH_CUDA_ARCH_LIST` before running the helper to override that list. Keep
`RMBENCH_PYTHON` separate from the root policy environment. The asset helper
pins the published RMBench dataset revision, downloads in parallel, and
materializes the absolute-path embodiment configurations.

If the renderer machine lacks `nvcc`, build a wheel on a CUDA development
machine with the same Linux architecture, Python 3.10 ABI, PyTorch/CUDA stack,
and target GPU support. Transfer the wheel, then run:

```bash
export IMPRINT_CUROBO_WHEEL=\
/path/to/nvidia_curobo-0.7.8-cp310-cp310-linux_x86_64.whl

bash scripts/reproduce/setup_rmbench_eval.sh \
  "$RMBENCH_ROOT" \
  .external/rmbench-env
```

The helper installs the wheel's non-Torch runtime dependencies and verifies the
result with `uv pip check`; it leaves the simulator's pinned PyTorch/CUDA stack
unchanged. Do not reuse a wheel built for a different Python ABI or an
incompatible CUDA stack.

### 2. Run the mandatory smoke

The launcher starts the policy server, waits for it to become ready, invokes
the canonical evaluator, and cleans up the server. It first checks a CUDA
operation, imports the CuRobo planner, initializes SAPIEN on the selected
simulator GPU, and writes `simulator_preflight.log`:

```bash
python benchmarks/rmbench/eval/run_local.py \
  --rmbench-root "$RMBENCH_ROOT" \
  --checkpoint \
    artifacts/checkpoints/rmbench-full-attention/stage2/step-100000 \
  --dinov3-model "$DINOV3_MODEL_PATH" \
  --architecture full_attention \
  --task cover_blocks \
  --trials 1 \
  --start-seed 100000 \
  --policy-python "$PWD/.venv/bin/python" \
  --sim-python "$RMBENCH_PYTHON" \
  --policy-gpu 0 \
  --sim-gpu 1 \
  --output outputs/rmbench/cover-smoke
```

Inspect `policy.log`, `evaluator.log`, `trials.jsonl`, and `summary.json` in
the output directory. Continue only after this smoke succeeds.

### 3. Run all nine tasks

```bash
for task in \
  battery_try blocks_ranking_try cover_blocks observe_and_pickup \
  press_button put_back_block rearrange_blocks swap_T swap_blocks
do
  python benchmarks/rmbench/eval/run_local.py \
    --rmbench-root "$RMBENCH_ROOT" \
    --checkpoint \
      artifacts/checkpoints/rmbench-full-attention/stage2/step-100000 \
    --dinov3-model "$DINOV3_MODEL_PATH" \
    --architecture full_attention \
    --task "$task" \
    --trials 100 \
    --start-seed 100000 \
    --policy-python "$PWD/.venv/bin/python" \
    --sim-python "$RMBENCH_PYTHON" \
    --output "outputs/rmbench/full-attention/$task"
done
```

The fixed protocol uses three views, BGR policy input, causal previous actions,
`EVAL_CONSOLIDATE_ALL=1`, and 50-step prediction/execution chunks. The display
videos remain RGB.

## 🏠 Evaluate on RoboCasa

RoboCasa requires the archived patched environment. A stock installation does
not contain the memory tasks, replay fixes, legacy scene mapping, or task
success trackers used by the paper.

### 1. Create the simulator environment

```bash
mkdir -p artifacts/environments/robocasa
hf download Amazon-FAR/imprint_datasets \
  simulation/robocasa/environment/robocasa-memory-setup.tar.gz \
  --repo-type dataset \
  --local-dir artifacts/downloads/imprint_datasets

bash scripts/reproduce/setup_robocasa_eval.sh \
  artifacts/downloads/imprint_datasets/simulation/robocasa/environment/robocasa-memory-setup.tar.gz \
  .external/robocasa-memory-setup

export ROBOCASA_ROOT=$PWD/.external/robocasa-memory-setup
export ROBOCASA_PYTHON=$ROBOCASA_ROOT/.venv-mem/bin/python
```

The setup command downloads the official RoboCasa assets (approximately
10 GB), adds the legacy RoboCasa 0.2 assets used by the memory tasks, and
verifies the migrated scene inventory.

### 2. Download the recorded replay scenes

The public replay bundle contains the exact recorded XML/state inputs for all
12 task groups, including the checksum-pinned Retrieve lineage manifest. The
installer pins dataset revision
`6f2e15fe0d28f85a5dfe0b2fbdf49fa1267dab75` and archive SHA256
`ce73132d4bcad7476c730db9bbf0c007346d229d4b0e72ea2e607cb123121166`:

```bash
python scripts/reproduce/setup_robocasa_replay_assets.py \
  --output artifacts/replay/robocasa-v1
export ROBOCASA_REPLAY_ASSETS=$PWD/artifacts/replay/robocasa-v1
```

The installer downloads the archive from `Amazon-FAR/imprint_datasets`,
verifies its SHA256, extracts it without following links, and validates the
recorded-scene inventories and Retrieve source hashes. A successful install
prints `ROBOCASA_REPLAY_ASSETS_READY`. Pass
`--archive /path/to/robocasa-replay-assets-v1.tar` to install an existing
offline download.

The task-specific flags remain available as explicit overrides, but a normal
paper reproduction only needs `--replay-assets-root`.

If the preserved policy environment uses a Transformers build without native
DINOv3 support, pass
`--dinov3-runtime-overlay /path/to/python-package-overlay`. The overlay is
prepended only to the policy process; omit it for a current environment that
already imports DINOv3.

### 3. Run a one-trial smoke

```bash
python benchmarks/robocasa/eval/run_local.py \
  --robocasa-root "$ROBOCASA_ROOT" \
  --checkpoint \
    artifacts/checkpoints/robocasa-full-attention/stage2/step-150000 \
  --dinov3-model "$DINOV3_MODEL_PATH" \
  --architecture full_attention \
  --task-group organize_condiments \
  --pilot-trials 1 \
  --replay-assets-root "$ROBOCASA_REPLAY_ASSETS" \
  --policy-python "$PWD/.venv/bin/python" \
  --sim-python "$ROBOCASA_PYTHON" \
  --policy-gpu 0 \
  --sim-gpu 1 \
  --output outputs/robocasa/organize-condiments-smoke
```

Set `--pilot-trials 0` for the full 50-trial task group. Use a fresh output
directory for every task. The launcher applies EGL rendering and the archived
scene-compatibility patches automatically. It also enforces the paper protocol:
recorded-state replay for Retrieve, no synthetic Wash exploration, and causal
previous-action alignment. The recorded-scene helper defaults to
`$ROBOCASA_ROOT/scripts/replay/replay_memory_demos.py`; override it with
`--memory-replay-helper` only when the customized environment uses another
layout. The launcher preserves the evaluator's raw success fields and also
writes `paper_metrics.json`.

For `sweeten_hot_chocolate`, the paper reports placement completion:
`stable_placement_count == goal_count`. This historical post-processing does
not require the arm to return to its initial pose. The JSON records both the
raw strict result and the paper-compatible result so they cannot be confused.
For every other RoboCasa task, the paper metric equals raw evaluator success.

## ✅ Validate the Installation

Run the release checks after setup or any code change:

```bash
bash scripts/reproduce/check_release.sh

JAX_PLATFORMS=cpu uv run --frozen pytest -q \
  src/mme_vla_suite/tests/cosmo/test_cosmo_buffer.py \
  src/mme_vla_suite/tests/cosmo/test_serving.py \
  benchmarks/rmbench/eval/tests
```

Validate downloaded stores and checkpoints:

```bash
python scripts/reproduce/validate_store.py \
  --feature-store artifacts/datasets/rmbench/stage1 \
  --policy-store artifacts/datasets/rmbench/stage2 \
  --episodes 450 \
  --action-dim 16 \
  --proprio-dim 16

python scripts/reproduce/validate_checkpoint.py \
  --stage1 \
    artifacts/checkpoints/rmbench-full-attention/stage1/stage1_step160000.npz \
  --stage2 \
    artifacts/checkpoints/rmbench-full-attention/stage2/step-100000 \
  --architecture perceiver_v1 \
  --num-latents 256
```

## 🧰 Troubleshooting

| Symptom | Check |
| --- | --- |
| Policy exits before the simulator connects | Read `policy.log`; verify checkpoint metadata and `DINOV3_MODEL_PATH` |
| Recent feature width is wrong | Use DINOv3 ViT-B/16; online features must be 64 tokens/view with width 768 |
| RMBench colors look swapped | Policy input is BGR by protocol; saved videos are RGB |
| RoboCasa task is missing | Use the archived patched environment, not stock RoboCasa |
| RoboCasa reports a missing legacy scene | Rerun setup in a new destination; successful setup prints `ROBOCASA_LEGACY_ASSETS_READY` |
| RoboCasa replay fails immediately | Rerun `setup_robocasa_replay_assets.py` in an empty destination and pass that root with `--replay-assets-root` |
| CUDA out of memory during evaluation | Put policy and simulator on different GPUs |
| RMBench setup says `nvcc` is missing | Use a CUDA development image, or set `IMPRINT_CUROBO_WHEEL` to a compatible Python 3.10 wheel |
| CuRobo or PyTorch does not support the simulator GPU | Use L40S or a graphics-enabled Hopper host with the default upstream stack; B200 requires a separate `sm_100`-compatible PyTorch/CuRobo build |
| `uv` reuses an environment from the container image | Run `UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --python 3.11 --frozen` from the repository root |
| `uv` reports a shared-cache permission error | Retry with `UV_CACHE_DIR=$PWD/.uv-cache UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --python 3.11 --frozen` |
| A launcher refuses an output path | Choose a new empty directory; outputs are immutable by design |

## 🗂️ Repository Layout

```text
src/mme_vla_suite/        memory models, stores, training, and serving
src/openpi/               pinned OpenPI fork used by the experiments
configs/paper/            exact benchmark and architecture profiles
scripts/reproduce/        download, training, validation, and setup entrypoints
benchmarks/rmbench/       RMBench adapter and evaluator launcher
benchmarks/robocasa/      customized RoboCasa evaluator and launcher
envs/                     isolated environment recipes
artifacts/                immutable release manifests and Hub metadata
docs/                     detailed installation and protocol references
```

## 📚 Documentation

- [Installation](docs/INSTALL.md)
- [Training](docs/TRAINING.md)
- [Evaluation](docs/EVALUATION.md)
- [Datasets](docs/DATASETS.md)
- [Checkpoints](docs/CHECKPOINTS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)

## 🔎 Provenance

The original experiment snapshots were archived before cleanup. Run IDs,
checkpoint steps, dimensions, token caps, and freeze recipes are preserved in
`configs/paper/**/provenance.yaml`. Every released checkpoint also includes its
effective history configuration and input provenance.

## 📄 License

Code is provided under the Apache License 2.0 unless a file or dependency states
otherwise. Gemma components remain subject to `LICENSE_GEMMA.txt`. Simulator,
base-model, dataset, and DINOv3 licenses apply independently.
