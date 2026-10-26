# Training

## Data Contract

The logical training dataset contains:

- `data/<sample>.pkl`: current policy observation, action target, task prompt,
  episode ID, and step ID.
- `features/episode_<id>/dinov3.npy`: `(T, 3, 64, 768)` pooled DINOv3 features.
- `features/episode_<id>/latents_v*.npy`: eight offline visual-history variants.
- `features/episode_<id>/action_proprio.npy`: raw action and proprio streams.
- `features/episode_<id>/action_causal_v*.npy`: variant-matched causal actions.
- `meta/`: sample index, normalization statistics, and immutable build manifests.

RoboCasa keeps these fields in one shared store. RMBench uses two immutable
roots: `stage1/` owns `features/` and the Stage-1 sample index, while `stage2/`
owns the horizon-50 policy samples. The Stage-2 launcher patches
`feature_store_path` to the Stage-1 root at launch time.

Both training launchers enable W&B logging by default. Set `WANDB_API_KEY` (or
log in with the W&B CLI) before starting a run. Set
`IMPRINT_WANDB_ENABLED=0` to run without W&B; this does not change the training
configuration or checkpoint contents.

Release retraining is deterministic by default. The launchers seed Python,
NumPy, Torch workers, and the Stage-2 variant sampler, and enable deterministic
XLA GPU operations with a deterministic cuBLAS workspace. Keep
`IMPRINT_DETERMINISTIC_TRAINING=1` when comparing independent retrainings.
Setting it to `0` restores the faster historical execution semantics but does
not reproduce a particular historical checkpoint.

At history step `t`, the action memory receives the previously executed action.
The first causal action is zero and subsequent entries are shifted by one:
`action_causal[t] = action[t-1]`. Variant actions must be sampled with the same
frame IDs as their corresponding visual variant.

### Wan chunk alignment

The memory reads Wan latent `c` as kept frames `4c` to `4c+3`. The causal
Wan2.2 VAE encodes the first frame alone, so in the released stores (the
`native` alignment) latent `c >= 1` holds kept frames `4c-3` to `4c`. After `k`
chunks, kept frames `4k-3` to `4k-1` are in neither the latent stream nor the
recent window, and recurrent chunk event `c` pairs these latents with the
actions and states of kept frames `4c` to `4c+3`. The released checkpoints
were trained this way, so `native` stays the default.

To train with exact alignment, set
`cosmo_bottleneck_memory.wan_chunk_alignment: replicate_first_frame`. The
frontend then encodes a copy of frame 0 first and drops its latent. Build a
derived store with
`python -m mme_vla_suite.dataset_builder.reencode_wan_latents` and retrain
both stages. This tool supports only the RMBench store layout: `frames.npy` at
the Wan input size in each episode, and latents pooled per view on disk. The
online (`latent_source: online`) path supports only `native`.

The store, the Stage 1 checkpoint (`metadata/wan_chunk_alignment`), the
`precompute_mem.py` output and the Stage 2 run directory
(`wan_chunk_alignment.txt`) record the alignment. A missing record means
`native`. Training, precompute and policy load refuse a record that differs
from the config. For RMBench evaluation, pass the same value to
`benchmarks/rmbench/eval/run_local.py --wan-chunk-alignment`.

## Stage 1

Stage 1 trains only the memory encoder and reconstruction decoder. The VLA is
absent. Each update samples historical query locations and reconstructs:

- DINOv3 target features, weight `1.0`
- causal action, weight `0.5`
- proprioceptive state, weight `0.5`
- normalized progress, weight `0.5`

The paper profiles use AdamW, peak learning rate `1e-4`, weight decay `1e-4`,
1,000 warmup updates, cosine decay, batch size 64, seed 0, 16 query samples,
three views, and 256 memory tokens of width 1,024.

The history-length sampling schedule is part of the paper recipe:

| Profile | Stage-1 latent buckets | Length-bucketed batches |
| --- | --- | --- |
| RMBench full attention | `4096` | no |
| RMBench recurrent | `256,512,1024,2048,4096` | yes |
| RoboCasa full attention | `256,512,1024,2048,4096` | no |
| RoboCasa recurrent | `256,512,1024,2048,4096` | yes |

Changing this schedule changes the Stage-1 training distribution and should be
treated as a new experiment rather than an equivalent reproduction.

```bash
IMPRINT_STAGE1_STORE=/path/to/store/stage1 \
IMPRINT_OUTPUT_DIR=/path/to/checkpoints \
bash scripts/reproduce/train_stage1.sh \
  configs/paper/rmbench/recurrent/experiment.env
```

RMBench trains for 160k updates. RoboCasa trains for 180k updates and uses the
160k checkpoint for Stage 2, matching the reported experiments.

## Stage 2

Stage 2 loads `pi05_base`, transfers the Stage-1 memory encoder, discards the
Stage-1 decoder, and optimizes the diffusion action objective. The memory
encoder is finetuned in all four canonical profiles.

Shared settings are batch size 64, seed 42, EMA 0.999, 500 warmup updates,
peak learning rate `2.5e-5`, cosine decay to `2.5e-6`, and four data workers.
Worker-local Python and NumPy streams are derived from Torch's seeded worker
stream, so offline history-variant selection is repeatable across launches.

| Profile | Horizon | Updates | FSDP | Frozen |
| --- | ---: | ---: | ---: | --- |
| RMBench full attention | 50 | 100k | 4 | image tower + language embedding table |
| RMBench recurrent | 50 | 100k | 4 | image tower |
| RoboCasa full attention | 20 | 150k | 1 | image tower |
| RoboCasa recurrent | 20 | 150k | 4 | image tower |

```bash
IMPRINT_STAGE1_STORE=/path/to/store/stage1 \
IMPRINT_STAGE2_STORE=/path/to/store/stage2 \
IMPRINT_STAGE1_CHECKPOINT=/path/to/stage1_step160000.npz \
IMPRINT_PI05_PARAMS=/path/to/pi05_base/params \
IMPRINT_OUTPUT_DIR=/path/to/checkpoints \
bash scripts/reproduce/train_stage2.sh \
  configs/paper/robocasa/full_attention/experiment.env
```

The entrypoint writes Orbax parameter checkpoints. Publish `params`, checkpoint
metadata, normalization assets, the history YAML, and provenance; `train_state`
is not required for evaluation.
