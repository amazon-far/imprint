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

At history step `t`, the action memory receives a previously executed action.
Two streams exist:

- `action_proprio.npy` key `action_causal` is the canonical stream at every raw
  step: zero first, then `action_causal[t] = action[t-1]`. The online Stage-1
  path and the loader fallback for stores without variant files read it at the
  kept frames.
- `action_causal_v*.npy` belongs to visual variant `v*` and its frame IDs
  `f_0, ..., f_n`: zero first, then the action executed at the previous kept
  frame, `action[f_(j-1)]` (paper Eq. A.1). The offline Stage-1 and Stage-2
  loaders read these files. Do not index the canonical stream at skipped frame
  IDs to rebuild them.

The RMBench evaluator stores the action executed one raw step earlier by
default (`causal_raw_previous`). With kept frames two raw steps apart, this
differs from the variant files. `run_local.py --memory-action-alignment
previous_kept_frame` applies the variant-file rule at evaluation.

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
