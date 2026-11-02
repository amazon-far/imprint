# RMBench Evaluation Protocol

This protocol applies to the released causal Pi0.5 memory-finetuned
checkpoints. The policy and simulator use separate Python environments and may
run on separate GPUs.

## Fixed Contract

- camera order: `head_camera`, `left_camera`, `right_camera`
- policy image color order: `bgr`
- visual memory stream: DINOv3
- prediction horizon: 50
- execution horizon: 50
- strict previous-action causality
- memory action alignment: `causal_raw_previous` (released evaluator default).
  Each kept memory frame stores the action executed one raw step earlier. The
  offline variant files (`action_causal_v*.npy`) store the action executed at
  the previous kept frame (paper Eq. A.1). Select that rule with
  `--memory-action-alignment previous_kept_frame`. `run_local.py` records the
  value in `policy_server.yaml`.
- online recent-frame features
- `EVAL_CONSOLIDATE_ALL=1`
- `MEMFT_LEGACY_EMPTY_RECENT=0`
- first evaluation seed: 100000
- accepted trials per task: 100

The BGR policy conversion is independent of display-video encoding. Recorded
videos are RGB and must not be used to infer the policy input convention.

## Smoke Gate

Run `cover_blocks` seed 100000 before a full evaluation:

```bash
python benchmarks/rmbench/eval/run_local.py \
  --rmbench-root /path/to/RMBench \
  --checkpoint /path/to/checkpoint \
  --dinov3-model /path/to/dinov3-vitb16 \
  --architecture full_attention \
  --task cover_blocks \
  --trials 1 \
  --start-seed 100000 \
  --output outputs/rmbench/cover-smoke
```

Do not start the full suite unless the smoke result is successful and the
policy log identifies the intended checkpoint, architecture, visual stream,
camera order, color order, and horizon.

## Full Evaluation

Repeat the command for each of the nine task names listed by
`run_local.py --help`, using `--trials 100`. Use a distinct empty output
directory for each task. The task-balanced score is the arithmetic mean of the
nine per-task success rates.

Retain the evaluator arguments, policy and simulator logs, per-trial JSON,
checkpoint provenance, and videos. A trial is successful only under the
benchmark's terminal predicate; manual failure categories do not alter the
primary denominator.
