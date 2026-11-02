# RMBench

RMBench evaluation uses the benchmark's official task environments and policy
client. IMPRINT supplies a raw-TCP policy adapter and a recording wrapper around
the canonical evaluator.

The paper protocol is:

- nine tasks
- 100 trials per task
- seeds beginning at 100000
- prediction horizon 50
- execution horizon 50
- strict causal action history
- online recent DINOv3 features
- `EVAL_CONSOLIDATE_ALL=1`
- memory action alignment `causal_raw_previous` (released evaluator default;
  the offline variant files use the action executed at the previous kept
  frame, paper Eq. A.1; see `--memory-action-alignment`)
- policy input color order `bgr`

Install RMBench separately, then run a smoke:

```bash
python benchmarks/rmbench/eval/run_local.py \
  --rmbench-root /path/to/RMBench \
  --checkpoint artifacts/checkpoints/rmbench-full-attention/stage2/step-100000 \
  --dinov3-model /path/to/dinov3-vitb16 \
  --architecture full_attention \
  --task cover_blocks \
  --trials 1 \
  --output outputs/rmbench/cover-smoke
```

The policy process defaults to GPU 0 and the simulator to GPU 1. Use
`--policy-gpu` and `--sim-gpu` to change the assignment. A successful smoke is
required before the full task suite.

The official RMBench repository and its renderer assets are not vendored here.
