# Qualified full-training totals and traces

This independent harness does not modify the frozen P0 or its qualification identity. `TRAINING_SOURCE_ROOT` must name the qualified `experiments/megamoe_training` directory, including its tests and native-library configuration. All16 rank JSONs from each backend must be collected into that backend's own qualification directory on both hosts. Each process checks current implementation/runtime identity, exact world/shape/seed/BF16/shared/backend, fixed limits, full-gradient element counts and finite statistics, balanced route counts and intermediate shapes. Rank agreement also binds harness SHA and execution arguments. Missing/stale/incompatible evidence stops every rank before workloads.

Default is **gate only**. Formal `--benchmark`, `--profile`, and `--debug-steps N` are mutually exclusive runs. Both routed backends reuse the exact same X/weights/IDs/gates/dY objects; every step resets leaf gradients and calls forward then backward. There is no reference recomputation in the measured step and no `no_grad`. Shared is P0-only and requires corresponding BF16/shared qualification; its generator matches the frozen validator. No CANN shared claim is made.

Example arguments for the environment agent's existing two-host torchrun launcher (do not launch before reserving devices):

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --rdzv-backend=static --master-addr=177.22.27.1 --master-port="$MASTER_PORT" \
  /ABS/training_perf_probe/probe.py \
  --B 8192 --H 6144 --I 2048 --E 64 --K 8 --seed 4317 \
  --backends shmem_catlass,cann_ai_cpu \
  --qualification shmem_catlass=/ABS/QUALIFIED_P0_ALL16 \
  --qualification cann_ai_cpu=/ABS/QUALIFIED_CANN_ALL16 \
  --benchmark --warmup 5 --iters 30 --output /ABS/NEW_PERF_RUN
```

Use NODE_RANK0 on177.22.27.1 and NODE_RANK1 on177.22.28.1 with the environment agent's reserved MASTER_PORT. For shared, use only `--backends shmem_catlass --shared` and its own shared qualification directory. For profiling, replace `--benchmark` with `--profile --profile-steps 3` and use a separate output directory. For qualification-only inspection omit all mode flags.

Formal totals call the frozen `benchmark_compare.timing_round_robin` unchanged, in a grad-enabled context. Backend order rotates each iteration; barrier and prior synchronize are outside each interval, event objects are created before wall-clock starts. Current-stream NPU events may include idle/communication waits; synchronized wall time includes gradient reset, Python call/output creation, forward/backward and device work. Reports retain every rank/iteration and max-rank per iteration. Profile-only steps carry `record_function` labels and emit per-rank CPU/NPU traces; benchmark steps have no extra labels. These are full-training totals, not comm-only timing or proof of overlap. Effective GEMM work is `6*B*K*H*I` forward and `18*B*K*H*I` full train per rank, plus shared with K=1 when requested; no unverified peak MFU/HFU is reported.

Deferred diagnostics, **not enabled by this harness**: `deep_ep/buffers/elastic.py:928` exposes optional dispatch `profile[64,32]` / `detail_profile[64,32]`, detail mode1 counters/mode2 timing; `dispatch/profile.hpp` includes WQE/doorbell/quiet/peer-wait measurements, and combine has profile/chunk/peer hooks. Production kernels have separate uninstrumented instances; diagnostics select profile kernels. A `GetSystemCycle` comment of1000ticks/us is not calibration for this run. Calibrate the clock and quantify instrumentation perturbation before deriving device durations or bandwidth. No extra device counters or bandwidth formulas are implemented here.

CPU gate checks: `python -m unittest discover -s /ABS/training_perf_probe -p test_gate.py -v`.
