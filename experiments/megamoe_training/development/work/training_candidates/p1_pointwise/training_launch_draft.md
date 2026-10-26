Draft only: no command has been launched. Environment agent must reserve devices,
obtain standalone success and separate full-layer authorization before execution.
Run one stage at a time; review every rank and cleanup before advancing. No timing.

Verified base layout from `outputs/training_results_20261025_0745/full-training-profile/rack05os7/perf_rank0.json`
and frozen provenance's source roles: the supplied freeze directory is a repository
root; **TRAINING_SOURCE_ROOT includes `/experiments/megamoe_training`**. Its children
must include `tests/provenance.py`, `tests/validate_training_npu.py`, `native/`, and
`megamoe_training/`. The candidate package directory must literally be
`p1_pointwise` with `__init__.py`, adapter and all eight bound files. Deploy it to
the proposed new path below on both hosts, outside the frozen repository.
The existing standalone `candidate-p1` directory alone is not an importable
`p1_pointwise` package. Keep its qualified SO at the existing location; no rebuild
or rename of frozen P0 is required. All source files copied into the new package
must match current manifests (including corrected CPP ee58581a...).

Environment on each participating host, using its reserved venv/CANN/SHMEM setup:

```bash
source /home/zzy/deploy/megamoe-20261024/training/activate-qualified-p0.sh
export TRAINING_SOURCE_ROOT=/home/zzy/deploy/megamoe-20261024/training/qualified-p0-20261025/experiments/megamoe_training
export CANDIDATE_SOURCE_ROOT=/home/zzy/deploy/megamoe-20261024/training/candidate-p1-training/p1_pointwise
export POINTWISE_SO=/home/zzy/deploy/megamoe-20261024/training/candidate-p1/build/libmegamoe_pointwise.so
# Environment agent supplies a fresh run root and a reserved port for each stage.
: "${RUN_ROOT:?fresh absolute output root required}"
: "${MASTER_PORT:?reserved rendezvous port required}"
test -f "$TRAINING_SOURCE_ROOT/tests/provenance.py"
test -f "$TRAINING_SOURCE_ROOT/megamoe_training/training.py"
test -f "$CANDIDATE_SOURCE_ROOT/__init__.py"
test "$(basename "$CANDIDATE_SOURCE_ROOT")" = p1_pointwise
test -f "$POINTWISE_SO"
COMMON=(--run-npu --training-source-root "$TRAINING_SOURCE_ROOT"
  --candidate-source-root "$CANDIDATE_SOURCE_ROOT" --pointwise-library "$POINTWISE_SO"
  --seed 4317 --oracle-chunk 1024 --max-reference-gib 12)
```

The activation supplies the frozen CATLASS library and DeepEP import paths; retain
them. `CatlassGemm()` resolves `MEGAMOE_TRAINING_LIBRARY` or frozen
`experiments/megamoe_training/build/libmegamoe_training.so`. The validator inserts
the frozen experiment/tests and candidate **parent** import roots itself. It rejects
wrong module paths, overlapping roots and different per-rank content/config identities.

EP2, one reserved host, two processes. These are separate invocations (distinct
outputs); do not run the second until the first has passed and released resources:

```bash
torchrun --nnodes=1 --nproc-per-node=2 --node-rank=0 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port="$MASTER_PORT" \
  "$CANDIDATE_SOURCE_ROOT/validate_training_p1.py" "${COMMON[@]}" \
  --world-size 2 --B 32 --H 128 --I 128 --E 8 --K 4 \
  --cases balanced,zero_gate,empty,skew --weight-dtype bf16 --output "$RUN_ROOT/ep2-bf16"

torchrun --nnodes=1 --nproc-per-node=2 --node-rank=0 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port="$MASTER_PORT" \
  "$CANDIDATE_SOURCE_ROOT/validate_training_p1.py" "${COMMON[@]}" \
  --world-size 2 --B 32 --H 128 --I 128 --E 8 --K 4 \
  --cases balanced,zero_gate,empty,skew --weight-dtype fp32 --shared --output "$RUN_ROOT/ep2-shared-fp32"
```

EP16, two hosts: NODE_RANK=0 on177.22.27.1 and NODE_RANK=1 on177.22.28.1.
Run each command on both hosts with the same reserved port/stage/roots. Each host
keeps its rank0..7 or rank8..15 JSONs; environment must collect all16 after success.
Small cross-host cases, first BF16 then shared+FP32:

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" --rdzv-backend=static \
  --master-addr=177.22.27.1 --master-port="$MASTER_PORT" \
  "$CANDIDATE_SOURCE_ROOT/validate_training_p1.py" "${COMMON[@]}" \
  --world-size 16 --B 32 --H 128 --I 128 --E 64 --K 8 \
  --cases balanced,zero_gate,empty,skew --weight-dtype bf16 --output "$RUN_ROOT/ep16-bf16"

torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" --rdzv-backend=static \
  --master-addr=177.22.27.1 --master-port="$MASTER_PORT" \
  "$CANDIDATE_SOURCE_ROOT/validate_training_p1.py" "${COMMON[@]}" \
  --world-size 16 --B 32 --H 128 --I 128 --E 64 --K 8 \
  --cases balanced,zero_gate,empty,skew --weight-dtype fp32 --shared --output "$RUN_ROOT/ep16-shared-fp32"
```

Main EP16, balanced only, first routed BF16 then shared BF16. Under this exact
route each rank receives65536 rows, four experts each16384; full GLM E256 is a
different configuration. The shared run remains accuracy-only with the same12GiB
reference-reservation cap; native/framework/workspace memory is additional.

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" --rdzv-backend=static \
  --master-addr=177.22.27.1 --master-port="$MASTER_PORT" \
  "$CANDIDATE_SOURCE_ROOT/validate_training_p1.py" "${COMMON[@]}" \
  --world-size 16 --B 8192 --H 6144 --I 2048 --E 64 --K 8 \
  --cases balanced --weight-dtype bf16 --output "$RUN_ROOT/ep16-main-bf16"

torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" --rdzv-backend=static \
  --master-addr=177.22.27.1 --master-port="$MASTER_PORT" \
  "$CANDIDATE_SOURCE_ROOT/validate_training_p1.py" "${COMMON[@]}" \
  --world-size 16 --B 8192 --H 6144 --I 2048 --E 64 --K 8 \
  --cases balanced --weight-dtype bf16 --shared --output "$RUN_ROOT/ep16-main-shared-bf16"
```

All stages require every rank `all_passed=true`, correct counts/shapes/dtypes and
all gradient metrics. P0/standalone reports cannot substitute for P1 full-layer
qualification. Preserve node return codes and actual cleanup status; a partial set
of rank files is insufficient. These commands contain no benchmark/profiler flags.
