# TrainingMoE NPU validation

This harness uses independent HCCL all-gather and manually specified forward/backward math. It does not reuse candidate SHMEM handles, dispatch, combine or autograd. It must run only on the environment agent's reserved devices after the module/native build is ready.

The device oracle explicitly rounds G/U, A, Z, p*Z, p*D, dA, dGU and each expert dX to BF16. It reduces routed Y/dX in FP32 and casts the final sum to BF16. Native multi-server combine additionally rounds each server partial to BF16 before the source sum, so this independent mathematical reference does not have the identical reduction tree/cast locations. Fixed thresholds remain unchanged. dW uses FP32 matmul and accumulation with BF16 operands; dP uses the saved, rounded forward Z. Chunked reference dW accumulation can differ in last FP32 bits from a single grouped GEMM. HF32 is explicitly disabled using `torch.npu.matmul.allow_hf32=False`; internal format is disabled. RMS uses device FP32 chunk reductions and host float64 scalar accumulation, requiring no NPU float64 support and no full host tensor copies.

Every rank writes `rankN.json`, containing full-tensor finite checks, element count, max absolute/relative error, RMS error, reference RMS, failed-element count and fixed limits for Y, dX, dP and FP32 dW1/dW2. Leaf weight gradients are also compared after the matching dtype cast. All requested cases must pass on all ranks before benchmark begins. Limits are fixed at absolute 0.003, relative 0.03, RMS-relative 0.02; they must not be loosened to make a failing implementation pass. Relative error uses a 1e-8 denominator floor for reporting. An identically zero reference uses absolute RMS tolerance.

Deploy only this directory beside the module. Run from the remote repository root with the deployment agent's existing CANN/SHMEM activation and host/rank configuration. The examples show one host; EP16 across hosts requires the environment agent's existing torchrun rendezvous parameters, not a second independent launch.

```bash
# EP2: small full-gradient oracle and all edge cases.
torchrun --standalone --nproc-per-node=2 experiments/megamoe_training/tests/validate_training_npu.py \
  --B 32 --H 128 --I 128 --E 8 --K 4 \
  --cases balanced,zero_gate,empty,skew --output /ABS/RESERVED_RUN/ep2-small

# EP16: run on BOTH hosts; export NODE_RANK=0 on 177.22.27.1 and
# NODE_RANK=1 on 177.22.28.1. Environment agent must supply MASTER_PORT.
# Small full-gradient oracle. Repeat --weight-dtype fp32 separately.
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --rdzv-backend=static --master-addr=177.22.27.1 --master-port="$MASTER_PORT" \
  experiments/megamoe_training/tests/validate_training_npu.py \
  --B 32 --H 128 --I 128 --E 64 --K 8 \
  --cases balanced,zero_gate,empty,skew --output /ABS/RESERVED_RUN/ep16-small

# Final EP16 user-shape case: only after the small cases pass.
# E64 means four experts/rank; this is not full GLM E256.
# Balanced construction gives exactly 65536 received rows/rank, 16384/expert.
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --rdzv-backend=static --master-addr=177.22.27.1 --master-port="$MASTER_PORT" \
  experiments/megamoe_training/tests/validate_training_npu.py \
  --B 8192 --H 6144 --I 2048 --E 64 --K 8 --cases balanced \
  --oracle-chunk 1024 --max-reference-gib 12 --benchmark --warmup 5 --iters 30 \
  --output /ABS/RESERVED_RUN/ep16-user-shape

# Small independent unweighted shared expert, including every shared gradient.
torchrun --standalone --nproc-per-node=2 experiments/megamoe_training/tests/validate_training_npu.py \
  --B 32 --H 128 --I 128 --E 8 --K 4 --shared --weight-dtype fp32 \
  --cases balanced,zero_gate,empty,skew --output /ABS/RESERVED_RUN/ep2-shared-fp32
```

Balanced routing uses `idx=(token+rank*B+slot*(E//K))%E` when K divides E evenly (`E%K==0`), spreading a token across rank/server ownership. The final E64/K8 shape reaches eight ranks and, with contiguous eight-rank host ownership, both servers, while preserving exact balance. For nondivisible E/K the fallback uses consecutive expert assignments.

The large oracle has substantial reference cost: it gathers BF16 X/dY and FP32 gates/int64 IDs across ranks and processes local experts in chunks. The cap is a conservative estimate, not a guarantee about NPU allocator or GEMM workspace. No source-token-zero case is requested because current SHMEM high-throughput routing restricts that case. Source B is always checked against the HT limit 16384. Skew can have much larger expert counts than balanced routing; run it only with small shapes. No benchmark numbers are emitted before precision passes. Timings are synchronized full forward/backward wall-clock, five warmups and 30 measured iterations by default, with every rank and per-iteration max-rank values retained. Device-only kernel attribution requires a separate profiler.

Local authoring validation is Python syntax only when PyTorch/NPU is unavailable; a prior NumPy float64 mathematical check is separate evidence and does not validate this BF16 NPU contract.

`benchmark_compare.py` prepares the CANN ai_cpu/ccu versus P0 SHMEM+CATLASS **forward-only** comparison. Before either `--benchmark` or `--profile`, it requires `--training-qualification-dir /ABS/FULL_GRADIENT_QUALIFICATION`. That directory must contain every `rank0.json` through `rank15.json` for EP16, copied onto both hosts before starting. Each report must have `all_passed=true`, matching rank/world and exact B/H/I/E/K, `weight_dtype="torch.bfloat16"`, and a balanced case with `all_ranks_numerical_pass=true`. Qualification file hashes are recorded. Two host-local sets of eight files are insufficient: the environment agent must collect all16 and distribute the combined directory to each host. Pure forward precision runs do not require this argument. Forward Y passing alone cannot authorize performance or profiling.

The comparison's main timing scope refreshes routing plans on every call for both paths. Optional CANN prepared-plan and prepared-input operator measurements are separate scopes and cannot support a speedup claim against a fresh-routing candidate. Use BF16 parameters for performance; FP32 master-weight checks are precision-only. The separate full-training baseline below remains unqualified on hardware.

`validate_training_npu.py --backend shmem_catlass|cann_ai_cpu|cann_ccu` now selects full-gradient training backends against the same independent oracle. `cann_training.py` prepares a CANN training baseline with fused communication/GMMs in forward and dgrad, separate pack/unpack/activation/topk, and the **same CATLASS FP32 wgrad** as the candidate. It has no torch.mm production fallback. Exported X_e/dZ_e is tuple element **2** (the third output); tuple element1 is optional shared-mm output and is absent. CANN `trans_gmm_weight=False` and `permute_out_flag=True` on950 still require hardware qualification. `--shared` is candidate-only. Use separate output directories for each backend, for example `--backend cann_ai_cpu --output /ABS/RESERVED_RUN/cann-ai-cpu-small`. No backend is considered supported from syntax alone.

Every full-gradient JSON now contains `backend`. Forward comparison's training qualification accepts **only** `backend="shmem_catlass"`; old unlabelled reports or passing CANN reports cannot qualify the candidate. Do not mix backend directories. Full-training timing from `validate_training_npu.py --benchmark` remains separately gated by that backend's complete numerical run; it cannot be substituted for forward-only comparison timing.

Review revision: each validator run now records a content-bound `identity`; the comparison builds the current identity before accepting any full-gradient qualification. Stable logical roles cover the training/gemm Python implementation, native `gemm.cpp`, CMake, package init, validator/oracle/limits, CANN wrappers/oracle, comparison and provenance helper; every Python file in the actually imported DeepEP package; actual loaded CATLASS and DeepEP core `.so`; SHMEM selected by `core.shmem_library_path()` (`dladdr`) and verified against `/proc/self/maps`; actual mapped CANN ACL/runtime/opapi and relevant dependency contents; Torch/PTA versions and read-back precision switches. No SDK path or library name guess substitutes for a missing actual mapping. Missing files/options/maps, a deleted mapped library, or ambiguous same-name/different-content libraries fail closed. Collection loads libraries only and creates no new ElasticBuffer/communication domain; failures are exchanged on the existing process group before any rank exits.

Absolute paths remain diagnostic and are excluded from the content digest. All rank reports must match the current content digest. Old reports lacking `identity`, stale source/native binaries, different CANN/SHMEM/framework contents, or changed oracle/limits cannot authorize timing. Equal contents under different host directory prefixes remain valid. Performance reports retain qualification report SHA256 and identity digest. This review version changes only tests; keep previous artifact packages immutable and create a new review artifact rather than replacing an old snapshot.

CPU loader inspection found the formal9.2 `libopapi.so` primary library can be lazy even when `libopapi_transformer.so` and other operator libraries are already mapped. For a required unmapped primary, the collector explicitly loads only `ASCEND_HOME_PATH/lib64/<required-library>` and verifies its resolved file remains inside that formal root and is actually mapped. The handle stays alive. Missing formal path/load/mapping still rejects qualification. All already mapped `libopapi*` library contents are also retained. Explicitly mapping the primary does **not** prove that a specific operator executes from it; backend execution attribution still requires runtime/profiler evidence.

CPU-only regression, requiring no Torch/NPU/SDK or new package install:

```bash
python -m unittest discover -s experiments/megamoe_training/tests -p test_provenance.py -v
```
