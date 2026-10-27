# SHMEM / CATLASS training prototype

This is the correctness-first training implementation for Ascend 950. Its
communication uses DeepEP's BF16 SHMEM high-throughput path. Its six training
GEMMs use CATLASS (FP32 accumulation, BF16 activations, FP32 weight-gradient
outputs). It does not call torch.mm or HCCL for candidate data-path GEMMs or
communication. PyTorch executes the pointwise SwiGLU/gating operations.

The current implementation launches communication and compute separately. It
is **not yet a fused training MegaMoE kernel**. Qualification belongs to the
exact source/library/configuration identities in each report, not to a source
checkout or a successful build. See [the separate P1/P2 candidates](../megamoe_training_candidates/README.md)
for their scope and validation requirements. Historical development evidence,
P3-P5 experiments and repeated source snapshots are preserved on the
[`archive/megamoe-training-20261026`](https://github.com/baimangmang3/DeepEP/tree/archive/megamoe-training-20261026/experiments/megamoe_training/development)
branch. They are not part of the training runtime checkout.

## Build and use

Activate the task's official CANN 9.2 and SHMEM environment, then run from the
DeepEP repository root:

```bash
cmake -S experiments/megamoe_training -B experiments/megamoe_training/build
cmake --build experiments/megamoe_training/build
export PYTHONPATH="$PWD/experiments/megamoe_training${PYTHONPATH:+:$PYTHONPATH}"
```

The current device code targets dav-3510, uses the repository's pinned CATLASS
headers, and is separate from the existing inference extension. ctypes loads
the built library without a Torch C++ ABI dependency. No fallback silently
replaces native CATLASS computation.

```python
from megamoe_training import TrainingMoE

# dist is an initialized EP process group; tensors reside on this rank's NPU.
with TrainingMoE(group=dist.group.WORLD, max_tokens=B, hidden=H,
                 topk=K, num_experts=E) as moe:
    y = moe(x, w1, w2, expert_ids, selected_weights)
    y.backward(dy)
    dw1_fp32, dw2_fp32 = moe.last_grads_fp32
```

Shapes: x[B,H], w1[E/EP,2I,H], w2[E/EP,H,I], expert_ids[B,K] int64,
selected_weights[B,K] FP32. x and compute operands are BF16. Parameter storage
may be BF16 or FP32. FP32 master parameters retain FP32 `.grad`; BF16 leaf
parameters receive BF16 `.grad` and expose the FP32 intermediate separately.
An optional local shared expert takes shared_w1[2I,H], shared_w2[H,I].

Selected weights receive `dp = sum(dY * Z)` using unweighted rounded forward
Z, including when p=0. Routing indices are nondifferentiable; a differentiable
router outside the operator receives dp through its usual PyTorch graph.
The backward path reuses the exact forward routing handle to dispatch dY and
returns dx and dp via SHMEM. Second derivatives are explicitly unsupported.

## Validation contract

1. Validate standalone CATLASS NN/NT/TN and FP32 wgrad, including tails/zero K.
2. EP2 and EP16 small full-gradient cases: balanced cross-server routes,
   zero gate, empty experts, skew, FP32 master parameters, shared expert.
3. EP16 B8192 H6144 I2048 E64 K8. Each rank receives exactly 65536 rows;
   each of its four experts receives 16384. This preserves the supplied
   profiling shapes; E64 here is not the complete GLM model's E256.
4. Only after all-rank accuracy passes, measure synchronized forward and full
   forward/backward separately, retaining individual rank samples.
5. Compare the native CANN three-operator chain with identical parameters,
   routes, gate placement and BF16 rounding. Confirm the actual backend and
   profiler evidence; unsupported AIV must not be reported as CCU timing.

See tests/README.md for the independent HCCL oracle and fixed error thresholds.
The test-only CANN training baseline composes the two fused CANN operators in
forward and dgrad and reuses CATLASS for both FP32 wgrads. Its exported-input
and nontransposed-weight paths require separate Ascend950 qualification. It
is explicit custom autograd, not a claim that the official operators register
backward. Each backend has separate full-gradient reports; forward performance
requires the complete candidate qualification reports for the exact shape.
Native combine rounds server partial sums to BF16; the independent oracle
uses a flat FP32 mathematical reduction. Their differing reduction trees are
explicitly recorded. Source B must be 1..16384, H<=7168 and BF16-aligned, K<=16,
and the EP group must follow the existing ElasticBuffer topology constraints.
Zero received experts are covered; zero source tokens are not supported by
the current communication contract. Destroy the buffer collectively after
all its forwards and backwards, and never close it with a pending backward.

## Next implementation stage

After correctness, replace per-expert host launches with a grouped device
scheduler; fuse SwiGLU and its gradient where the BF16 boundaries permit;
then pipeline chunk-ready SHMEM transfers with CATLASS tiles. Backward wgrad
must retain FP32 accumulation and exclusive tile ownership. Do not relax
ready/epoch or remote completion protocols to obtain an apparent speedup.
