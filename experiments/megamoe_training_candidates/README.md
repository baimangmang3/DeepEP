# Separate training candidates

These candidates build on the P0 implementation in `../megamoe_training`:

| Package | Scope |
| --- | --- |
| `p1_pointwise` | Native SwiGLU, its gradient and row gating; communication and CATLASS GEMMs remain separate. |
| `p2_backward_overlap` | P1 plus explicit serial/overlap schedules for weight gradients and the return of input/router gradients. This is not a fused communication/GMM kernel. |

The packages are siblings of P0 so their explicit source roots remain disjoint.
Runtime/native code and fixed numerical thresholds retain their original bytes.
Only the retained CPU tests' repository-relative paths were updated.

## Build and import

From the repository root with the qualified CANN/SHMEM environment active:

```bash
cmake -S experiments/megamoe_training_candidates/p1_pointwise -B build/megamoe_pointwise
cmake --build build/megamoe_pointwise
export PYTHONPATH="$PWD/experiments/megamoe_training:$PWD/experiments/megamoe_training_candidates${PYTHONPATH:+:$PYTHONPATH}"
```

P0's native GEMM library must also be built as described in its README.
`p1_pointwise.validate_training_p1.py` and
`p2_backward_overlap.validate_training_p2.py` accept explicit P0/P1 source roots,
GEMM/pointwise library paths and output directories. Run the files directly with
`--help`; their default action prints a plan without importing Torch or starting
NPU work. P2 requires an explicit `--backward-schedule serial` or `overlap`.
Validate the standalone pointwise operator before full-layer P1 qualification.

For the retained device-free P2 gates:

```bash
python -I experiments/megamoe_training_candidates/p2_backward_overlap/tests/test_qualification_cpu.py
python -I experiments/megamoe_training_candidates/p2_backward_overlap/tests/test_lifecycle_cpu.py
```

Generated test reports are not source and should not be committed. CPU checks
are not NPU precision, overlap or performance qualification. Existing reports
do not qualify a changed source, library or shape. No EP64 accuracy/performance
result is claimed by this export.

## Archive

The original development export, P3-P5 experiments, EP16 adaptation exploration,
historical diagnostics/reports, CPU math fixtures and duplicated source
snapshots remain on
[`archive/megamoe-training-20261026`](https://github.com/baimangmang3/DeepEP/tree/archive/megamoe-training-20261026/experiments/megamoe_training/development)
at commit `deb68640f7b239c4f43d2dfe0b9c19cadaa3b697`. Its historical controllers
are evidence, not current permission to launch a job. No binaries or credentials
are added by this reorganization.
