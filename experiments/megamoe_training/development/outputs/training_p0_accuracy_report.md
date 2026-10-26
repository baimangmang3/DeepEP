# P0 NPU training accuracy

Two hosts,16 Ascend950DT NPUs. Frozen SHMEM+CATLASS P0 passed complete forward/backward checks; no performance measurement.

| Stage | Shape B/H/I/E/K | Shared | Weights | Cases x ranks | Node rc |
|---|---|---|---|---|---|
| 20261025_0647 ep16-shmem-small | 32 / 128 / 128 / 64 / 8 | False | torch.bfloat16 | 4 x16 pass |0 /0|
| 20261025_0653 ep16-shmem-shared-fp32-small | 32 / 128 / 128 / 64 / 8 | True | torch.float32 | 4 x16 pass |0 /0|
| 20261025_0658 ep16-user-shape-bf16 | 8192 / 6144 / 2048 / 64 / 8 | False | torch.bfloat16 | 1 x16 pass |0 /0|
| 20261025_0703 ep16-user-shape-shared-bf16 | 8192 / 6144 / 2048 / 64 / 8 | True | torch.bfloat16 | 1 x16 pass |0 /0|

All ranks audited Y/dx/dp/FP32dW1/dW2 and leaf gradients; shared additionally dSW1/dSW2 and leaf shared gradients. Exact dtypes, element counts, counts, global case gates and unchanged limits(atol0.003,rtol0.03,RMSrel0.02) were checked. Main cases each have4 experts x16384 rows =65536 expert rows/rank; actual Xe/GU/A/W1/W2/Z shapes were checked.

Actual NPU content identity: `15399c4d3624b1e6fd5f5789e81e80b4d4c62bc73d98fa3be9874fb1613c6015`. Frozen native SHA256: `f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32`. Source/native/manifest/runtime-library identities/precise argv are preserved at `/home/zzy/deploy/megamoe-20261024/training/qualified-p0-20261025` on both hosts. Activate with `source /home/zzy/deploy/megamoe-20261024/training/activate-qualified-p0.sh`. This is a content-verified copy of the tested review source; raw reports retain original tested paths. Existing training venv/CANN/SHMEM remain read-only dependencies.

Both main qualifications are separated and include rank0..15 on each host under `training/qualification/20261025_0658_ep16-user-shape-bf16` and `training/qualification/20261025_0703_ep16-user-shape-shared-bf16`. File SHA checks passed on both hosts. Source versions were preserved; future candidates must use a separate directory.

All recorded own worker/torchrun/supervisor PIDs exited with node rc0. The07:08:56/07:09:03 fresh gate found16 free cards and no own/foreign processes before the separate CANN small-stage start (controller evidence `training_round_20261025_0710/ep16-cann-ai-cpu-small-controller/gates.json`).

Earlier startup/AICPU initialization failures are not proven fixed by these successful runs. This report is accuracy for this16-card module and exact cases, not64-card execution, MFU/bandwidth/overlap evidence, or CANN/AIV baseline qualification. CANN `ai_cpu` small accuracy is a separate active stage.
