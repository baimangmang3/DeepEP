# MegaMoE training development snapshot

This branch preserves the SHMEM/CATLASS training work through 2026-10-26.
The user stopped the recurring `megamoe-16` task before this source export.
All historical pending or conditional execution permissions are revoked;
the review files below are evidence, not instructions to launch a job.

The repository retains upstream DeepEP history at
`35fa7d91b321340c153b098bf166202645976a75`. This is a private copy of the
repository; it does not change the public upstream or its submodule commits.

## Contents

| Path | Contents |
| --- | --- |
| `../megamoe_training`, `../native`, `../tests` | Staged SHMEM/CATLASS training API, native GEMM launcher and validation |
| `work/training_candidates` | Separate P1 pointwise, P2 backward overlap, P3 activation, P4 compute-wave and P5 role-refactor candidates; CPU models and qualification code |
| `work/ep_gmm_ep16_patch`, `work/ep_gmm_training_validation` | Separately labelled EP16 native adaptation, training adapter validation and original EP8 scope |
| `work/training_diagnostics` | Bounded diagnostic controllers, ownership gates and child-first cleanup |
| `work/torchrun_event_hostname_wrapper` | Per-run Elastic event hostname wrapper; no system resolver modification |
| `work/training_perf_probe` | Timing scope, provenance and qualification gates |
| `outputs` | Design reviews, checkpoints and selected strict validation summaries |
| `SOURCE_ARCHIVE_MANIFEST.json` | Original relative paths, sizes and SHA-256 hashes of copied files |

Copied sources retain their original bytes and layout under `work/` and
`outputs/`. Machine-specific absolute paths, original deployment roots and
historical hashes are intentionally retained for provenance. Review these
before adapting an entry point; this archive is not a portable one-command
deployment. Some preparation tools expect a separate workspace containing
`work/DeepEP` pointing to a checkout. Create such a scratch workspace explicitly
if needed; do not run historical once-only launch scripts from this archive.

Credentials, native binaries, build caches, SDK header copies and large raw
runtime traces are excluded. Original local/remote evidence is retained in
the working session. Existing unrelated nested CATLASS submodule changes are
not part of this commit.

## Measured status and limits

- P0/CANN target reports exist with pinned source/library identities. They do
  not automatically qualify changed candidates or a new build.
- P1 cold-skew small validation passed 704 metrics and target routed validation
  passed 112. The later shared run passed 176 numerical metrics but lacks
  complete exit-status evidence after interruption; it is not fully qualified.
- P2 EP2 serial and overlap each passed 56 fixed-threshold metrics with internal
  initial/final library identity and normal exit evidence. EP16 serial remained
  pending behind foreign-launcher gates. No P2 performance claim follows.
- P4 compute-wave v1/v2 CPU builds and component/control preparation completed.
  The component device run had not started. It is not a complete SHMEM training
  fusion or full backward implementation.
- P5 identity64 refactors seven Dispatch files to explicit logical roles while
  retaining the original 64-role protocol. Its CPU compile controller run
  `20261026_0708_p5_identity64_cpu_once` exited with real code 1 before invoking
  the compiler: `prepare_snapshots.py` was absent from the deployed package,
  although the manifest checks all scripts. The report has `runs=[]`.
  Neither original nor candidate compilation succeeded in that attempt.
  The last collected recorder/driver PID-start checks showed both gone.
- The earlier 1431 Dispatch reuse/vector-core failure remains unexplained;
  later non-reproduction does not prove a root cause or fix.

Use `outputs/training_handoff.md`, `training_development_checkpoint.json` and
the specific identity-bound result/review together. Synthetic CPU checks,
source inspection and native compilation are distinct from NPU correctness.
No new build, device test or benchmark was run for this repository export.

The target remains BF16 EP16 B8192 H6144 I2048 E64 K8, four experts and
65536 dispatched rows per rank, with Y/dx/dp/dW1/dW2/shared validation at fixed
thresholds. A future performance comparison must retain identical routing,
weights, casting, gating, saved Xe and recomputation scope, plus raw per-rank
timings and verified cleanup. MoE-layer throughput is not full-model MFU.
