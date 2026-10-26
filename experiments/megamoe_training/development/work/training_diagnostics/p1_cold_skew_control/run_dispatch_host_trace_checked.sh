#!/usr/bin/env bash
set -eo pipefail
T=/home/zzy/deploy/megamoe-20261024/training
source "$T/activate-qualified-p0.sh"
export DISPATCH_DIAGNOSTIC_ROOT="$T/diagnostic-p1-cold-skew-20261026"
export CANDIDATE_SOURCE_ROOT="/home/zzy/deploy/megamoe-20261024/training/candidate-p1-host-v2/p1_pointwise"
export POINTWISE_SO="$T/candidate-p1/build/libmegamoe_pointwise.so"
set -u
NODE_RANK=${1:?node rank}; STAGE=p1-host-v2-cold-skew-shared-fp32-small
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]]
: "${RUN_TAG:?same explicit RUN_TAG on both hosts}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
RUN="$T/runs/$RUN_TAG/$STAGE"
mkdir -p "$RUN"
MARKER="$RUN/.node$NODE_RANK-launch-once"
if ! mkdir "$MARKER" 2>/dev/null; then
 echo 'Duplicate launch marker: refusing to overwrite logs or start' >&2; exit 11
fi
trap 'saved_rc=$?; trap - EXIT; mkdir -p "$RUN/logs"; printf "%s\n" "$saved_rc" > "$RUN/logs/node$NODE_RANK.rc"; exit "$saved_rc"' EXIT
python -c 'import os,sys,pathlib,json,time; p=pathlib.Path("/proc")/sys.argv[1]; pathlib.Path(sys.argv[2]).write_text(json.dumps({"pid":int(sys.argv[1]),"pgid":os.getpgid(int(sys.argv[1])),"process_start_ticks":p.joinpath("stat").read_text().rsplit(")",1)[1].split()[19],"boot_id":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"argv":[v.decode(errors="replace") for v in p.joinpath("cmdline").read_bytes().split(b"\0") if v],"time_ns":time.time_ns()}))' "$$" "$MARKER/identity.json"
: "${DISPATCH_MANIFEST_SHA256:?controller must supply reviewed deployment manifest SHA}"
timeout 45 python "$DISPATCH_DIAGNOSTIC_ROOT/scripts/verify_deployment.py" "$DISPATCH_DIAGNOSTIC_ROOT/diagnostic-deployment-manifest.json" "$DISPATCH_MANIFEST_SHA256"
mkdir -p "$RUN/logs" "$RUN/ascend-log" "$RUN/shmem-debug"
export ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_PROCESS_LOG_PATH="$RUN/ascend-log"
export SHMEM_LOG_LEVEL=DEBUG SHMEM_LOG_PATH="$RUN/shmem-debug"
export SHMEM_UID_SOCK_IFNAME=enp35s0f2 HCCL_SOCKET_IFNAME=enp35s0f2 GLOO_SOCKET_IFNAME=enp35s0f2
export DEEPEP_SHMEM_ENDPOINT="tcp://177.22.27.1:27774"
TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-29926}
# Last local gate follows the controller's parallel all16 / own-process gate.
timeout 30 npu-smi info > "$RUN/logs/npu-before-node$NODE_RANK.txt"
if [ "$(grep -c 'No running processes' "$RUN/logs/npu-before-node$NODE_RANK.txt")" != 8 ]; then
 echo 'NPU occupied; no device stage launched' >&2
 echo 10 > "$RUN/logs/node$NODE_RANK.rc"
 exit 10
fi
ARGS=(--run-npu --training-source-root "$TRAINING_SOURCE_ROOT"
 --candidate-source-root "$CANDIDATE_SOURCE_ROOT" --pointwise-library "$POINTWISE_SO"
 --seed 4317 --oracle-chunk 1024 --max-reference-gib 12 --world-size 16
 --B 32 --H 128 --I 128 --E 64 --K 8 --cases skew,balanced,zero_gate,empty
 --weight-dtype fp32 --shared --output "$RUN")
CMD=(python -m torch.distributed.run --nnodes=2 --nproc-per-node=8
 --node-rank="$NODE_RANK" --rdzv-backend=static --master-addr=177.22.27.1
 --master-port="$TRAIN_MASTER_PORT" "$CANDIDATE_SOURCE_ROOT/validate_training_p1.py" "${ARGS[@]}")
python -c 'import os,json,sys; print(json.dumps({"argv":sys.argv[1:],"diagnostic_only":True,"environment":{k:os.getenv(k) for k in ("RUN_TAG","TRAINING_SOURCE_ROOT","CANDIDATE_SOURCE_ROOT","DISPATCH_DIAGNOSTIC_ROOT","POINTWISE_SO","PYTHONPATH","MEGAMOE_TRAINING_LIBRARY","SHMEM_LOG_LEVEL","SHMEM_LOG_PATH","HCCL_SOCKET_IFNAME","DEEPEP_SHMEM_ENDPOINT","ASCEND_GLOBAL_LOG_LEVEL","ASCEND_PROCESS_LOG_PATH","ASCEND_HOME_PATH","LD_LIBRARY_PATH","HCCL_OP_EXPANSION_MODE")}},indent=2))' "${CMD[@]}" > "$RUN/logs/launch-node$NODE_RANK.json"
cd "$T"
set +e
timeout --signal=TERM --kill-after=30 600 "${CMD[@]}" > "$RUN/logs/node$NODE_RANK.log" 2>&1
RC=$?
set -e
echo "$RC" > "$RUN/logs/node$NODE_RANK.rc"
timeout 30 npu-smi info > "$RUN/logs/npu-after-node$NODE_RANK.txt" || true
exit "$RC"
