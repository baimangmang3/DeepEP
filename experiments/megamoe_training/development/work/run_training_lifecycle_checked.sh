#!/usr/bin/env bash
set -eo pipefail
T=/home/zzy/deploy/megamoe-20261024/training
source "$T/activate-qualified-p0.sh"
set -u
NODE_RANK=${1:?}; STAGE=${2:?}; SCRIPT=${3:?}; shift 3
: "${RUN_TAG:?}"
case "$STAGE" in *[!a-zA-Z0-9_-]*) exit 2;; esac
test "$SCRIPT" = "$T/scripts/lifecycle_probe.py"
RUN="$T/runs/$RUN_TAG/$STAGE"
mkdir -p "$RUN/logs" "$RUN/shmem-debug"
timeout 30 npu-smi info > "$RUN/logs/npu-before-node$NODE_RANK.txt"
if [ "$(grep -c 'No running processes' "$RUN/logs/npu-before-node$NODE_RANK.txt")" != 8 ]; then exit 10; fi
export SHMEM_UID_SOCK_IFNAME=enp35s0f2 HCCL_SOCKET_IFNAME=enp35s0f2 GLOO_SOCKET_IFNAME=enp35s0f2
export DEEPEP_SHMEM_ENDPOINT="tcp://${TRAIN_MASTER_ADDR:-177.22.27.1}:27774"
export SHMEM_LOG_LEVEL=DEBUG SHMEM_LOG_PATH="$RUN/shmem-debug"
cd "$T"
python -c 'import os,json,sys; print(json.dumps({"argv":sys.argv[1:],"environment":{k:os.getenv(k) for k in ("RUN_TAG","TRAINING_SOURCE_ROOT","SHMEM_LOG_LEVEL","SHMEM_LOG_PATH","SHMEM_UID_SOCK_IFNAME","HCCL_SOCKET_IFNAME","DEEPEP_SHMEM_ENDPOINT","NNODES","NPROC_PER_NODE","LIMIT_SECONDS")}},indent=2))' "$SCRIPT" --output "$RUN" "$@" > "$RUN/logs/launch-node$NODE_RANK.json"
set +e
timeout --signal=TERM --kill-after=30 "${LIMIT_SECONDS:-600}" python -m torch.distributed.run \
 --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" --rdzv-backend=static \
 --master-addr="${TRAIN_MASTER_ADDR:-177.22.27.1}" --master-port=29910 \
 "$SCRIPT" --output "$RUN" "$@" > "$RUN/logs/node$NODE_RANK.log" 2>&1
RC=$?
set -e
echo "$RC" > "$RUN/logs/node$NODE_RANK.rc"
timeout 30 npu-smi info > "$RUN/logs/npu-after-node$NODE_RANK.txt"
exit "$RC"
