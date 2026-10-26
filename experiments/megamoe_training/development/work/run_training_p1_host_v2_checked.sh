#!/usr/bin/env bash
set -eo pipefail
T=/home/zzy/deploy/megamoe-20261024/training
source "$T/activate-qualified-p0.sh"
export CANDIDATE_SOURCE_ROOT="$T/candidate-p1-host-v2/p1_pointwise"
export POINTWISE_SO="$T/candidate-p1/build/libmegamoe_pointwise.so"
set -u
NODE_RANK=${1:?node rank 0/1}; STAGE=${2:?stage name}; SCRIPT=${3:?script path}; shift 3
: "${RUN_TAG:?Set the same RUN_TAG on both hosts}"
case "$STAGE" in *[!a-zA-Z0-9_-]*) exit 2;; esac
case "$SCRIPT" in /*) ;; *) SCRIPT="$TRAINING_SOURCE_ROOT/$SCRIPT";; esac
test -f "$SCRIPT"
NNODES=${NNODES:-2}; NPROC_PER_NODE=${NPROC_PER_NODE:-8}; LIMIT_SECONDS=${LIMIT_SECONDS:-600}
TRAIN_MASTER_ADDR=${TRAIN_MASTER_ADDR:-177.22.27.1}; TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-29910}
RUN="$T/runs/$RUN_TAG/$STAGE"
mkdir -p "$RUN/logs"
case "$STAGE" in
 *-main*) export SHMEM_LOG_LEVEL=INFO SHMEM_LOG_PATH="$RUN/shmem-info" ;;
 *) export SHMEM_LOG_LEVEL=DEBUG SHMEM_LOG_PATH="$RUN/shmem-debug" ;;
esac
mkdir -p "$SHMEM_LOG_PATH"
timeout 30 npu-smi info > "$RUN/logs/npu-before-node$NODE_RANK.txt"
if [ "$(grep -c 'No running processes' "$RUN/logs/npu-before-node$NODE_RANK.txt")" != 8 ]; then
 echo 'NPU occupied: do not start; notify controller and defer.' >&2; exit 10
fi
export SHMEM_UID_SOCK_IFNAME=enp35s0f2 HCCL_SOCKET_IFNAME=enp35s0f2 GLOO_SOCKET_IFNAME=enp35s0f2
export DEEPEP_SHMEM_ENDPOINT="tcp://$TRAIN_MASTER_ADDR:27774"
cd "$T"
echo "$(date -Is) stage=$STAGE node=$NODE_RANK pid=$$" > "$RUN/logs/launch-node$NODE_RANK.txt"
python -c 'import os,json,sys; print(json.dumps({"argv":sys.argv[1:],"environment":{k:os.getenv(k) for k in ("RUN_TAG","TRAINING_SOURCE_ROOT","CANDIDATE_SOURCE_ROOT","POINTWISE_SO","SHMEM_LOG_LEVEL","SHMEM_LOG_PATH","HCCL_SOCKET_IFNAME","DEEPEP_SHMEM_ENDPOINT","NNODES","NPROC_PER_NODE","LIMIT_SECONDS")}},indent=2))' "$SCRIPT" --output "$RUN" "$@" > "$RUN/logs/launch-node$NODE_RANK.json"
set +e
timeout --signal=TERM --kill-after=30 "$LIMIT_SECONDS" python -m torch.distributed.run \
 --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" --node-rank="$NODE_RANK" \
 --rdzv-backend=static --master-addr="$TRAIN_MASTER_ADDR" --master-port="$TRAIN_MASTER_PORT" \
 "$SCRIPT" --output "$RUN" "$@" > "$RUN/logs/node$NODE_RANK.log" 2>&1
RC=$?
set -e
echo "$RC" > "$RUN/logs/node$NODE_RANK.rc"
timeout 30 npu-smi info > "$RUN/logs/npu-after-node$NODE_RANK.txt"
echo "stage=$STAGE node=$NODE_RANK rc=$RC logs=$RUN/logs"
exit "$RC"
