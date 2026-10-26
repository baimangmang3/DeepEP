#!/usr/bin/env bash
set -eo pipefail
T=/home/zzy/deploy/megamoe-20261024/training
source "$T/activate-qualified-p0.sh"
set -u
NODE_RANK=${1:?node rank 0/1}; STAGE=${2:?stage name}; SCRIPT=${3:?script path}; shift 3
: "${RUN_TAG:?Set the same RUN_TAG on both hosts}"
case "$STAGE" in *[!a-zA-Z0-9_-]*) exit 2;; esac
case "$SCRIPT" in /*) ;; *) SCRIPT="$TRAINING_SOURCE_ROOT/$SCRIPT";; esac
test -f "$SCRIPT"
NNODES=${NNODES:-2}; NPROC_PER_NODE=${NPROC_PER_NODE:-8}; LIMIT_SECONDS=${LIMIT_SECONDS:-900}
TRAIN_MASTER_ADDR=${TRAIN_MASTER_ADDR:-177.22.27.1}; TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-29910}
RUN="$T/runs/$RUN_TAG/$STAGE"
mkdir -p "$RUN/logs"
npu-smi info > "$RUN/logs/npu-before-node$NODE_RANK.txt"
if [ "$(grep -c 'No running processes' "$RUN/logs/npu-before-node$NODE_RANK.txt")" != 8 ]; then
 echo 'NPU occupied: do not start; notify controller and defer.' >&2; exit 10
fi
export SHMEM_UID_SOCK_IFNAME=enp35s0f2 HCCL_SOCKET_IFNAME=enp35s0f2 GLOO_SOCKET_IFNAME=enp35s0f2
export DEEPEP_SHMEM_ENDPOINT="tcp://$TRAIN_MASTER_ADDR:27774"
cd "$T"
echo "$(date -Is) stage=$STAGE node=$NODE_RANK pid=$$" > "$RUN/logs/launch-node$NODE_RANK.txt"
set +e
timeout --signal=TERM --kill-after=30 "$LIMIT_SECONDS" python -m torch.distributed.run \
 --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" --node-rank="$NODE_RANK" \
 --rdzv-backend=static --master-addr="$TRAIN_MASTER_ADDR" --master-port="$TRAIN_MASTER_PORT" \
 "$SCRIPT" --output "$RUN" "$@" > "$RUN/logs/node$NODE_RANK.log" 2>&1
RC=$?
set -e
echo "$RC" > "$RUN/logs/node$NODE_RANK.rc"
npu-smi info > "$RUN/logs/npu-after-node$NODE_RANK.txt"
echo "stage=$STAGE node=$NODE_RANK rc=$RC logs=$RUN/logs"
exit "$RC"
