#!/usr/bin/env bash
set -eo pipefail
T=/home/zzy/deploy/megamoe-20261024/training
D=/home/zzy/deploy/megamoe-20261024/training/candidate-p3-up-swiglu-20261025
source "$T/activate-qualified-p0.sh"
set -u
NODE_RANK=${1:?}; STAGE=${2:?}; SCRIPT=${3:?}; shift 3
: "${RUN_TAG:?}"
test "$NODE_RANK" = 0
test "${NNODES}" = 1
test "${NPROC_PER_NODE}" = 1
test "${LIMIT_SECONDS}" = 600
test "$SCRIPT" = "$D/test_up_swiglu_npu.py"
test "$#" = 0
RUN="$T/runs/$RUN_TAG/$STAGE"
mkdir -p "$RUN/logs"
python -c 'import os,json,sys;print(json.dumps({"script":sys.argv[1],"environment":{k:os.getenv(k) for k in ("RUN_TAG","ASCEND_HOME_PATH","LD_LIBRARY_PATH","TRAINING_SOURCE_ROOT","ASCEND_GLOBAL_LOG_LEVEL","ASCEND_PROCESS_LOG_PATH")}},indent=2))' "$SCRIPT" > "$RUN/logs/launch-node0.json"
set +e
python "$T/scripts/p3_unit_supervisor.py" --execute --run-tag "$RUN_TAG" --output-dir "$RUN" --probe "$SCRIPT" --up-library "$D/build/libmegamoe_up_swiglu.so" --p0-source-root "$TRAINING_SOURCE_ROOT" --catlass-library "$TRAINING_SOURCE_ROOT/build/libmegamoe_training.so" --timeout-seconds 600
RC=$?
set -e
echo "$RC" > "$RUN/logs/node0.rc"
timeout 30 npu-smi info > "$RUN/logs/npu-after-node0.txt"
exit "$RC"
