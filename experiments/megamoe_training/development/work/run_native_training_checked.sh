#!/usr/bin/env bash
set -eo pipefail
T=/home/zzy/deploy/megamoe-20261024/training
source "$T/activate-review.sh"
: "${RUN_TAG:?}"
RUN="$T/runs/$RUN_TAG/native-gemm"
mkdir -p "$RUN"
npu-smi info > "$RUN/npu-before.txt"
if [ "$(grep -c 'No running processes' "$RUN/npu-before.txt")" != 8 ]; then echo 10 > "$RUN/rc"; exit 10; fi
set +e
timeout --signal=TERM --kill-after=30 90 python "$T/scripts/query_training_cores.py" > "$RUN/core-query.log" 2>&1
RC=$?
if [ "$RC" = 0 ]; then
 timeout --signal=TERM --kill-after=30 180 python "$TRAINING_SOURCE_ROOT/tests/test_native_gemm.py" --output "$RUN/results.json" > "$RUN/native.log" 2>&1
 RC=$?
fi
echo "$RC" > "$RUN/rc"
npu-smi info > "$RUN/npu-after.txt"
exit "$RC"
