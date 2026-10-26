#!/usr/bin/env bash
set -eo pipefail
T=/home/zzy/deploy/megamoe-20261024/training
R="$T/validation-p4-component-v2-control-20261026"
source "$T/activate-qualified-p0.sh"
export RUN_TAG="${1:?RUN_TAG}" ASCEND_RT_VISIBLE_DEVICES=0 P4_ROLE=supervisor
RUN="$T/runs/$RUN_TAG/p4-v2-component-small"
export ASCEND_PROCESS_LOG_PATH="$RUN/ascend-log" ASCEND_GLOBAL_LOG_LEVEL=0
mkdir -p "$RUN/ascend-log"
set +e
"$T/venv/bin/python" "$R/scripts/supervisor.py" --execute --run-tag "$RUN_TAG"
RC=$?
printf '%s\n' "$RC" > "$RUN/supervisor.rc"
exit "$RC"
