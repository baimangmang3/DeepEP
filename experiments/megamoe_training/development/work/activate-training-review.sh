source /home/zzy/deploy/megamoe-20261024/training/activate.sh
export TRAINING_SOURCE_ROOT=/home/zzy/deploy/megamoe-20261024/training/review-20261024/experiments/megamoe_training
export MEGAMOE_TRAINING_LIBRARY="$TRAINING_SOURCE_ROOT/build/libmegamoe_training.so"
_review_python_path="$TRAINING_SOURCE_ROOT"
IFS=: read -ra _review_python_parts <<< "${PYTHONPATH:-}"
for _review_python_part in "${_review_python_parts[@]}"; do
    if [ -n "$_review_python_part" ] && [ "$_review_python_part" != "$TRAINING_DEPLOY/DeepEP/experiments/megamoe_training" ] && [ "$_review_python_part" != "$TRAINING_SOURCE_ROOT" ]; then
        _review_python_path="$_review_python_path:$_review_python_part"
    fi
done
export PYTHONPATH="$_review_python_path"
unset _review_python_path _review_python_parts _review_python_part
