source /home/zzy/deploy/megamoe-20261024/activate.sh
source /home/zzy/deploy/megamoe-20261024/training/venv/bin/activate
export PATH=/home/zzy/deploy/megamoe-20261024/training/venv/bin:/root/miniconda3/envs/zzy/bin:$PATH
export TRAINING_DEPLOY=/home/zzy/deploy/megamoe-20261024/training
export SHMEM_ROOT=/home/zzy/deploy/megamoe-20261024/DeepEP/third-party/shmem/install/shmem
export SHMEM_SOURCE_PATH=/home/zzy/deploy/megamoe-20261024/DeepEP/third-party/shmem
export CATLASS_ROOT=/home/zzy/deploy/megamoe-20261024/DeepEP/third-party/catlass
export CATCCOS_SOURCE_PATH=/home/zzy/deploy/megamoe-20261024/training/deps/catccos
_megamoe_training_python_path="$TRAINING_DEPLOY/DeepEP/experiments/megamoe_training"
IFS=: read -ra _megamoe_training_python_parts <<< "${PYTHONPATH:-}"
for _megamoe_training_python_part in "${_megamoe_training_python_parts[@]}"; do
    if [ -n "$_megamoe_training_python_part" ]; then
        _megamoe_training_python_path="$_megamoe_training_python_path:$_megamoe_training_python_part"
    fi
done
export PYTHONPATH="$_megamoe_training_python_path"
unset _megamoe_training_python_path _megamoe_training_python_parts _megamoe_training_python_part
