set -eo pipefail
T=/home/zzy/deploy/megamoe-20261024/training
source "$T/activate.sh"
set -u
cd "$T"
date -Is > logs/final-cpu-time.txt
git -C DeepEP rev-parse HEAD > logs/base-head.txt
git -C /home/zzy/deploy/megamoe-20261024/DeepEP status --porcelain > logs/inference-source-status.txt
sha256sum DeepEP/experiments/megamoe_training/build/libmegamoe_training.so > logs/native-final.sha256
ldd DeepEP/experiments/megamoe_training/build/libmegamoe_training.so > logs/native-final-ldd.txt
if grep -q 'not found' logs/native-final-ldd.txt; then exit 2; fi
cp /home/zzy/deploy/megamoe-20261024/DeepEP/third-party/shmem/install/shmem/deepep-sdk.json logs/shmem-sdk-manifest.json
python -m py_compile scripts/probe_cann_aiv.py scripts/validate_training_cpu.py
bash -n activate.sh scripts/run_training_checked.sh
printf '0\n' > logs/cpu-final-syntax.rc
printf 'NOT_RUN: foreign NPU ownership gate stopped device test before launch\n' > logs/native-gemm-device-test.status
printf '%s\n' 'source /home/zzy/deploy/megamoe-20261024/training/activate.sh' 'cd /home/zzy/deploy/megamoe-20261024/training/DeepEP' 'cmake -S experiments/megamoe_training -B experiments/megamoe_training/build -DCATLASS_ROOT=/home/zzy/deploy/megamoe-20261024/DeepEP/third-party/catlass' 'cmake --build experiments/megamoe_training/build --parallel 8' > logs/native-build-command.txt
tar --exclude='*.tar.gz' -czf logs-evidence.tar.gz logs
sha256sum logs-evidence.tar.gz
