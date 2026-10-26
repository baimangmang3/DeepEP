set -euo pipefail
D=/home/zzy/deploy/megamoe-20261024
T="$D/training"
mkdir -p "$T"/{deps,logs,scripts,downloads}
if [ ! -d "$T/DeepEP/.git" ]; then git clone --no-hardlinks "$D/DeepEP" "$T/DeepEP"; fi
git -C "$T/DeepEP" remote set-url origin https://github.com/lueelu/DeepEP.git
git -C "$T/DeepEP" rev-parse HEAD
if [ ! -d "$T/venv" ]; then /root/miniconda3/envs/zzy/bin/python -m venv --system-site-packages "$T/venv"; fi
if [ ! -d "$T/deps/catccos/.git" ]; then
 timeout 180 git clone https://gitcode.com/cann/catccos.git "$T/deps/catccos"
 git -C "$T/deps/catccos" checkout --detach 878c2e0a504be8c49c03b5e1157a7c714a628e29
fi
timeout 180 git -C "$T/deps/catccos" submodule update --init 3rdparty/catlass
git -C "$T/deps/catccos" submodule status --recursive 3rdparty/catlass
echo PREPARED
