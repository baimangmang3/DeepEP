# Review version 2026-10-24

03:00 续测使用本review版本：source /home/zzy/deploy/megamoe-20261024/training/activate-review.sh。TRAINING_SOURCE_ROOT=/home/zzy/deploy/megamoe-20261024/training/review-20261024/experiments/megamoe_training；scripts/run_training_review_checked.sh 的相对脚本路径也从该根解析。原 scripts/run_training_checked.sh 已转发到新review launcher；旧实现保留为 run_training_legacy_checked.sh。

review源16文件（11 Python）与旧13文件快照分开；native SO沿用同一SHA。旧manifest/ZIP/大bundle保持不变。新的完整qualification必须在此源码/库/precision身份下生成；旧报告不能复用。先按双机foreign门禁，若占用延后10分钟，不运行设备代码。
# 2026-10-25 03:00 后续测手册

当前仅完成源码/构建/CPU部署核对；训练设备精度与性能尚未运行。两机为 `177.22.27.1`、`177.22.28.1`，每机8卡。

## 开始条件和超时

首次03:00检查后，按已设heartbeat每10分钟重查，只有两机同时无foreign任务才开始。先核对已有本任务进程和结果，避免重复启动。保留两机 `npu-smi info` 和相关PID/命令，不能仅看利用率0；仍占用则顺延，不终止外部进程。

`/home/zzy/deploy/megamoe-20261024/training/scripts/run_training_review_checked.sh` 在每次本地启动前重新确认8卡均无运行进程，否则返回10。控制端必须先完成两机全局门禁，再同时启动两个节点。脚本使用独立训练端口29910/27774、static rendezvous、默认900秒watchdog，先TERM、30秒后KILL本次timeout进程组；大规格可设 `LIMIT_SECONDS=3600`。任一节点非零立即停止该阶段，确认本次rank退出，收集另一节点日志，不启动下一阶段。禁止宽泛 `pkill python`；清理只能针对已核实属于本次运行的PID/进程组。

两机启动前共同设置相同标签；27设置 `NODE=0`，28设置 `NODE=1`：

```bash
export T=/home/zzy/deploy/megamoe-20261024/training
source "$T/activate-review.sh"
export RUN_TAG=20261025_0300
# 177.22.27.1: NODE=0；177.22.28.1: NODE=1
export NODE=0
```

所有EP16阶段都在**两台机器同时执行同一命令**，仅 `NODE` 不同。绝不能单机 `--nproc-per-node=16 --standalone`。包装器实际使用：

```bash
python -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
  --node-rank="$NODE" --rdzv-backend=static \
  --master-addr=177.22.27.1 --master-port=29910 <脚本> <参数>
```

## 顺序

1. 对照 `review-20261024/source-manifest.json` 校验16个源码及SO，CPU验证日志必须成功，native SO SHA为 `f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32`。默认cores32。
2. 两机全局门禁通过后，仅27执行单卡原生测试，保存rc并确认进程退出：

```bash
npu-smi info > "$T/logs/npu-before-native-retry.txt"
test "$(grep -c 'No running processes' "$T/logs/npu-before-native-retry.txt")" = 8 || exit 10
timeout --signal=TERM --kill-after=30 180 python \
  "$TRAINING_SOURCE_ROOT/tests/test_native_gemm.py" \
  --output "$T/runs/$RUN_TAG/native-gemm/results.json"
```

3. 仅27依次执行两个EP2阶段；第二个同时覆盖共享专家和FP32 master weights：

```bash
NNODES=1 NPROC_PER_NODE=2 TRAIN_MASTER_ADDR=127.0.0.1 \
 bash "$T/scripts/run_training_review_checked.sh" 0 ep2-shmem tests/validate_training_npu.py \
 --B 32 --H 128 --I 128 --E 8 --K 4 --cases balanced,zero_gate,empty,skew --backend shmem_catlass
NNODES=1 NPROC_PER_NODE=2 TRAIN_MASTER_ADDR=127.0.0.1 \
 bash "$T/scripts/run_training_review_checked.sh" 0 ep2-shared-fp32 tests/validate_training_npu.py \
 --B 32 --H 128 --I 128 --E 8 --K 4 --cases balanced,zero_gate,empty,skew \
 --backend shmem_catlass --shared --weight-dtype fp32
```

4. 两机EP16小规格。先 `shmem_catlass`，再该后端加 `--weight-dtype fp32` 使用新stage名，随后分别 `cann_ai_cpu`、`cann_ccu`，每个后端独立结果目录：

```bash
bash "$T/scripts/run_training_review_checked.sh" "$NODE" ep16-shmem-small tests/validate_training_npu.py \
 --B 32 --H 128 --I 128 --E 64 --K 8 --cases balanced,zero_gate,empty,skew --backend shmem_catlass
```

CANN是我们用融合op组合的custom autograd，不是官方原生反向注册；`permute_out=True` 和反向 `trans_gmm_weight=False` 尚待本轮资格验证。所有Y/dX/dP/FP32dW和leaf梯度、counts必须通过固定门限，不能放宽。

5. EP16主规格，依次三后端分别完整精度；该配置每rank接收65536行、每专家16384行，对应用户权重 `[4,4096,6144]` / `[4,6144,2048]`：

```bash
LIMIT_SECONDS=3600 bash "$T/scripts/run_training_review_checked.sh" "$NODE" ep16-shmem-main tests/validate_training_npu.py \
 --B 8192 --H 6144 --I 2048 --E 64 --K 8 --cases balanced --backend shmem_catlass \
 --oracle-chunk 1024 --max-reference-gib 12
```

6. 收集两机 `ep16-shmem-main/rank0.json` 至 `rank15.json`，集中到同一完整qualification目录，再将相同16文件分发两机，例如 `$T/qualification/$RUN_TAG/main16`。`benchmark_compare.py --benchmark/--profile` 强制读取此目录，要求同shape、BF16参数、backend=shmem_catlass、完整训练全通过并记录SHA；缺rank或旧报告会拒绝。精度齐备后方可重跑完整训练benchmark（各后端独立stage加 `--benchmark --warmup 5 --iters 30`），并作同scope正向比较：

```bash
LIMIT_SECONDS=3600 bash "$T/scripts/run_training_review_checked.sh" "$NODE" forward-compare-main tests/benchmark_compare.py \
 --B 8192 --H 6144 --I 2048 --E 64 --K 8 --cases balanced --modes ai_cpu,ccu \
 --benchmark --warmup 5 --iters 30 --training-qualification-dir "$T/qualification/$RUN_TAG/main16"
```

主比较包括每次动态metadata与pack/unpack，CANN必须导出Xe。prepared-plan和prepared-input算子区间只能另列；不与fresh候选作speedup。完整训练耗时不能与正向耗时混比。需要另加 `--profile` 的独立短采集证明真实kernel/tiling、AICPU/CCU后端；NPU event时间不等于纯kernel时间。

7. AIV拒绝在单独torchrun子进程中采证，避免污染其他HCCL上下文。仅27运行如下EP8 probe；它直接调用原生torch_npu接口，绕过wrapper模式限制，记录每rank异常和真实SO SHA。若意外成功，应追加精度和kernel资格，而不能直接标AIV可比结果。当前950源码和实际SO表明不支持；CCU不替代AIV命名。

```bash
NNODES=1 NPROC_PER_NODE=8 TRAIN_MASTER_ADDR=127.0.0.1 \
 bash "$T/scripts/run_training_review_checked.sh" 0 aiv-rejection "$T/scripts/probe_cann_aiv.py"
```

最后检查本次rank/PID全退出，再保存npu-smi、逐rank结果、profiler、SHA和源码归档。若出现foreign，保留其归属，停止新设备阶段并继续既定占用重查。


## ep_gmm_fused 新增对照

先读取两机training/EP_GMM_PREFLIGHT.md。保留pinned原版EP8独立wheel/环境/哈希；当前仅依赖准备，尚无其构建或设备资格。EP16适配另立版本，先核查rank数组、metadata ABI、UB/workspace、queue-owner、ready协议及跨机UDMA，再完整精度，之后才采同scope性能、带宽、通算掩盖和MFU。原版无autograd，不得称完整训练后端；两组EP8不等于单EP16。原版可先两host各4卡组成单EP8域验收。03:00/每10分钟占用门禁与timeout不变；现有训练源码快照和证据包不重打包。
