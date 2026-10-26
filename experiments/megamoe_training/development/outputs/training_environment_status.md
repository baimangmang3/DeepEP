# 训练环境最终状态（2026-10-24）

**完成CPU部署、原生构建和产物同步；本轮没有执行任何NPU训练测试，训练精度与性能尚未验收。** 20:12:33/34 CST 的测试前门禁发现两机16卡被foreign作业占用，已停止上板。03:00后的自动续接负责重新检查并执行。

## 已部署

两机：`177.22.27.1` / `177.22.28.1`，各8张Ascend950DT。统一训练根目录：

```bash
source /home/zzy/deploy/megamoe-20261024/training/activate.sh
```

源码位于 `training/DeepEP/experiments/megamoe_training/`，独立venv位于 `training/venv/`，推理环境保留。已核对DeepEP实际从训练venv加载，实验模块从该训练源码目录加载；激活脚本去掉PYTHONPATH空项以避免checkout根覆盖已安装native包。

- 两机13个源码逐文件对齐最终 `training_source_manifest.json`；8个Python文件均py_compile成功，两机CPU部署核对rc=0。
- 最终源码ZIP：`training_source_snapshot.zip`，SHA256 `d7df9794d10b8aa92a016b396d6db2ff1f85752f2ebce546a095191e3db801ba`。
- 原生库仅27机编译一次、同一份SO分发两机；SHA256 `f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32`，backend字符串 `CATLASS-Ascend950-BF16-FP32acc-basic-v1`，默认cores32。
- 原生库路径：`training/DeepEP/experiments/megamoe_training/build/libmegamoe_training.so`。构建rc=0，ldd的runtime/acl来自本次正式CANN9.2。
- 训练核心包两机使用相同已验证main35fa7d9 wheel，SHA256 `1e0992a7e7123f69ea5719b5db82fc9b137ae5d7c32a946cd18f8112197fd233`，各自安装rc=0。两机SHMEM SDK manifest一致。
- 依赖复用：CANN9.2.0、Torch2.12.0+cpu、torch_npu2.12.0.post2；CATLASS `2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1`、SHMEM `4dd80e2129da42bd773982c2bb3758af1ed0bba8`。未重复安装CANN。

可复现构建命令（27机，已执行成功）：

```bash
source /home/zzy/deploy/megamoe-20261024/training/activate.sh
cd /home/zzy/deploy/megamoe-20261024/training/DeepEP
cmake -S experiments/megamoe_training -B experiments/megamoe_training/build \
 -DCATLASS_ROOT=/home/zzy/deploy/megamoe-20261024/DeepEP/third-party/catlass
cmake --build experiments/megamoe_training/build --parallel 8
```

## 占用与未执行部分

20:03首次检查空闲；20:12测试前检查发现foreign EP64 Dispatch profiling：

- 27机：NPU PID3870261–3870268，父3870047。
- 28机：NPU PID3917137–3917144，父3916977。
- 命令使用 `/root/miniconda3/envs/deepep-copy/bin/python tests/elastic/test_dispatch.py --ep-size64 --master-addr177.22.21.1 --master-port19090`，输出至 `/home/y00889330/perf-part1-ep64-rerun/`，不是本次任务。

本任务没有启动设备测试、没有终止外部进程。20:32的两机CPU核对中，`torch.npu.is_initialized()` 在导入前后均为false，本任务设备测试进程清单为空。这里不将整台机器表述为空闲/已释放；最后foreign占用观测为20:12，03:00应重新取证。

NN/NT/TN、FP32输出、zero-K测试已加入正式 `tests/test_native_gemm.py`；EP2/EP16完整训练、M65536主case和性能均待执行。CANN反向由我们的custom autograd组合两个通信/GMM方向及CATLASS wgrad，不能称为官方算子原生autograd；`permute_out=True`、`trans_gmm_weight=False` 尚未完成950设备资格验证。

## AICPU / AIV 实证边界

原生schema存在 `comm_mode`，实际调用拼写使用 `ai_cpu` / `ccu`。`libopapi.so` SHA256为 `a83959f0291e5df5eb29b4e69c8832103ecf9de7e5cb355366f7ca1abab411e1`，包含 `For A5, commMode only support 'ccu' or 'ai_cpu'` 校验字符串。

已安装正式CANN源码证据（路径前缀为 `cann-env/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/`）：

- `allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul.cpp:52–69`：AIV分支限定arch220，其他架构trap，注释说明Host先拒绝。
- `grouped_mat_mul_allto_allv/op_kernel/grouped_mat_mul_allto_allv_tiling_key.h:21–32`：仅CCU=0、AICPU=1。
- `allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/hccl_a2av_op.h:47–54`：分别实例化 `HCCL_SERVER_TYPE_CCU` / `HCCL_SERVER_TYPE_AICPU`。

源码/实际SO显示当前950不提供该AIV基线；实际运行拒绝与AICPU成功kernel/tiling profiler证据仍待空闲后补齐。已准备 `training/scripts/probe_cann_aiv.py` 直接调用原生API，绕过wrapper参数门禁；没有将CCU改名AIV，也没有仅凭环境变量宣称不同后端。

## 自动续测与交付

本聊天heartbeat `megamoe-16` 已由主任务设置：2026-10-25 03:00（Asia/Shanghai）首次检查，其后每10分钟重查foreign占用；只有两机同时空闲才开始。具体命令见 [training_runbook.md](training_runbook.md)：native小用例→EP2/shared/FP32→EP16小用例→三后端M65536完整精度→集中并分发16份qualification→同scope性能与profiling→本任务释放及归档。固定数值门禁不放宽。

远端 `training/RUNBOOK.md` 和 `scripts/run_training_checked.sh` 提供正确的两host×8卡static命令、启动前门禁、训练专用端口与超时watchdog。源码、SO、编译日志、13文件与SO哈希、selector源码/metadata、占用及本任务无设备进程证据单独归档为 `training_environment_bundle.tar.gz`；既有推理证据保持分开。

最终环境包大小240815字节；SHA256：`08060b2dc21b06c8a445c59390c5c0f84cdf99085452c4a69d1dcb3ae4b2acd7`。包内71份文件/记录已核查；两机13源码、native哈希、CPU部署rc0、本任务设备进程清单为空均一致。
