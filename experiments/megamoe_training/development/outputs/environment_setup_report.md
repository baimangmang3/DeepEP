# MegaMoE 双机环境部署与验证记录

2026-10-24 完成两机独立部署：官方最新正式版 CANN 9.2.0、当前 DeepEP main、固定 SHMEM/CATLASS。单卡、单机 EP2、GLM-5.2 维度 EP8、双机 EP16 精度均通过，最终包装返回码均为 0。收尾确认两机全部 16 张 NPU 无运行进程，本次测试进程已退出。

## 环境和路径

| 项目 | 177.22.27.1 / rack05os7 | 177.22.28.1 / rack05os8 |
|---|---|---|
| NPU | 8 × Ascend950DT，ACL 型号 Ascend950DT_9582 | 相同 |
| HBM | npu-smi 每卡显示 98304 MB，96 GiB 级产品 | 相同 |
| 健康/初始占用 | 8 卡 OK，无运行 NPU 进程 | 相同 |
| 系统 | openEuler 24.03 LTS-SP3 / aarch64 | 相同 |
| 驱动 | 25.1.rc3.1 | 相同 |
| 固件 | 9.2.0.101.231，package 25.6.rc1.b010 | 相同 |
| Python | 3.12.14 | 相同 |
| Torch / torch_npu | 2.12.0+cpu / 2.12.0.post2 | 相同 |
| CANN | 9.2.0，compiler timestamp 20261015_201912253 | 相同 |
| 最终释放检查时间 | 2026-10-24 19:55:58 +08:00 | 2026-10-24 19:55:55 +08:00 |

两机新部署统一位于 `/home/zzy/deploy/megamoe-20261024`：

- `activate.sh`、`README.md`：日常入口和说明。
- `DeepEP/`：完整 Git checkout；现有 `/home/zzy/src` 内工作树和修改保留。
- `cann-env/Ascend/cann-9.2.0/`：官方 Toolkit + 950-ops，先 Toolkit 后 ops。
- `venv/`：新建 Python overlay，读取 `/root/miniconda3/envs/zzy` 的既有 Torch/torch_npu 和编译运行库；不是完全独立复制的 Python 环境。
- `downloads/`：官方安装包、原始 repodata、选择清单、SHA256、Git bundles。
- `scripts/`、`golden/`、`results/`、`logs/`：可复用脚本、CPU Golden、逐 rank 精度报告和日志。
- `docs/training/`：已同步训练设计、GLM 官方配置、参考实现、验证结果与容量分析。
- `docs/official_cann_metadata/`：官方发布及兼容性 API 快照。

实际 `deep_ep` 和 MegaMoE Python/native 扩展均从新 `venv/lib/python3.12/site-packages/` 加载。Torch/torch_npu 从既有 zzy 环境加载。进程 maps 确认 `libascendcl`、`libruntime`、`libhccl`、`libhcomm` 来自本次正式 CANN 前缀，没有混入 B080、beta.2 或 `/home/pkg` 库。SHMEM 来自本次 checkout 构建的 SDK。原始及补充框架库路径后的 ldd、native import/maps 均保存在证据中。

驱动/固件没有升级、重启或复位。官方安装器修改的 `/etc/Ascend/ascend_cann_install.info` 已从其自动备份恢复，并通过 cmp；原登记仍指向 `/home/pkg/b112`。本环境始终显式激活新路径。

## 官方 CANN 来源

采用最新**正式版** 9.2.0（2026-10-17 发布），没有选择 2026-10-21 preview/Weekly。[官方下载页](https://www.hiascend.com/cann/download?versionId=810)、[实时版本 API](https://www.hiascend.com/ascendgateway/ascendservice/cannDownload/version/list?lang=zh)、[官方 AArch64 repodata](https://repo.huaweicloud.com/ascend/repos/conda/linux-aarch64/repodata.json)；原始 JSON 已归档。

| 包 | 官方下载 URL | SHA256 |
|---|---|---|
| Toolkit | [官方 conda 包](https://repo.huaweicloud.com/ascend/repos/conda/linux-aarch64/ascend-cann-toolkit-9.2.0-0.conda) | `29ea56811737ec10c89b2d3bee886568b9dce06a106b7aee2eaf3f32f8f0247d` |
| 950-ops | [官方 conda 包](https://repo.huaweicloud.com/ascend/repos/conda/linux-aarch64/ascend-cann-950-ops-9.2.0-0.conda) | `fa1253c24c4915db82e63dcdabfc0c6d9506eeec915d87815fcce41d4cf4a649` |

两机安装包均通过官方 repodata SHA256 校验。查询到的公开兼容性助手尚未提供完整 CANN9.2.0 / 950DT / 当前驱动组合的认证矩阵；本报告提供该具体软件栈和下述用例的现场运行证据，不将其表述为官方全量兼容认证。

## 源码和产物身份

| 项目 | 版本 |
|---|---|
| DeepEP | `35fa7d91b321340c153b098bf166202645976a75` |
| SHMEM | `4dd80e2129da42bd773982c2bb3758af1ed0bba8` |
| CATLASS | `2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1` |
| nlohmann/json | v3.11.3 / `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03` |
| wheel | `ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl`，MegaMoE + timer 构建 |

两机无源码补丁，最终 source diff 为空。SHMEM 的 SDK manifest 记录具体 CANN timestamp、源码版本和头文件/库哈希。没有为训练研发扩大构建 ep_gmm_fused。

| 主机 | wheel 绝对路径 | SHA256 |
|---|---|---|
| rack05os7 | `/home/zzy/deploy/megamoe-20261024/DeepEP/dist/build-q8umdcpg/ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl` | `1e0992a7e7123f69ea5719b5db82fc9b137ae5d7c32a946cd18f8112197fd233` |
| rack05os8 | `/home/zzy/deploy/megamoe-20261024/DeepEP/dist/build-knoe0vnc/ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl` | `90891f2486956fe6821ecaee3bfd3c5931fea8de51fc3d29c7df995a3cfcf3ed` |

## 实际验证

全部精度使用量化感知 NumPy CPU Golden、全输出检查（`sampled=0`），并检查专家计数及非零初值的重复累加计数；不是性能测量。模型维度用随机生成权重验证，没有加载完整模型 checkpoint。

| 用例 | 范围与形状 | 结果 |
|---|---|---|
| ACL / 单卡 MatMul | 两机各 device0，8×8 FP32 MatMul 对 CPU | PASS |
| EP2 example | 两机各自两卡，B17/H1024/I256/E4/K2/S0，finite output | PASS |
| EP2 FP8 输入精度 | 两机，B17/H1024/I256/E8/K2/S0，FP4 权重 | 全 rank PASS，最大 relative RMSE `3.14655e-5` |
| EP2 BF16 输入精度 | 同上；输入内部量化，FP4 权重 | 全 rank PASS，最大 relative RMSE `3.14655e-5` |
| GLM-5.2 维度 EP8 | rack05os7，B1/H6144/I2048/E256/K8/S1 | 8/8 PASS，最大 relative RMSE `6.86870e-5`，最大绝对误差 `1.52588e-5` |
| 双机 EP16 | 两机各8卡；B17/H1024/I256/E32/K2/S0，BF16 输入 | 16/16 PASS，最大 relative RMSE `1.96822e-4`，最大绝对误差 `1.22070e-4` |

所有精度用例 `counts_equal=true`、`cumulative_counts_equal=true`。

最终 `install-cann.rc`、`build.rc`、`smoke-local.rc`、`smoke-two-node.rc` 两机均为 0；rack05os7 的 `smoke-glm52.rc` 为 0；最终取证脚本两机也为 0。

关键结果路径：

- rack05os7 EP2：`/home/zzy/deploy/megamoe-20261024/results/local_ep2_20261024_194059_{fp8,bf16}/`。
- rack05os8 EP2：`/home/zzy/deploy/megamoe-20261024/results/local_ep2_20261024_193248_{fp8,bf16}/`。
- GLM EP8：`/home/zzy/deploy/megamoe-20261024/results/glm52_ep8_20261024_194520_bf16/`。
- EP16：rack05os7 `results/two_node_ep16_20261024_194931_bf16/`，rack05os8 `results/two_node_ep16_20261024_194927_bf16/`，均位于统一部署根目录。

## 复用命令

两机均可执行单机验证；脚本会先检查 NPU 占用。

```bash
source /home/zzy/deploy/megamoe-20261024/activate.sh
python -m pip show ascend-deepep torch torch_npu
bash /home/zzy/deploy/megamoe-20261024/scripts/smoke-local.sh
```

GLM 维度验证入口目前在 rack05os7：

```bash
bash /home/zzy/deploy/megamoe-20261024/scripts/smoke-glm52.sh
```

双机需同时启动，27 机器执行第一条，28 机器执行第二条：

```bash
bash /home/zzy/deploy/megamoe-20261024/scripts/smoke-two-node.sh 0
bash /home/zzy/deploy/megamoe-20261024/scripts/smoke-two-node.sh 1
```

双机使用管理网卡 `enp35s0f2`，Torch TCPStore 端口 29874，SHMEM endpoint `tcp://177.22.27.1:26774`；单机使用 127.0.0.1、端口 29872/26774。脚本使用明确的 static rendezvous 和合理 timeout，结果写入新的时间戳目录。构建复用 `scripts/build.sh`，完整安装与构建日志位于 `logs/install-cann.log`、`logs/build.log`。

## 已解决问题、剩余边界和释放

首次启动缺少 SHMEM endpoint，以及更新运行中脚本产生的 EOF，均属于启动脚本问题；已修正并完整复验，原失败日志保留，不计入成功用例。主机名 `getfqdn()` 曾测得 10.47 秒，最终启动脚本采用显式 IP static rendezvous；没有修改系统 hosts/DNS。27 机器的公网 PyPI 缺少可用 matplotlib 包，附带工具依赖改从华为 PyPI 镜像安装。

已补齐 cann-sim、msboost、ms-service-profiler、MindStudio KPP 的缺失 Python 依赖。最终 `pip check` 仅剩一项：`mindstudio-kl 26.2.0` 要求 `numpy<=1.26.4`，当前核心栈为 `numpy 2.5.3`。保留核心版本，MindStudio KL 在该 overlay 中未完成兼容验证。具体安装清单和原始检查输出分别见 `logs/python-freeze-final.txt`、`logs/pip-check-final.txt`。

当前 MegaMoE 是推理 FP8×FP4 路径，BF16 输入会内部量化；没有反向/autograd。两机只有16卡，GLM EP8 用例每卡32个路由专家，不能据此宣称64卡/每卡4专家的训练方案已经验证。用户新增 M65536 的 alltoallvGmm/gmmAlltoallv 大用例未运行。没有进行大型 benchmark。

在 19:55:55/58 +08:00 最终检查中，两机 `npu-smi` 的8张卡均显示 `No running processes`；本次 example/benchmark/golden 进程列表为空。未发现 foreign NPU 任务，也未终止外部进程。环境已释放。

## 小型证据归档

只打包安装/构建/验证日志、版本和哈希、脚本、逐 rank 精度 JSON；不包含巨大 SDK、wheel 或源码全集。

- `environment_evidence_rack05os7.tar.gz`：SHA256 `76ea27bb730f2f99e1f209d8f75e89738809438ac0af5e881f38dd9b7b1a856c`。
- `environment_evidence_rack05os8.tar.gz`：SHA256 `e0c097f3e2f960f8e71febb6bc430b6c96b76fdfe13759f768f1b16ef6c9dbca`。

远端亦保存 `/home/zzy/deploy/megamoe-20261024/evidence.tar.gz`。其中 `logs/accuracy-summary.json`、`logs/npu-final.txt`、`logs/remaining-test-processes.txt`、`logs/final-import-paths.txt`、`logs/loaded-libraries-native.txt`、`logs/wheel.sha256`、`logs/shmem-sdk-manifest.json` 是主要核查入口。
