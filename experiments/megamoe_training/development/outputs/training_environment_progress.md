# 训练环境阶段记录

2026-10-24 20:03 CST 首查两机16张 Ascend950DT 全部健康、空闲；20:12:33/34 CST 在单卡测试前复查发现16卡均被 foreign EP64 Dispatch profiling 作业占用。启动门禁已停止本次NPU测试，未终止外部进程。按用户规则等待主任务设置10月25日03:00起、每10分钟重查的调度。继续复用已验证正式 CANN9.2.0，不重复安装。

- 独立训练根目录：`/home/zzy/deploy/megamoe-20261024/training`；两机均已建立 `DeepEP/` clone、独立 `venv/` 和 `activate.sh`，保留推理环境。
- 两机原 SHMEM SDK manifest 完全一致，可以构建一次并同步同一候选产物。
- `ep_gmm_fused` 固定 CatCCOS `878c2e0a504be8c49c03b5e1157a7c714a628e29`、CATLASS `db93081ce9b0ad6c04e99b9ed5f2fba081ccdb00` 及 googletest 已准备；按新独立训练实现方案不额外编译该整包。
- `experiments/megamoe_training` 首批 CATLASS ctypes BF16 GEMM 已在27机原样编译成功，`training/logs/native-build.rc=0`；NN/NT/TN、BF16/FP32输出和zeroK小用例脚本已准备，设备验证被占用门禁拦截，尚未执行。

原生 CANN 基线审计：

- 当前 torch_npu2.12.0.post2 实际注册 `npu_alltoallv_gmm` / `npu_gmm_alltoallv`，均有 `comm_mode` 参数；dispatch表为 PrivateUse1 + Meta，未见专用 Autograd 注册或对应 grad schema。这不是反向可用性的设备结论。
- 正式 CANN9.2 的 libopapi 实际导出两方向和 V2 接口，安装了 ascend950 BF16 kernel/config。
- 安装源码 `allto_allv_grouped_mat_mul.cpp` 的 AIV 分支仅允许 `__CCE_AICORE__==220`；其他架构明确 `trap()`，注释说明 Host 会拒绝不支持架构。950 的普通分支走 HcclA2avOp；其 GMM→AllToAllV tiling key仅列 CCU=0、AICPU=1。
- V2 aclnn头文件将 `commMode` 声明为 `""`、`aicpu`、`ccu`。AIV不能通过简单改环境变量视为950基线，后续将记录真实拒绝和profiling所选kernel/tiling。

占用归属：27机NPU PID3870261–3870268、父3870047；28机PID3917137–3917144、父3916977。命令为既有 `/root/miniconda3/envs/deepep-copy/bin/python tests/elastic/test_dispatch.py --ep-size64 --master-addr177.22.21.1 --master-port19090`，输出至 `/home/y00889330/perf-part1-ep64-rerun/`，不属于本次任务。

原始审计和构建日志位于远端 `training/logs/`。未运行M65536主case、EP16训练或性能对比；等待空闲窗口与本地训练链路代码继续同步。
