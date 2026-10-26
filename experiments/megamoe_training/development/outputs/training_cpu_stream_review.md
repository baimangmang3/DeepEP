# CPU review: stream ordering and training mathematics

本轮仅 CPU；没有 set_device、NPU tensor、ElasticBuffer 实例或通信域。旧训练快照、wheel、归档未覆盖。新增 review 源在 rack05os7 的 `/home/zzy/deploy/megamoe-20261024/training/review-20261024/experiments/megamoe_training`。

## Stream ordering

不能将当前 ctypes GEMM 路径判为已证实的 PTA host queue 乱序缺陷。实际安装 torch_npu 为 2.12.0.post2，git `63f87eb51580e2664061cdfdd2375b528ca02ad2`。该版本 Python `.npu_stream` getter 调用默认 `NPUStream::stream()`。[绑定源码 Stream.cpp:84–87](https://github.com/Ascend/pytorch/blob/63f87eb51580e2664061cdfdd2375b528ca02ad2/torch_npu/csrc/npu/Stream.cpp#L84)

默认 `stream()` 在普通 managed stream 上等待已初始化的 PTA host task queue 清空：per-stream queue 模式清本流，否则清 default queue；sync-launch stream 和 external stream 有不同分支。`stream(false)` 则直接返回 handle，`stream(true)` 调用默认版本。此处等待 host 下发，不等同于 device synchronize。[NPUStream.cpp:390–417、785–793](https://github.com/Ascend/pytorch/blob/63f87eb51580e2664061cdfdd2375b528ca02ad2/torch_npu/csrc/core/npu/NPUStream.cpp#L390)；[NPUQueue.cpp:224–275](https://github.com/Ascend/pytorch/blob/63f87eb51580e2664061cdfdd2375b528ca02ad2/torch_npu/csrc/core/npu/NPUQueue.cpp#L224)

实际 `libtorch_npu.so` SHA256 为 `b8dbe2eadae0f7ae5d4bafc2a5617d65fc84172daf5fff6a1d25efe2dd970e48`。从真实 getset descriptor 取得 getter 地址，经 dladdr 换算 ELF 地址 `0x4b8b2b0`；`0x4b8b2ec` 调用 `NPUStream::stream()@0x18cba10`，随后调用 PyLong_FromVoidPtr。只检查类元数据，没有创建 stream 或执行 getter；NPU initialized=false。详见 evidence 下 identity/disassembly。

`gemm.py:48–50` 参数依次求值：data_ptr 后读取 `.npu_stream`，继而 ctypes 调用。因此在正常 managed stream、同调用线程、无未声明跨流依赖的条件下，先前 PTA host tasks 已下发，再把 raw GEMM 加入同 ACL stream；之后调用的 PyTorch ops 晚于 raw launch。这是源码/二进制顺序结论，尚无设备回归。`record_stream` 用于分配器生命周期，不承担前述排序。

可落实的性能改进点是每次 expert GEMM 都经过 host queue drain；更合适的生产包装可参考 EP-GMM 的 `stream(false)` + `OpCommand::RunOpApi`，避免 Python 循环频繁 drain。其 RunOpApi 在 task queue 启用时将 handler 放入 EXECUTE_OPAPI 队列，否则直接执行。[OpCommand.cpp:170–213](https://github.com/Ascend/pytorch/blob/63f87eb51580e2664061cdfdd2375b528ca02ad2/torch_npu/csrc/framework/OpCommand.cpp#L170)

Pinned DeepEP 的实际调用位置：MegaMoE `experiments/megamoe/csrc/forward.cpp:451–454` 默认 stream()；EP-GMM forward `csrc/all_to_allv_grouped_mat_mul_udma/ascend950/all_to_allv_grouped_mat_mul_udma.cpp:190,254` 与 reverse `csrc/grouped_mat_mul_all_to_allv_udma/ascend950/grouped_mat_mul_all_to_allv_udma.cpp:265,308` 使用 stream(false)+RunOpApi。并未把这些调用习惯当作实际性能结果。

## CPU regressions

实际 CPU torch `2.12.0+cpu`。runner 禁止真实 torch_npu 导入，允许测试注入 mock。命令：

```bash
T=/home/zzy/deploy/megamoe-20261024/training
TORCH_DEVICE_BACKEND_AUTOLOAD=0 timeout 120 /root/miniconda3/envs/zzy/bin/python \
  "$T/scripts/run_cpu_regression.py" \
  "$T/review-20261024/experiments/megamoe_training/tests/test_training_math_cpu.py" -v
```

新增 test_training_math_cpu.py：4/4 passed，rc0，0.355s。执行真实 training.py 的 _RoutedMoE/_SharedExpert；只在 AST 副本中将 _validate 唯一 NPU device 比较改为 CPU。CPU buffer stub 按专家排序/展开、combine 求和及逆置换 dp；CPU GEMM stub 保留 BF16 输出边界和 FP32 accumulation。独立显式导数 oracle 核对 Y、dx、dp、FP32 dW1/dW2 和叶子梯度。

覆盖 p=0 且 dp 非零、空专家、FP32 master+shared、同一 engine 两种路由 forward 后逆序 backward、close 后 backward 拒绝。测试文件不纳入固定 production identity。结果只验证 Python 数学/上下文；不验证 NPU/CATLASS/SHMEM，不可用作 qualification。

test_provenance.py：15/15 passed，rc0，0.069s。

## Real loaded identity CPU probe

初始 collector 可复现失败：主 libopapi.so 尚未映射，尽管正式 9.2 的 runtime/ACL 与 libopapi_transformer 等分库已经映射。修复 helper 后，仅从 ASCEND_HOME_PATH/lib64 显式 CDLL 所缺 mandatory 主库，保留句柄，再核实真实 maps。

CPU probe 复验 passed=true；before/after npu_initialized 均 false。实际主库路径 `/home/zzy/deploy/megamoe-20261024/cann-env/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so`，由正式根 lib64 symlink 解析得到。身份 digest `a211e4e826c4d0bf40c1da141a9bce73cd0b1241a54247693e15e1b061898895`。显式加载只使 manifest 完整，实际算子后端仍需运行/tiling/profiler 证明。

证据在同目录 `training_cpu_review_evidence/`：math/provenance logs、identity JSON、getter identity/反汇编、review source SHA 清单。CPU probe/import 不代表硬件空闲；本轮没有检查或干预 foreign NPU 作业。
