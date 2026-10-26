# 无NPU占用条件下的训练代码review

范围：新增 `experiments/megamoe_training`，以及DeepEP固定提交 `35fa7d91b321340c153b098bf166202645976a75` 的 `ep_gmm_fused`。本轮未启动设备任务；数学回归使用真实CPU PyTorch与通信/GEMM stub，不能替代NPU精度验收。

## 确认的问题及处理

### 1. P1：旧梯度报告可能放行改过代码或二进制的新版本——已修复

原 `tests/benchmark_compare.py::require_training_qualification` 只核对shape、world、BF16、backend及passed。完成一次验证后，修改training.py或替换CATLASS库，旧报告仍可放行性能测试；优化阶段可能绕过反向回归。

新增 `tests/provenance.py`：身份覆盖训练/测试oracle与阈值源码、CATLASS实际库、DeepEP Python/core、实际绑定SHMEM、CANN已加载库、Torch/PTA版本及读回的精度开关。当前身份必须匹配所有rank报告；缺身份、源码/库变化、缺加载证据均拒绝，错误先通过已有组传播，避免一rank先退出、其他rank继续进入性能collective。绝对路径仅供诊断，不进入跨机内容digest。

CPU实测还发现主 `libopapi.so` 为lazy load，正常导入时并未映射。已修复collector：仅从核实的正式CANN根显式加载所缺库，再验证实际映射；保留已加载的 `libopapi_transformer` 等分库身份。显式加载不等于证明某算子从该SO执行，最终仍需runtime/profiler归因。

15项stdlib回归通过。两机真实CPU collector成功，内容digest均为 `a211e4e826c4d0bf40c1da141a9bce73cd0b1241a54247693e15e1b061898895`，前后NPU均未初始化。

### 2. EP16适配阻塞：H6144默认前向工作区越界——必须重设分块

前向Host文件 `all_to_allv_grouped_mat_mul_udma.cpp:72` 默认 `stageRows=1280, workspaceStages=2`；`:96–119` 的选择逻辑对本次M65536/K6144/N4096/E64走默认profile。`tiling_validation.hpp:17,153–158` 将数据区限制为400MiB。

若只放宽EP8门禁，EP16需要 `1280 × 16 × 6144 × 2字节 × 2stage = 480MiB`，必然被容量校验拒绝；不能删除校验强行启动。相同布局下，stageRows对齐128后的上限为1024，对应384MiB。1024仅是容量上可行的候选，仍需验证控制区、调度、通信计数和性能，尚未作为已验收配置。

这是EP16新规格的兼容性阻塞，不是宣称原版EP8越界。原版保持不变，适配版单独开发。

### 3. EP16适配阻塞：逆向默认通信轮次可能不能覆盖N6144

逆向 `grouped_mat_mul_all_to_allv_udma.cpp:55` 默认 `N0=256, commInterval=10`；`:215–228` 以及 `tiling_validation.hpp:183–191` 要求每rank一轮有至少 `ceil(N/N0)` 个tile。

本规格需要24个N tile。在32个AIC条件下，`32 × 10 / 16 = 20 < 24`，仍会拒绝。应先核实设备实际AIC数，再求合法的comm_interval或显式stageRows；32 AIC、当前tile的最小interval为12。这是条件明确的静态推导，不是硬件核数或最优性能断言。

同时不能只改rank模板：路由metadata由Python固定offset和C++结构体共同定义。保持max_local_experts=64时，rank上限8→16将metadata从12880B变为25232B，Python打包、C++静态断言、结构布局和所有调用方必须一致。

### 4. 接入风险：ep_gmm权重布局与训练API相反

两方向Host分别在 `all_to_allv_grouped_mat_mul_udma.cpp:272–293`、`grouped_mat_mul_all_to_allv_udma.cpp:162–182` 要求 `[E_local,K,N]`；训练API和用户规格是 `[E_local,N,K]`。不能原样接线。非方阵会被K检查拒绝，方阵则可能掩盖转置错误，必须由独立oracle检验。

优先评估添加直接消费训练布局的内核变体；若使用transpose+contiguous桥接，完整训练计时必须处理每次权重更新后的转换/缓存失效成本，并还原梯度布局。算子prepared-input分项可另列，不能把预先转好的静态权重成本隐藏在训练端到端比较之外。

## 已排除的疑点与验证边界

实际PTA2.12.0.post2的Python `.npu_stream` getter调用默认 `NPUStream::stream()`，该路径会清空先前host task queue再返回ACL stream。源码及真实libtorch_npu.so反汇编均相符。在当前managed stream、单调用线程的条件下，不能将ctypes raw launch直接判成乱序bug。它不等于device synchronize；`record_stream`负责allocator生命周期，不提供计算依赖。外部stream、多线程和graph capture仍需单独验证。详细证据见 `training_cpu_stream_review.md`。

实际 `training.py` 的CPU autograd回归4/4通过：只在AST副本中放开NPU设备比较，通信和GEMM用CPU stub，独立手写BF16边界oracle核对Y、dx、dp、FP32 dW和leaf梯度。覆盖零gate仍有dp、空专家、FP32 master/shared、两种路由连续forward后逆序backward及提前close拒绝。当前未发现这些路径的Python公式、转置或ctx串扰错误。

这不证明CATLASS真实数值、跨机通信、设备内存/异步生命周期、65536行或EP16流水正确。二阶梯度明确不支持。FP32 master返回FP32累加dW是本实现的训练契约，不应描述为与先产生BF16参数梯度再转FP32的所有autocast实现逐位一致。

## 训练规格下优先验证的优化点

1. **先让输入梯度回传与wgrad重叠。** `training.py:68–73` 当前先算dW2再dA，先算dW1再dX，最后才combine。dW依赖已保存的激活和对应上游梯度，并不需要挡住整条dX关键路径。精度通过后可优先调度dA→dGU→dX，用通信stream回传dX，同时计算FP32 dW；完成事件、tensor生命周期、buffer复用必须明确。AIV/AIC及HBM是否能实际并行需要打点与无打点对照，不预报加速比。
2. **减少Python逐专家提交与host queue drain。** 四专家×六GEMM每层至少24次raw launch，每次都读取`.npu_stream`。可先改为一次host封装内调度分组，或按PTA的 `stream(false)+OpCommand::RunOpApi` 方式正确入队，再考虑device grouped scheduler。不能单纯缓存raw handle省掉必要的队列顺序。
3. **融合SwiGLU、导数和gate计算，减少大FP32中间张量。** 单份 `[65536,6144]` 的FP32临时量就是1536MiB；当前dp表达式要将dY/Z转FP32再相乘，gate和激活也有大量往返。融合时仍需保留约定的G/U、A、Z BF16舍入点。保存的未加权Z用于dp，尤其p=0时不能改成除p或用已加权dA替代。
4. **控制训练保存态和调试梯度保留。** 本规格X_e、G/U、A、Z共2304MiB/rank；`last_grads_fp32`再持有576MiB/rank。FP32 master的leaf梯度还可能与调试保留造成额外存储。生产路径应将这种保留变成显式诊断选项；保留/重算策略须单独核算显存、执行FLOPs和dp精度，不能直接删Z称为无损。
5. **针对每rank四专家裁剪metadata/调度容量。** EP16本规格和目标EP64/E256均是local_experts=4，可以评估专用容量变体，避免扩大rank后仍为64本地专家保留大量数组。需保留明确的通用回退和能力边界，不能用固定形状假设覆盖未来用户规格。

这些优化目前只是源码支持的候选方向。排序需结合通信实际payload、纯GEMM效率、关键路径等待、MFU及打点扰动实测；不凭代码顺序或总时长断言瓶颈占比。

## 本轮交付与凌晨入口

修复仅涉及测试资格/溯源与CPU回归，未改主数学实现、SHMEM同步协议或CATLASS原生库。新review源码16文件/11 Python已单独同步两机，旧13文件ZIP与旧环境bundle保持原样。原生SO仍为 `f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32`。

新入口为 `/home/zzy/deploy/megamoe-20261024/training/activate-review.sh`，源码根为 `training/review-20261024/experiments/megamoe_training`，启动器为 `scripts/run_training_review_checked.sh`；旧启动器已转发。后续若修改native源码，必须在新源码根重新构建并记录构建输入/产物哈希，不能只复制旧SO再宣称新内核已生效。

已更新同一凌晨heartbeat：读取本review和 `training_review_runbook.md`，03:00检查占用、冲突每10分钟复查，先重新生成绑定当前版本的NPU完整梯度资格，再做融合流水和性能优化。
