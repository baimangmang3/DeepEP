# MegaMoE 训练版开发方案与 GLM5.2 64 卡规格

建议新增独立的 BF16 训练入口，使用 SHMEM 完成专家通信、CATLASS 完成前向和反向分组矩阵乘。先完成前向、dX、dW、路由权重梯度及共享专家的完整正确性，再按 tile 重叠通信与计算。当前推理入口的 FP8/FP4 量化、同长 batch 和短生命周期工作区不能直接作为训练语义。

本方案以 **EP=64 的单层 MoE 算子**作为首要设计目标；“64 卡完整模型训练”的 EP/TP/PP/DP/CP 划分、优化器和显存策略等待补充。两台现有机器合计 16 张 Ascend950DT，可完成 EP8/16 验证，不能替代 64 卡验收。分析日期为 2026-10-24。

## 已核实的模型规格

| 项目 | 官方配置 | 算子含义 |
|---|---:|---|
| Hidden size H | 6144 | Dispatch、输出和 dX 的宽度 |
| MoE intermediate I | 2048 | SwiGLU 后宽度；GMM1 输出 4096 |
| 路由专家 E | 256 | EP64 下每 rank 4 个专家 |
| 每 token 专家数 K | 8 | 每 token 最多 8 个专家贡献 |
| 共享专家 | 1 | 独立 SwiGLU MLP，与路由输出相加 |
| 主干层数 | 78，前 3 层 dense | 主干 75 个 MoE 层；MTP 另行处理 |
| 权重公开 dtype | BF16 | 训练首版选择 BF16；不从推理权重量化推导训练精度 |
| Router | FP32、sigmoid、归一化 TopK、scale 2.5 | 返回对最终路由权重的梯度 |
| 最大位置长度 | 1,048,576 | 不能直接当作每卡每次 MoE 的 token 数 B |

来源：[Z.ai 配置固定版本 cf457fa](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json)。配置快照 SHA256 为 `185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a`。

已阅读 [Transformers 对应实现](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L453)：gate/up 后做 SiLU 乘法，down 输出再乘路由权重；共享专家直接相加。路由校正 bias 用于选专家，实际权重来自未加 bias 的 sigmoid 分数。训练中 bias 更新、辅助损失和优化器策略仍由上层决定，公开加载实现不代表完整预训练方案。

## 用户补充的 profiling 主规格

2026-10-24 补充的 ops-transformer 算子规格作为开发主 case，优先级高于前面的通用 B 扫描。原始形状保留如下；dtype、格式、transpose 属性和 counts 尚未给出。

| 算子 | 输入 X | 专家权重 | 输出 |
|---|---|---|---|
| alltoallvGmm | [65536,6144] | [4,4096,6144] | [65536,4096]；[65536,6144] |
| gmmAlltoallv | [65536,2048] | [4,6144,2048] | [65536,6144] |

这与每 rank 4 专家、H6144、I2048 的完整 gate/up/down 结构一致。以全局 E256 推算 EP64 是一致的解释，仍需确认是否有额外专家/张量并行分片。权重逻辑为 `[E,N,K]`；当前 ep_gmm_fused 的接口要求 `[E,K,N]`，必须在适配层显式处理格式和转置，不能只靠形状匹配直接传入。

alltoallvGmm 第二输出 `[65536,6144]` 暂按 dispatch 后的 X_e 理解，对应参考实现的 export 路径，能直接供 dW1 和激活重算使用。最终要以原算子的输出名称、属性和数据顺序确认，单靠 shape 不能证明其语义。65536 是整个 rank 的分组行数，不能给 4 个专家各分配 65536 行。若输入是 Top8 全展开且无 padding/drop，则源 B=8192；若 4 专家均衡则各 M=16384，这两个条件都不是 profiling 形状本身证明的事实。

在该展开解释下，gmmAlltoallv 的 `[65536,6144]` 仍是返回源 rank 的专家路由行，后面还需按原 token/TopK slot 做加权 unpermute，才得到 `[8192,6144]`。端到端训练接口和性能对比必须覆盖这一步及前面的 permute，不能把这两个融合算子输出的行数直接视为最终 token 数。

按每专家 M=16384 的候选均衡 case，六类 GEMM 为：

| 阶段 | 左矩阵 | 右矩阵 | 结果 |
|---|---|---|---|
| FWD gate/up | [16384,6144] | [6144,4096] | [16384,4096] |
| FWD down | [16384,2048] | [2048,6144] | [16384,6144] |
| BWD dA | [16384,6144] | [6144,2048] | [16384,2048] |
| BWD dX | [16384,4096] | [4096,6144] | [16384,6144] |
| BWD dW2 | [6144,16384] | [16384,2048] | [6144,2048] |
| BWD dW1 | [4096,16384] | [16384,6144] | [4096,6144] |

每 rank 共 4 组。按一次乘加为 2 FLOPs，路由专家前向约 4.9478 TFLOPs、反向约 9.8956 TFLOPs，合计 14.8434 TFLOPs；不含共享专家、激活、路由、通信、重算及低精度额外操作。以上为运算量，不是吞吐或延迟实测。

若张量为 BF16，X_e 和 down 输出各 768 MiB，gate/up 输出 512 MiB，激活 A 为 256 MiB；FP32 dW1/dW2 分别 384/192 MiB。完整保存 X_e/G/U/A/Z 共 2304 MiB/层/rank，必须使用有界 staging，不能把整批作为每槽通信缓冲。

开发先对齐这组前向布局和 export 输出，再完成六类 GEMM 与 dp/shared 分支的逐项精度验证。当前 16 卡可用每 rank 4 专家、总 E64 的缩小拓扑 case 保持局部 GEMM 形状，或用 E256/EP16 验证模型路由；两者都必须标记配置，均不能替代 E256/EP64 的真实通信验收。

## 两套参考代码的可复用边界

源码固定在 DeepEP `35fa7d91b321340c153b098bf166202645976a75`；SHMEM `4dd80e2129da42bd773982c2bb3758af1ed0bba8`；CATLASS `2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1`。

| 实现 | 可以复用 | 训练必须新增或修改 |
|---|---|---|
| MegaMoE | SHMEM bootstrap、对称区预算、compact route、逐 tile ready、wave 调度、共享专家重叠、输出归并 | BF16 GEMM、完整 backward、训练 context、可变 B/零 token、跨调用窗口所有权 |
| ep_gmm_fused | BF16 CATLASS GEMM、AllToAllV→GMM 与 GMM→AllToAllV 的 staging、dispatch 导出、UDMA completion | dW 的转置归约 GEMM、autograd、TopK 归并、EP64 调度与拓扑扩展 |

关键源码事实：

- [MegaMoE API](https://github.com/lueelu/DeepEP/blob/35fa7d91b321340c153b098bf166202645976a75/experiments/megamoe/python/deep_ep_experimental/megamoe/api.py#L468) 显式拒绝 `requires_grad`；BF16 输入会进入量化路径，不等于 BF16 GEMM。
- [ProcessWave](https://github.com/lueelu/DeepEP/blob/35fa7d91b321340c153b098bf166202645976a75/experiments/megamoe/csrc/kernels/mega_moe/mega_moe_arch35.hpp#L908) 组织输入准备、共享 GMM1、路由通信、专家 wave、共享尾部和最终跨卡握手；[A8W4 wave](https://github.com/lueelu/DeepEP/blob/35fa7d91b321340c153b098bf166202645976a75/experiments/megamoe/csrc/kernels/mega_moe/mega_moe_wave_a8w4.hpp#L475) 交错下一 wave 的 Dispatch 与当前 GMM1，再消费 GMM2/Combine。
- [ep_gmm_fused runtime](https://github.com/lueelu/DeepEP/blob/35fa7d91b321340c153b098bf166202645976a75/experiments/ep_gmm_fused/python/deep_ep_experimental/ep_gmm_fused/runtime.py#L131) 和 native 拓扑检查限制 EP≤8。其“64 卡 EP4”示例是 16 个独立小域，不是 EP64。
- 正向最多 64、逆向最多 32 个本地专家，见 [正向](https://github.com/lueelu/DeepEP/blob/35fa7d91b321340c153b098bf166202645976a75/experiments/ep_gmm_fused/csrc/all_to_allv_grouped_mat_mul_udma/ascend950/all_to_allv_grouped_mat_mul_udma.cpp#L32) 和 [逆向](https://github.com/lueelu/DeepEP/blob/35fa7d91b321340c153b098bf166202645976a75/experiments/ep_gmm_fused/csrc/grouped_mat_mul_all_to_allv_udma/ascend950/grouped_mat_mul_all_to_allv_udma.cpp#L119)。因此 GLM 的 E256 在 EP8 恰好 32 专家/rank；EP4 的 64 专家/rank 不能直接使用现有逆向算子。
- [通信 completion](https://github.com/lueelu/DeepEP/blob/35fa7d91b321340c153b098bf166202645976a75/experiments/ep_gmm_fused/csrc/catccos_overlay/include/catccos/comm/block/comm_block_mte_udma.hpp#L49) 先排空 owner 的 UDMA 请求再发布完成信号；普通 fence 不能替代 UDMA quiet。

现有 MTE/UDMA runtime 共用单进程 SHMEM owner。研发阶段先用独立作业分别验证两套路径；新训练模块统一持有一个通信 session，不并存初始化两个不兼容的域。

## 训练接口和数据契约

建议建立 `experiments/megamoe_training`，保留现有推理入口行为。下列为拟定接口，尚未实现：

```python
y = train_moe(x, topk_idx, topk_weight, w_gate_up, w_down,
              shared_w_gate_up, shared_w_down,
              plan=plan, checkpoint="save", deterministic=False)
# autograd: dx, dtopk_weight, dw_gate_up, dw_down, dshared_weights
```

逻辑布局采用 PyTorch 权重习惯：`x[B_r,H]`、`ids[B_r,K]` int32、`p[B_r,K]` FP32、`W1[L,2I,H]`、`W2[L,H,I]` BF16。GMM1 的 2I 为先 gate 后 up。CATLASS 具体布局由适配层转换；dX 需要的转置视图和 dW 需要的归约布局应成为 kernel 模板参数，避免每 microbatch 复制整套权重。若必须预打包，按 parameter version 缓存并在 optimizer step 后失效。

`Plan` 保存 EP 成员映射、实际网络拓扑、dtype、最大 token/capacity、SHMEM 内存和 tiling。每次 `Context` 保存本次 `B_r`、route、counts、prefix、激活或重算输入、parameter version 和 completion event。允许非连续全局 rank 组；全局 rank、EP rank、local device 分别记录。

每条专家路由必须保存 `(source_ep_rank, source_token, topk_slot, expert_id, expert_row)`。发送端去重与接收端专家展开是两个层次：通信可复用同一 token，GMM 仍需每专家的一行。反向使用 forward 的映射，不重新执行 TopK，不按当前权重重算路由。

首版支持不等长 `B_r`、`B_r=0`、空专家、尾 tile 和偏斜路由；正常训练同 token 不允许重复专家 ID。所有 rank 即使本地无工作也必须参与控制协议。默认不丢 token；容量不足走全组一致的扩容/分块路径，禁止静默裁掉路由。H、I 的泛化按原生 128 倍数组织，具体上界待补充，不通过改变逻辑尺寸掩盖不支持的规格。

路由选择和辅助损失留在上层：算子返回 `dp`，框架继续反传 sigmoid/归一化及 router 参数；router 对 X 的梯度由 autograd 与专家 dX 相加。ids 不求导。可选 FP32 `main_grad` 累加接口需明确首次清零与多 microbatch 累加，避免与框架 `.grad` 重复累加。

## 前向和反向算法

对某专家 e 收到的 token 矩阵 X，令 Wg、Wu 为 W1 的两半，Wd=W2；p 为每行对应的路由权重。

```text
G = X @ Wg.T                 U = X @ Wu.T
A = SiLU(G) * U              Z = A @ Wd.T
Y[token] += p * Z            Y += shared_mlp(original_X)
```

给定源 token 的上游梯度 D=dY：

```text
dZ = p * D
dp = sum_H(D * Z)                  # 不乘 p，p=0 时仍可能非零
dWd = dZ.T @ A
dA = dZ @ Wd
dG = dA * U * sigmoid(G) * (1 + G * (1 - sigmoid(G)))
dU = dA * SiLU(G)
dWg = dG.T @ X                     dWu = dU.T @ X
dX_expert = dG @ Wg + dU @ Wu
dX[token] = sum_over_routes(dX_expert) + dX_shared
```

共享专家走相同 MLP 反向，直接使用 D，不乘路由权重。每个专家 dW 仅由拥有它的 EP rank 累积；不能对不同专家的 dW 在 EP 域直接 AllReduce。若同一专家跨外层 DP/E-DP 有副本，再按训练框架要求做对应参数的归约。共享专家若在各 EP rank 复制且处理不同 token，其 dW 必须在实际共享参数副本域归约；TP/CP/损失归一化会改变归约组和缩放，交由上层明确。

固定 TopK 集合内，s=sigmoid(logit)、q=sum(selected s)+epsilon、p_i=2.5*s_i/q：

```text
ds_i = 2.5/q * (dp_i - sum_j(dp_j*s_j)/q)
dlogit_i = ds_i*s_i*(1-s_i)
```

未选中 logits 对该路由任务损失的梯度为零；辅助损失另算。不对离散 TopK 做未经定义的直通估计。

GEMM 使用 BF16 operand、FP32 accumulation；dW 持久累加为 FP32，TopK sum/激活导数/路由梯度尽量保留 FP32。必须锁定各中间 cast 位置并与训练框架 oracle 对齐，FP32 中间结果何时转 BF16 属于接口精度语义。FP8 训练和 FP4 权重训练放在后续独立阶段，需另定义 scale、反向 dtype 与误差标准。

## 通信与计算流水

先实现可诊断的多 kernel 基线，再融合为有界 chunk 的流水。forward 与 backward 共用路由计划，但持有各自的工作区 epoch。

| 阶段 | AIV 与 SHMEM | AIC 与 CATLASS | chunk 完成条件 |
|---|---|---|---|
| F0 | 统计 counts、prefix、打包路由 | 可执行本地共享专家 | 元数据和所有源输入可见 |
| F1 | 接收/读取 X chunk j+1，发布 ready | GMM1 chunk j | X 数据传输完成且有效行数确定 |
| F2 | 对 G/U 做 SwiGLU，保存或重算策略记账 | GMM2 处理已激活 chunk | 对应 G/U 全部就绪 |
| F3 | p 加权、返回 Z，源 rank FP32 TopK 归并 | 继续下一计算 chunk | 返回数据完成；源 token 收齐 K 个贡献 |
| B0 | 按 forward route 发送未加权 D 及所需 p | 可执行共享专家反向 | D、route、保存的激活匹配同一调用 |
| B1 | 计算 dp，SwiGLU backward | dA=dZ@Wd | dp 使用未加权 D；dA 使用 dZ |
| B2 | 将 dX 返回源 rank 并累加 | dX=dGU@W1 | dX 远端可见后发布完成 |
| B3 | 空闲通信窗口可继续预取 | dW2=dZ.T@A、dW1=dGU.T@X | 本专家全部 chunk/microbatch 归约完成 |

同一 AIC 的 dX GEMM 和 dW GEMM 会竞争计算资源，不能把它们当作天然并行。优先保障 dX，缩短上一层反向等待；dW 按预算穿插或延后，但 optimizer/梯度归约必须等待 dW event。延后 dW 会延长 X、A、dGU 的生命期，内存紧张时改为逐 chunk 完成 dW，再释放激活。

dW 的归约维度是专家 token 数 M，与前向的 H/I 归约不同。先实现每个输出 tile 单 owner 顺序遍历 M，得到确定性 FP32 累加；需要并行 Split-K 时写 FP32 partial 后树形归并。禁止用无约束原子累加作为唯一精度/确定性路径。空专家 dW 明确为零，不能保留上一步内容。

初始候选 tiling 为 M=128/256、N=128/256、K=256，按 forward、dgrad、wgrad 分别选择。M 尾块和偏斜不依赖全局补齐；通信 chunk 从 256/512/1024 行的小范围测试起步。2 槽作为基线，只有逐 rank 等待数据证明有用才尝试 3 槽。

### EP64 的扩展方式

不能只放宽 `world_size<=8`。现有通信调度把 peer 数与 AIC/AIV worker 数关联；EP64 要把 peer 作为逻辑队列，由固定数量 worker 轮询 ready 的 peer/chunk，消除“一 peer 一 core”的硬条件。所有计数、工作区乘法和 offset 检查 64 位溢出。

先验证 950DT 实际 peer 可达性。若 64 卡同处可互访超节点，优先直接 peer 通信；跨服务器不自动意味着必须 RDMA。需要网络传输时，使用同一 SHMEM session 中明确支持的 UDMA 路径，遵循真实队列/完成语义。跨节点层次转发为后续选择：先节点聚合、跨节点一次发送、目的节点展开；只有带宽/注入压力收益抵过中转读写和屏障才启用。

初版四次数据交换为 X Dispatch、Z Combine、D Dispatch、dX Combine。按 BF16、完全专家展开、均衡、包含本地逻辑搬运计算，每 rank 共 `4*K*B*H*2` 字节；B=1024 时为 384 MiB，加上 route/count/dp。此数不是物理网络实测，也不重复计收发两端；实际网络需扣除本地目标并计算去重。dp 额外为 `B*K*4` 字节，可随返回元数据传送。

### 有界缓冲和正确性协议

一个槽的状态为 `FREE → WRITING → READY → READING → FREE`，携带调用 epoch、阶段、peer、chunk、有效行数。数据传输 completion 在 ready 之前；计算读完、保存激活导出完和所有消费者退出后才回收槽。forward 结束不代表 backward 所需激活可覆盖，持久 context 与通信 staging 分开分配。

UDMA nbi 的发起不等于远端可见；由队列 owner quiet/complete 后发布 ready。MTE/跨核 flag 与缓存可见性按实际 SDK API 实现，不能把本地 pipe barrier 当作跨卡 completion。复用现有异常 dump，增加 `(epoch,phase,peer,chunk,expected,observed)`，对等待设有限超时和全组失败退出。零 payload 也发送完成元数据；不得因空 rank 跳过公共协议而死锁。

初版保留必要的输入/输出全组握手。取消全局 barrier 属于后续单独优化，须以每块 credits 和 token 完成计数替代，逐项证明其保证。图捕获阶段使用预分配容量、设备侧动态 counts、稳定地址和可回放 epoch；当前推理图捕获限制不能靠包装 autograd 自动解除。

## 显存和重算

EP64 下每 rank 每层路由权重有 `4*3*6144*2048=150,994,944` 个参数，BF16 权重占 288 MiB。以 BF16 权重、FP32 梯度、FP32 master、两个 FP32 Adam moment 为示例，共 18 B/参数，即每层 2.53125 GiB，75 层仅路由专家就为 **189.84375 GiB/rank**，超过现有每卡 96 GB。实际优化器未确定，这里是清晰的容量反例，不是对 GLM 官方优化器的断言。

因此，64 卡单层算子可开发验证；完整模型训练仍需参数/优化器卸载、更低状态精度或更多设备等整体策略。单纯调整 TP/PP/EP 不会消除所有参数训练状态的总容量需求。

下表假设各 rank B 相同且路由完全均衡：每 rank 接收 8B 行，每专家 2B 行。保存 X_e、G、U、A、Z 均按 BF16；未含共享专家、源 X、梯度、通信窗、padding 和分配器开销。

| 每 rank B | 接收行数 | 保存 X_e/G/U/A/Z | 仅保存 X_e |
|---:|---:|---:|---:|
| 128 | 1024 | 36 MiB | 12 MiB |
| 1024 | 8192 | 288 MiB | 96 MiB |
| 4096 | 32768 | 1152 MiB | 384 MiB |
| 8192，条件推导的 profiling case | 65536 | 2304 MiB | 768 MiB |

偏斜情况下不能按均值分配：一个专家最多收到全组所有源 token；一个 rank 的上界还取决于本地专家数与 TopK。先统计真实 counts，再选择容量或分块；若上下游要求固定地址则按声明的容量契约分配并显式处理溢出。

重算模式有两种精度契约：

- 严格模式保存 forward Z，或保存 X_e 后重新计算 GMM1、SwiGLU 和 GMM2，复现 forward Z 的舍入。需冻结相同的参数版本并匹配计算确定性。
- 可选快速模式利用实数恒等式 `dp=<D,Z>=<D@Wd,A>`，用未乘 p 的 dA 求点积，不保存 Z。BF16 forward Z 的舍入会破坏逐位等价，必须独立验收 router 梯度；不能称严格无损。若只保存 X_e，此模式只需重算 GMM1 和激活。

快速模式还需明确计算成本：严格反向的 `dA=(p*D)@Wd` 不能在 p=0 时除以 p 恢复未加权结果。可另算 `D@Wd`，或将反向改为先算未加权乘积、再乘 p；前者增加 GEMM，后者改变 BF16 舍入顺序。因此是否采用该模式，须同时比较精度、计算量和省下的显存。

按每 peer 512 行、双缓冲、H6144、BF16 的简单 EP64 inbox 估算，一组输入 staging 已达 768 MiB。应比较全 peer 分槽与共享池/credits 两种方案；不能机械复制整个最坏 batch 到每个 peer，也不能忽略并行 microbatch 的窗口数量。

## 开发顺序与验收

| 阶段 | 交付 | 通过条件 |
|---|---|---|
| P0 | BF16 多 kernel 前后向、torch autograd、持久 route/context、共享专家 | 独立 oracle 对齐 y/dX/dW/dp，EP1/2 小例与空路由均通过 |
| P1 | SHMEM+CATLASS 通算流水，优先复用 dispatch export | EP8/16 下完整梯度正确，跨 microbatch 不覆盖激活，异常可退出 |
| P2 | peer 队列、64 位预算、EP64 通信和负载调度 | 实际 64 rank、全组/子组映射、极端偏斜和零 rank 通过 |
| P3 | 重算、dW 延后、确定性、图回放及模型接入 | 参数更新后多步 loss/梯度一致；确认真实框架和训练并行配置 |
| P4 | 可选 FP8 训练与更深融合 | 单独精度规范、收敛回归及受控性能收益 |

建议的代码单元为 `api/autograd`、`route_plan`、`train_context`、`shmem_transport`、`bf16_gmm`、`swiglu_backward`、`gmm_dgrad`、`gmm_wgrad`、`training_scheduler` 和 `tests`。先让各单元可独立替换、单独测量，再决定合并成几个 kernel；不要以单 kernel 数量作为完成标准。

正确性矩阵至少覆盖 B=0/1/127/128/129/255/256/257/1024/4096，以及 profiling 的分组 M=65536 主 case；还需 rank 间不同 B、空专家、同一 token 命中同 rank 的多个专家、强偏斜、p=0、连续多次 forward 后 backward、激活重算、权重更新缓存失效、多 microbatch 梯度累加。对 y、dX、每个 dW 和 dp 分别报告误差；先与框架非融合 BF16 基线比，再用 FP64 小例核验公式。阈值依据基线误差与模型训练要求锁定，不预设一个阈值覆盖所有梯度。

通信压力包含慢 rank、空 payload、槽回绕、错误 counts、容量不足、非连续 EP 组、共享专家/DP 通信的域隔离。上层未提供的 64 卡资源保持“未验证”。

性能统计分别给 forward、backward dX、backward dW、完整 forward+backward、峰值显存及实际通信字节；保留逐 rank 样本，用每轮最慢 rank 衡量训练吞吐。weight packing、路由、激活重算和梯度归约是否计入必须标明。先校验再计时，基线使用同一 route/权重/dtype，固定 warmup 和测量口径。诊断 timer 单独运行，不能将重叠阶段时间相加当作总耗时。

## 本轮验证和待补规格

`megamoe_training_reference.py` 已完成 CPU NumPy float64 验证：13 项检查通过，117 个元素做中心有限差分，最大绝对误差约 `1.0613e-11`。覆盖全部输入/参数梯度、共享专家、dp、固定 TopK sigmoid router、空专家、零 token、p=0 和源 rank token 数 `[2,0,3]` 的路由还原。该结果验证数学与索引，不代表 NPU backward 已实现。

后续补充时优先确定：64p 是否要求单 EP64 域、训练框架、EP/TP/PP/DP/CP/EFTP 组合、每 rank B 的范围与不等长性、权重/梯度 dtype、优化器与状态分片、激活重算策略、共享专家布局、路由辅助损失及校正 bias 更新、确定性与图捕获要求、验收精度和性能目标。这些字段均可覆盖初始配置，不需要改动模型规格来源。

调试占用规则：每次启动设备任务前复查两机 NPU 和相关进程，不能抢占其他任务。若本轮发现外部占用，则北京时间 2026-10-25 03:00 再检查，仍冲突时每 10 分钟重查，确认所需设备全部空闲后才启动。2026-10-24 19:37 左右的检查未见外部 NPU 任务，因此本轮继续当前环境验证，尚未触发延期调度。

配套机器可读文件：`glm52_training_spec.json`、`shape_budget.json`、`training_reference_results.json`。环境部署版本、路径和硬件实测结果见 `environment_setup_report.md`。
