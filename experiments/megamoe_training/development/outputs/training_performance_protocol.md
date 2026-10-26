# 16卡训练融合、对照与瓶颈分析口径

执行规则更新于2026-10-26，按本聊天最新要求执行。首次上板窗口为2026-10-25 03:00（Asia/Shanghai），环境占用则每30分钟复查；观察到空闲且没有本任务作业时，立即执行已复核的单次验证或有区分力的诊断。报错先保留首错、真实退出码、进程归属和身份，停止依赖阶段，不原样循环重试；未知根因不阻止获取缺失证据的诊断。部署、编译、测试继续由GPT-6.1-sol承担。本文为验收与分析协议，不包含尚未取得的硬件测量结果；当前状态以checkpoint、device_progress和最新原始证据为准。

## 执行顺序与对照身份

1. 单卡CATLASS、EP2小规格、EP16完整前后向，再验证BF16 B8192/H6144/I2048/E64/TopK8，每rank4专家、总接收65536行。所有Y、dx、dp、FP32 dW及leaf梯度、接收counts通过固定门禁；共享专家另测。该E64配置保留给定的每rank4专家规格，与完整GLM的E256分开记录。
2. 各基线分别通过同一独立oracle，再进入性能采集。完整梯度未合格时不计时、不凭输出Y合格跳过反向。
3. 精度通过后继续开发融合流水和性能优化；当前分阶段SHMEM/CATLASS实现仅作为可回归控制组，不是最终完成条件。
4. 最终表格同时包含：分阶段SHMEM/CATLASS控制组、融合训练候选、DeepEP `ep_gmm_fused`、CANN ops-transformer的两融合算子加SwiGLU链路。正向、完整前后向、算子输入已准备好的分项分别报告。

当前本地DeepEP固定提交为`35fa7d91b321340c153b098bf166202645976a75`。用户指定的`experiments/ep_gmm_fused`在该版本中有Python `routing.py`、Host `tiling_validation.hpp`等EP<=8限制，内核还有rank维度模板、元数据ABI与调度数组。原版先作为EP8支持范围对照，EP16适配版单独标识提交与修改；不能只放宽参数检查就称为支持16卡。EP8结果不能与EP16直接计算加速比。

EP16适配须逐层检查rank/expert数组、结构体ABI、UB/L1/工作区容量、核数与任务分配、peer/QP/WQE布局、跨机UDMA初始化及完成、ready发布和buffer回收。训练封装还须确认输入导出、权重转置、所有六个GEMM与dp返回的支持范围。优先复用已有timeline/profiling工具。保留原版和新适配版，分别通过精度门禁。

CANN列必须绑定实际ops-transformer源码/安装包版本、libopapi与kernel哈希以及comm_mode。当前950的AIV限制按既有证据和原生API运行结果记录；CCU单列。DeepEP内部AIV/UDMA实现与官方CANN的AIV模式是不同对照身份。CANN训练基线目前是自定义autograd组合，两个wgrad共用CATLASS，报告中明确写出，避免把差异归因于未改变的wgrad实现。

## 公平计时与证据

- 相同X、权重、TopK路由、FP32 gate、BF16舍入位置、专家行数、shared、重计算和权重梯度累加范围。主规格的alltoallvGmm包含X_e导出；不拿关闭导出的推理路径替代训练路径。
- 动态路由端到端包含双方每次counts/路由准备、pack/unpack、gating与TopK归并。固定路由已准备好的路径单列，同条件比较；缺少图捕获支持时单列eager结果，不能混用graph和eager计算加速比。
- 同shape、同拓扑、同进程组、同缓存/频率/温度策略；保存热身、每rank每轮原始样本、逐轮最慢rank、P50/P95和波动。各方案交替或ABBA运行，失败/超时/异常负载不得进入性能主表。
- 正式速度使用关闭打点的构建。另用profiler/抽样打点归因，记录其相对无打点版本的扰动和数值一致性。
- 保存step、rank、host、source/expert/chunk、buffer epoch、shape/counts、dtype、编译选项、依赖及二进制哈希。跨host日志先核实时钟偏移和误差；优先用相同调用和因果标识关联，不直接比较未同步的原始时间戳。

## 通信带宽

必须分三种分母/分子口径，不能把逻辑tensor字节直接称为网卡物理带宽：

| 指标 | 字节来源 | 时间与解释 |
|---|---|---|
| 逻辑通信吞吐 | 路由展开后的有效tensor尺寸 | 对应通信阶段或端到端时间；属于算法有效吞吐 |
| 实际payload带宽 | 对真正提交的peer/WQE、去重/聚合、重发及分块传输做字节记账 | 对应方向的传输窗口；分别标明payload、元数据和padding |
| 链路带宽 | 可隔离到本任务的链路/NIC/交换计数器增量 | 同一采样窗口；扣除可测背景，注明协议开销和计数器含义 |

区分机内/机间、发送/接收、单向/双向、单rank/单链路/全域。双向流量相加必须标注为双向aggregate；源和目的不能重复累计同一次传输。按接收端最终专家展开的65536行计量，不代表发送端必然传输了同样多的行：SHMEM可能按rank或server去重，combine还可能先归并server partial。

本规格一份BF16 `[65536,6144]` 逻辑tensor为805306368字节，即768MiB。它只用于核对逻辑量，不预设四次通信的实际线速字节。实际记录dispatch X、return Z、dispatch dY、return dX以及gate/dp/控制元数据，特别核对每次去重、跨server归并与导出是否额外访问GM。构造相同路由、分块、消息大小的comm-only对照测可达带宽，再判断实际通信是否饱和。

## 通算掩盖与关键路径

使用四组受控实验：compute-only、comm-only、相同分块的强制串行、融合重叠。isolated组必须保留与融合组一致的有效工作量和数据布局；如果差异无法消除，只能给出条件性估计。

记独立通信时间为C、计算时间为G、融合时间为F，均在同一rank/同一调用范围内。报告：

- 回收时间 `Delta = C + G - F`，同时保存实测串行时间S及`S-F`，核对额外启动/同步开销。
- 归一化回收比例 `Delta / min(C,G)`，是受控实验的重叠收益估计；保留原始值。负值或超过1时先检查计时、缓存、频率、工作量和算法变化，不裁剪成好看的比例。
- 另报告时间线上通信在途区间与AIC有效计算区间的交集/并集。通信在途或等待区间不等于链路正在发包，不能仅凭交集就宣称物理传输被掩盖。
- 报告不能被计算掩盖的尾部等待、流水填充/排空、最慢peer、rank间到达偏差、AIC等ready、AIV等buffer、quiet/completion和导出阻塞。

不能直接将多个重叠stage的时长相加推导总时间，也不能把一个wait归因于网络：它可能由对端GEMM、入队延迟或buffer回收触发。主结论基于最慢rank关键路径和对照实验。

## MFU与计算效率

先核实当前Ascend950DT实际芯片/逻辑设备映射、AIC数、频率与BF16 dense峰值；不使用未经核实的市场宣传值，也不把FP8/稀疏峰值作为BF16分母。若峰值暂不可确认，先报告有效TFLOP/s和相对同shape纯GEMM实测吞吐的效率，MFU留空。

主规格每rank六个GEMM，以一次乘加为2 FLOPs：

| 范围 | 有效FLOPs/rank |
|---|---:|
| 前向两GEMM | 4947802324992 |
| 反向四GEMM | 9895604649984 |
| 完整前后向 | 14843406974976 |

使用实测每专家M累加，不能把不均衡case套成平均M。对16个rank的完整同步步骤：

`有效TFLOP/s = sum_r(F_useful,r) / (T_critical_seconds * 1e12)`

`MoE层GEMM口径MFU = sum_r(F_useful,r) / (T_layer_seconds * sum_r(P_dense_BF16,r))`

这里明确排除router、激活标量运算、shared、optimizer及其他模型层，不能称作GLM5.2整模型MFU；若纳入shared或激活则同步更新分子和计时范围。有效FLOPs不含padding、冗余计算和checkpoint重计算，额外报告实际执行FLOPs及HFU/重计算开销，防止通过多算来抬高MFU。BF16输入、FP32累加/输出的wgrad仍按其实际BF16 Cube路径核算峰值，而非简单按输出dtype改成FP32峰值。

再以同shape、同dtype的纯CATLASS/CANN GEMM吞吐作计算上限实测参考，分别分析端到端效率下降来自通信、启动空洞、tile利用率、K/M尾块、load/store、L0/L1等待或反向wgrad调度。

## 打点与优化闭环

先使用现有profiler和ep_gmm_fused timeline解析器；信息不足时新增可关闭、低开销、限量的marker。覆盖pack/export、远端post、quiet/completion、ready发布/等待、GMM tile begin/end、gating/reduce、buffer回收及epoch。对forward和backward分别打点，关联step/rank/peer/chunk/expert/tile；核实AIC/AIV/host的时钟来源和换算关系，不默认共享时钟。

打点仅观察，不移除/弱化barrier、完成等待、ready、epoch与回收协议；不得为计时在内层加入改变原有重叠的全局同步。统计marker丢失/溢出，证明开关打点的精度一致并测扰动。用没有打点的构建复测最终收益。

按证据选择优化：通信提交不足查消息聚合和peer轮转；AIC等数据查分块和预取；AIV等buffer查流水深度和回收；GMM效率低查tile/K循环和grouped调度；中间张量GM往返占优再融合SwiGLU/导数；wgrad尾部查tile所有权、分K与FP32归约成本。每次只改变可解释的因素，保留对照、回归全梯度和主规格；无稳定收益或数值不合格的候选不进入最终版本。

最终交付包括同scope性能表、每rank数据、通信字节账本、重叠关键路径图、MFU分母出处、瓶颈证据、有效优化及无效尝试、源码/二进制/资格报告哈希。全部上板完成并确认本任务资源释放后归档。
