# ep_gmm_fused 对照静态预检

检查对象：DeepEP `35fa7d91b321340c153b098bf166202645976a75`；以下路径相对该仓库，`E=` 表示 `experiments/ep_gmm_fused/`。本次只读源码，没有放宽限制、编译新版本或上板；既有训练13文件快照与证据包保持原封。

## 原版边界

- 原版只声明 EP1–8、BF16 contiguous、权重 `[local_experts,K,N]`；前向本地专家≤64、逆向≤32；不是autograd封装，也不含token顺序恢复与TopK加权：`E/README.md:103–105,159–168`。
- “64卡”说明是16个独立EP4域及外层EFTP，不能读成单域EP64，更不能据此认定EP16可用：`E/README.md:147–155`。
- EP8门禁至少同时存在：Python runtime `E/python/deep_ep_experimental/ep_gmm_fused/runtime.py:131–132`；Host Attach `E/csrc/all_to_allv_grouped_mat_mul_udma/ascend950/all_to_allv_grouped_mat_mul_udma.cpp:175–180`；adapter `E/csrc/all_to_allv_grouped_mat_mul_udma/ascend950/udma_runtime.cpp:19–25`；公共UDMA工厂 `csrc/runtime/session.cpp:237–245`；tiling `E/csrc/tiling_validation.hpp:130–131`。只改一个Python检查不能形成EP16实现。
- 原版EP8必须保留独立wheel/环境/哈希作为对照；待开发的EP16适配版须另外命名、记录补丁和SHA，先完整精度再性能，不能把两组EP8冒称一个EP16域。
- 当前已准备固定CatCCOS/CATLASS/SHMEM依赖，但没有将原版EP8编译安装进当前训练venv；“依赖已齐”不等于原版对照已构建或设备验收。新增对照应隔离保留现有训练产物。

## EP16必须逐项审计

| 项目 | 准确源码入口 | 必须核查的变化/风险 |
|---|---|---|
| 编译期rank上限 | `E/csrc/all_to_allv_grouped_mat_mul_udma/ascend950/all_to_allv_grouped_mat_mul_udma_device.hpp:80` 的 `MoeConstraints<1,8,64>`；逆向 `E/csrc/grouped_mat_mul_all_to_allv_udma/ascend950/grouped_mat_mul_all_to_allv_udma_device.hpp:55` 的 `<1,8,32>` | TP/EP/rank的乘积和每专家容量均应独立核查；不能机械替换所有8。 |
| rank数组和前缀 | `E/csrc/catccos_overlay/include/catccos/dgemm/block/block_scheduler_alltoallv_allgather.hpp:35–43,53–54`；`.../comm/block/comm_block_scheduler_alltoallv_allgather.hpp:43,69–74`；`.../comm/block/comm_block_scheduler_gmm_alltoallv_target_order.hpp:75–77,108–113` | `[expert][rank]` token/offset、rank+1前缀、输入/输出group数组、迭代上界和索引类型都要随EP16核查。`...`均相对`E/csrc/catccos_overlay/include/catccos`。 |
| 路由metadata ABI | `E/python/deep_ep_experimental/ep_gmm_fused/ops.py:33–38,118–146`；`E/csrc/catccos_overlay/include/catccos/dgemm/moe_route_metadata.hpp:18–20,25–43` | Python byte offsets与C++结构体的8个peerRows、64专家、最多512 segments相互绑定；扩容会改变后续offset/sizeof，旧Python/旧SO不可混用；保留原始route-table fallback语义。 |
| UB和WQE scratch | `E/csrc/tiling_validation.hpp:17–21,130–158,182–205`；两方向device.hpp各`:39–42`；`E/csrc/catccos_overlay/include/catccos/comm/tile/tile_remote_copy_udma_2d.hpp:40–41,131,153–155,186–191` | 使用编译预算248KiB而非物理256KiB；WQE scratch256B和双UB拷贝、rank数组增长不能相撞；需重新计算边界与首个非法值，不靠padding掩盖。 |
| workspace及并发 | `E/csrc/all_to_allv_grouped_mat_mul_udma/ascend950/udma_runtime.hpp:19–24`；`E/csrc/tiling_validation.hpp:17,157–158,189–198`；`E/csrc/catccos_overlay/include/catccos/dgemm/kernel/ascend950_alltoallv_grouped_matmul.hpp:236–240,287–293` | 固定400MiB+4KiB借用空间；stages×rank×stageRows×K/N布局、comm_interval整除和每rank核数条件在16卡重新计算；保持2/3stage、buffer复用与同步区不重叠。 |
| queue所有权与完成性 | `E/csrc/catccos_overlay/include/catccos/comm/block/comm_block_remote_copy.hpp:325–343`；`.../comm/tile/tile_remote_copy_udma_2d.hpp:84,171,186–191` | deferred quiet跨peer前先Finalize，UDMA peer-scoped quiet必须由正确队列owner执行；EP16核到peer映射不得造成多人共享或无人drain。 |
| ready/回收协议 | `E/csrc/catccos_overlay/include/catccos/dgemm/kernel/ascend950_alltoallv_grouped_matmul.hpp:261–262,313,357,392–396,668`；`.../dgemm/kernel/grouped_matmul_alltoallv_tla.hpp:179–180,225,266,311–323,334,371–382,523,561–566` | AIC写完→AIV发出→peer完成→本地MTE完成→cross-core ready→workspace复用的顺序必须保留；空专家/空peer也须参加对应barrier，不能删ready/quiet来换性能。 |
| 跨机连接/销毁 | `E/python/deep_ep_experimental/ep_gmm_fused/runtime.py:106–116,157–180,196–205`；`csrc/runtime/session.cpp:240–276,283–297`；`E/README.md:121,138–155` | group真实成员/全局root→UID广播→UDMA初始化；复核16 peers的通道、UID ABI、连接容量、网络可达、单进程owner、同线程销毁与失败清理。已有HT EP16通过不等于此UDMA工厂/队列配置通过。 |

上述是适配审计入口，不是已完成的EP16修改；还需对所有显式常量、结构体大小、静态断言与测试catalog做完整diff审查。

## 现有采集与计时能力

- `E/tests/profiling/timeline.py:31–46` 探测实际profiler/help；`:49–88` 支持`op_summary`、`PipeTimeline`、`InstrTimeline`分次采集，内核过滤、skip warmup+一次精度调用及`--kill=off`；不支持的工具选项明确报错。
- 同文件`:92–137` 要求非空trace/binary和目标kernel事件，记录实际观测tracks/pipes与文件SHA；明确不假定完整core/指令覆盖。`E/tests/README.md:221–234,255–265` 记录逐rank采集/命令/help与两方向独立profile入口。
- `E/tests/profiling/msprof_parser.py:52–98` 解析导出`op_summary_*.csv`并检查设备数；`E/tests/performance/metrics.py:15–32` 先按样本序号对齐再取rank MAX，随后计算分位数。
- 前向kernel已有条件编译timer挂点：`E/csrc/catccos_overlay/include/catccos/dgemm/kernel/ascend950_alltoallv_grouped_matmul.hpp:25–26,253–270,315–354,438–439,621–622`。它们的存在不证明当前wheel已启用/导出完整打点，也不等于具有peer queue-ready/quiet的完整时间线；要先核对编译开关、buffer与事件覆盖，再按需要做单独诊断构建。

带宽分母需与实际peer payload及相同时间scope绑定，通算掩盖要有通信/计算时间线或严格同输入对照，MFU要绑定dtype/有效FLOPs/硬件峰值口径。现有duration分位数工具本身不自动提供三者结论；沿用主任务新增指标规范，不能凭总耗时直接宣称已隐藏通信。

## 凌晨执行约束

继续按03:00及每10分钟占用门禁。先保留原版EP8资格与性能对照；跨机可先验收2台各4卡组成的单EP8域，不能把两台各8卡直接送进当前EP8实现。EP16适配另立版本，先完成export/非export、zero/empty/skew、重复与大→小→大、queued与清理精度，再采相同scope的性能与profile。全部与当前训练P0/正式CANN ai_cpu/ccu明确分列；无autograd原版不能作为已完成完整训练后端。
