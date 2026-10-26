用户最新指令：停止定时任务并推送私仓 megaMoe_train 分支。megamoe-16 已 PAUSED，全部待执行/条件许可撤销，历史审批不再允许自动启动。

P5 RUN20261026_0708_p5_identity64_cpu_once 的控制程序在编译器启动前缺少 prepare_snapshots.py，真实 rc1、runs=[]；最后采集的 recorder28336/start24475898 与 driver28374/start24475900 均已退出。部署清单只包含两runtime脚本，而manifest验证全部scripts，属于打包闭包不一致；原始/候选均未编译，未原样重试。两个子代理随后触发用量限制。源码导出保留这一失败，不能宣称P5编译通过。

# MegaMoE NPU 训练开发交接

## 当前恢复入口（2026-10-26最新人类指令）

07:07当前恢复入口（覆盖下方快照）：0658门禁在CPU普查见每机各1外部EP64 launcher（case0103 FP8/bs2048），own0/端口pins过；没有NPU查询或启动，EP16许可未消费，不能把launcher普查称物理free/foreign16。下次最早07:27:38.470502 CST，root已指定未来唯一tag20261026_0728_p2_ep16_serial_once，但先收齐当前CPU任务与释放，禁止并发重复门禁。

P5 identity64的7文件显式角色补丁、8CPU组已父审；原DeepEP/P0-P4未改。独立原始/候选各88源码的对象编译方案经root修正设备ELF符号并集、异常child-first与实际依赖绑定后20CPU组通过。594实际依赖与compiler/readelf已一次只读核对rc0，bindings SHA1b5047cf…e3a00f。root已在 `p5_identity64_cpu_compile_parent_review.json` 授原environment一次CPU编译RUN20261026_0708_p5_identity64_cpu_once，新目录training/p5-identity64-object-compile-20261026，manifest9dc1cb9a…062762、compile_objects4961fd90…805ec0、recorderc4039f01…295b3。先original编译/依赖/设备符号/释放过再candidate；只-c对象，不link/install/import/NPU。先查原environment及最新RUN，不能重复启动；启动或未知即消费，失败停依赖保真rc/日志/PID-start。尚未有编译结果，CPU结束后再明确0727唯一门禁责任。EP16 serial448及成功后P4组件48条件许可不变，没有其他自动设备后继。


06:44当前恢复入口（覆盖下方旧快照）：0628唯一EP16 serial门禁在CPU普查仍被两机外部launcher挡住，own0、ports/pins/prior检查过；无NPU查询/启动/信号，许可未消费。下一检查不早于06:57:21.708174 CST，由原environment作为唯一执行者，使用<=60秒短等待，不另起timer或重复门禁。root为本次门禁指定新唯一tag `20261026_0658_p2_ep16_serial_once`；精确controller7c0e656a/manifest4908ff82不变，fresh全16free/own0/无foreign launcher/端口pins过即执行448项。成功完整审计和释放后，已有P4条件单次许可继续有效；没有其他自动设备后继。

实际SHMEM只读取证已完成：`shmem_installed_header_evidence_20261026.json`保存8个头/3个既有构建记录，6个安装private头与源码镜像及既有build-info SHA逐一一致；历史0526初末FD绑定libshmem9105/kernelb075，与当前磁盘hash分开陈述。`p4_shmem_frontier_source_review.json`明确qp_quiet只等待已登记CQE，不替unsignaled RO追加SO；CQE的16bit WQE BB计数不是逻辑行seq，SIMT int4循环不证明32B原子可见性。新增frontier必须保留SO/ACK生命周期、不可变通知源和逻辑seq到实际CQ/SQ的对应。`p4_clos_mailbox_alias_scope_review.json`确认现有mailbox只在同owner/lane/epoch内给state等价类，排除local-server默认q0，ACK后不读取可能已回收CQE指针。均为源码设计证据，未修复或归因1431，未产生融合或性能资格。


06:25离线补充：原environment仍为06:27:07.111519唯一门禁执行者，未新启动设备作业。source-structured wire decoder9项及跨rank LogicalWriter join13项CPU检查已通过并经root复核，见 `p4_metadata_writer_join_parent_review.json`。用persistent weight_return_meta+localcounts、原stride signed dst、primary pairtable和gateway forward_list即可确定五条最终logical writer路径；selfprimary分区采用实际front-loaded remainder，localexpand按原source CSR基址旋转。native_domain和seq仍明确null，没有真实NPU dump。`p4_frontier_transport_role_review.json`记录Clos最多两条CQ边界/ACK前不回收、mutable frontier DMA源可能提前发布的风险、mixed subset SyncAll未证明等约束；原environment将在本轮门禁busy停止或已审设备链正常释放后，低优先只读获取实际pin SHMEM头文件。`training_hbm_capacity_snapshot_review.json`核实旧0526快照全部16卡各98304MiB(96GiB)容量；不是当前空闲或工作负载峰值，也不是BF16算力峰值。

06:07最新恢复入口（覆盖下方旧快照）：0557唯一EP16 serial门禁在CPU普查被两机各1外部torchrun+8benchmark挡住，case_0080 FP8/bs4，own0、端口空、priorfalse、pins过；无NPU查询/启动/信号，许可未消费。原environment已完成0557门禁，现root已指定其为06:27:07.111519 CST唯一到期执行者，使用<=60秒短等待，不提前查占用/不建后台timer。先查代理/RUN/device_progress，禁止另起重复门禁。原EP16 serial448 exact controller7c0e656a/manifest4908ff82批准继续，fresh全16free/own0/无外部launcher/端口pins过即单次执行。证据见 `p2_ep16_serial_hostname_preparation/gates/20261026_0557_p2_ep16_serial_once/busy-guard-summary.json`。

P4控制器26项CPU检查及一次两机实际CPU部署完成，许可均消费；manifest04d326db…ce363、controller c9f86760…7da4、CPU部署summary36bc00e0…bf7bf。真实scope为双机磁盘verify/bash-n0、27源/SO验证0及control Python两个默认recording plan0，无Torch/PTA/native/NPU；没有重复training venv运行资格。root新增 `p4_component_device_parent_review.json` 明确单次条件设备许可：原EP16 serial448全部精度/身份/四case与PG结束/两node两supervisor真实rc0/无新失败/自有释放审计通过后，新fresh全16free/own0/无外部launcher及准确pins过，原environment可一次启用RUN20261026_p4_component_v2_once，无需再等父审。只27卡0，组件48指标（42native+6hostempty）/14native调用，worker180s，固定门限无计时。失败撤销依赖，busy不消费；当前尚未创建授权文件或启动P4。原控制器与native冻结，无EP16 overlap/目标规格/性能后继。

离线融合设计已追踪Xe五条最终writer路径（`p4_xe_writer_tile_ready_review.json`）：expert-major/source-major布局仍可能让128行tile跨源、QP和本地owner，primary Xe还供展开与FM读取。`p4_tile_frontier_protocol_design.json`采用完整持久Xe、各有序writer完成frontier、独立计算就绪与读者/反向/outer epoch回收。ForwardPeer的source-ready调度改变真实提交顺序，须运行时为每tile/domain在全部expected occurrence已提交后seal最大seq，不能只靠静态Notify排序或某个大frontier。独立online模型10项新增CPU检查及此前18项规范化检查已通过并经root读源码复核（含空rank必须完成hidden/aux/descriptor外层退休），见 `p4_tile_frontier_parent_review.json`；仍未接入真实Notify或设备内存序。无SHMEM融合或性能证明。
05:52当前补充：P4单worker控制器最终26CPU检查/9Python语法检查通过，root已按实际源码完成父审，见 `p4_component_control_parent_review.json`。manifest04d326db…ce363，controller c9f86760…7da4，绑定v2 SOeea6c53d和原组件validator未改。修复记录parent保护、出生/退出采样竞态、独立cleanup200s/SSH215s、Windows远端POSIX路径、foreign PID-start与时间窗口保存；真实退出后post foreign与own释放分开，7卡不完整采样拒绝。原environment获准一次两机新control目录CPU-only部署/磁盘verify/bash-n，27两个recording actor默认plan，不查NPU/不execute/不加载native；尚未收到结果，不能重启重复部署。05:56:51.864556唯一EP16门禁优先且授权不变。P4仍无设备或性能许可。reference正做有界离线hidden/aux等待依赖和32logical comm role源码分析，不改冻结控制器/native。

05:33当前恢复入口：0526 EP2 overlap已在fresh全16free时启动并完成，严格56/56（maxRMSrel2.3461230247959236e-5）、两rank初末各35FD且跨rank digest0ff37c6c…52f031无delta、四case/PG退出完整，原始node0及supervisor rc0，无清理信号、5个已采own PID-start和2marker均gone，own0。父审 `p2_ep2_overlap_0526_parent_review.json` 接受本小规格精度/身份/生命周期资格；许可消费，不能重跑补历史worker外采。post快照27free8/28foreign8，foreign start_ticks/单调时钟/采集起止缺失，不能证明外部进入相对训练结束的时刻，也不据此宣称运行冲突或计时资格。原environment已接手唯一05:56:51.864556 CST门禁，此前不查占用；EP16 serial448原条件许可保留且前继已通过，fresh全16free/own0/无foreign launcher/29926+27774/pins过即可单次执行，无需重复父审，busy30min。无EP16 overlap/target/性能后继。P4隔离包CPU许可已消费，reference仍在写独立有界单worker控制器和CPU故障检查，尚无其部署或设备许可；见下条05:21记录。

05:21新增进展：P4 v2隔离验证包一次CPU组装/default-plan已通过并消费许可，8个runtime/source精确文件及已编译SO eea6c53d…fc797ac磁盘哈希相符、真实rc0，前后Torch/PTA未导入且SO未映射；见 `p4_component_validation_parent_review.json` 和 `p4_component_validation_v2_deployment/CPU-deployment-audit.json`。原reference正在准备独立单worker有界控制器，尚无P4设备许可；CPU预检不得重复。48个计划GU/A/Z指标含6个host-empty，尚未执行，不能算设备精度。root已按源码记录Dispatch融合角色约束于 `p4_dispatch_role_integration_review.json`：现64 AIV中后32负责gate weight，物理0地址发布/SyncAll和物理7 incoming wait不能原样映射到P4仅subblock1，需显式logical role、每tile真正完成和epoch回收协议。已有environment继续唯一05:25:04.814088门禁，未提前检查或重复作业；EP2 overlap与EP16 serial条件链不变。

05:05最新状态：P4 v1原生CPU构建成功后保留原源码/SO，父审发现明确的vector-loop uint16兼容警告；reference在独立v2只预计算uint16循环边界，数学/舍入/fence/ready/其余5源不变。v2一次CPU编译也成功，configure/build/overall/readelf/ldd均rc0，警告从14降至13，原-Wcce-compat消失，其他warning按消息/类别/重复数完全一致、无新warning。v2 manifest81d92907…328949，CPP80a4c88e…913fb4，SOeea6c53d…fc797ac；见 `p4_compute_wave_v2_parent_review.json` 与 `p4_compute_wave_v2_build/CPU-build-audit.json`。两次CPU构建许可均已消费、相应PID-start已gone，未加载native/Torch/NPU，不授设备或性能资格。原environment已停止，无新NPU/归属动作。reference继续独立GU/A/Z验证入口及首设备小合同，需绑定实际v2产物后父审。EP2 overlap仍pending未消费，下一唯一门禁不得早于05:25:04.814088；原EP16 serial条件许可不变，禁止重复已消费CPU编译及旧RUN。

04:57恢复入口：0455唯一门禁再次被两机各1外部torchrun+8benchmark workers挡住（case0052 FP8/bs128），own0、pins/端口过，未查询NPU表或启动/发送信号；EP2 overlap设备许可未消费，EP16条件未触发。下一次最早05:25:04.814088 CST。原environment正在收P4一次CPU编译结果：只部署原manifest deec8aea及6个原源码，host27 CPU build PID4035691/PGID4035691/start23659991，未加载native/Torch/NPU。本编译许可已消费，结果待收；任何错误先保原始日志/真实rc，不原样重试。reference新增P4 validator使用附加manifest，不改当前编译源。下一heartbeat先查当前代理/构建结果及progress，避免重复编译或占用检查。

04:47最新状态：0425的唯一EP2 overlap门禁在CPU进程普查仍发现两机各1个外部torchrun和8个benchmark worker；own0、pins/端口过，尚未查询NPU表，无启动/信号，许可未消费。下一次最早为04:54:50.741532 CST（本地控制器收据时间加30分钟），原environment已接手短等待，禁止另起重复检查。既有EP16 serial448条件许可保持，只有EP2 overlap完整通过并释放后再fresh门禁全过才执行。原0354与新0425证据都保留。P4计算wave源码、wrapper、分配/尾块及CPU依赖模型已父审；见 `p4_compute_wave_parent_review.json`，批准host27新独立目录一次CPU部署/原生编译，排在到期设备检查之后。还没有P4编译结果或设备许可。reference并行准备独立staged-BF16 GU/A/Z验证入口；P4仅计算wave，不含SHMEM或完整训练反向融合。最新device_progress/代理结果优先。

用户明确要求：一旦环境空闲，立刻开始验证、调试，并行进行问题定位与优化点分析。下方10月25日的 blanket error_hold、“仅排查现有证据”和空设备队列已被此指令覆盖。根因未明不妨碍执行为定位根因设计的单次受控诊断；仍禁止未分析的原样重试、重复作业和干预外部PID。

04:19当前安排：原environment已接手04:24:39.007589到期的唯一EP2 overlap检查，采用短等待响应消息，此前不重复查占用。EP16两套独立控制经父审修复world2 worker过滤及清理未知状态漏判，24项定向CPU和8项双host回归通过；随后两机exact manifest/source/native/probe bytes、bash/default noTorch plan全部rc0，无NPU初始化。见 `p2_ep16_parent_review.json`。serial manifest4908ff82…b6a12d7、overlap1df3cfb0…3c9a97c。主代理批准仅一个条件后继：若pending EP2 overlap56/初末内部FD/身份/所有关闭与真实node+supervisor rc0/无新失败及自有释放完整通过，environment在新fresh全16free/own0/无外部launcher/端口pins过后即可单次执行EP16 serial B32H128I128E64K8 routedBF16四route448指标，无需再等重复父审；前继任何报错撤销，busy保留30分钟，启动消费。EP16 overlap和目标规格无自动许可。先查现有代理、RUN和progress，禁止重复启动。P3实际AIV角色与激活ready集成差异已整理于 `p3_wave_integration_review.json`，仅算法审查。

03:56最新状态：P2 EP2 serial `20261026_0348_p2_serial_once` 已严格56/56、初末各35FD且跨rank一致无变化、node/supervisor真实rc0、关闭/PG完成，无清理信号；当前own0/free16、记录parent的两个PID-start已退出。worker外部历史账本漏采但内部FD完整，不回填或重跑补证，完整边界见本RUN `strict-accuracy-release-audit.json`。主代理已接受，并独立批准同shape/routes的overlap一次。overlap隔离部署通过，controller53af428…cf60、manifest e0c8aafe…a45a15；0354门禁在CPU进程普查发现两机各1外部torchrun和8benchmark workers，尚未查询NPU表即阻止，无NPU启动/信号，许可未消费。最早04:24:39.007589 CST复查；到时freshfree16/own0/外部launcher与pins/端口均过即可由原environment执行一次，无需重复等父审。保留0354原始阻止证据，不覆盖、不称free16或NPU表foreign16。见 `p2_ep2_overlap_parent_review.json`、`p2_ep2_overlap_hostname_preparation/README.md`；reference正离线准备EP16控制，无EP16设备许可。

03:48当前推进：P2 child-first清理与每RUN事件hostname wrapper完成源码父审及有限集成；双机实际CPU --help真实外层rc0，27单stdlib worker正常rc0并释放，未初始化NPU/PG。完整manifest为662589d416fe3258385d0167080bc9660f284fbebf7ced8f21b36c361284db51，controller为79244132decd43be482d391a6d5681f99c6c7bdb720587ee2fa2813f578996af。一次fresh检查双机free16/own0、无可识别外部launcher、50源码和34库pins及端口均通过；主代理已在 `p2_ep2_serial_parent_review.json` 批准单次EP2 BF16 routed serial四路由56指标验证并交原environment立即执行。先查最新device_progress/RUN，运行中仅跟踪，不重复启动；busy保留许可，启动消费。任何报错停止依赖，按保rc的child-first只清理本任务；无overlap或计时后继。旧1431根因与0310证据边界保持。

先读 `training_development_checkpoint.json`、`training_device_progress.json` 和最新parent review。20261026_0100_hosttrace_once已于00:48观察到free16后立即启动，并完成16rank×4路由×11指标共704项精度检查；双机及外层supervisor均rc0，26个记录的自有进程已退出，00:55:17/20确认free16、foreign0。此单次许可已消费，不得再启动同一RUN。证据见 `training_results_20261026_0100_hosttrace_once/dispatch-host-trace-shared-fp32-small/diagnostic-audit.json` 和 `training_hosttrace_0100_analysis.json`。诊断保持all_passed=false，不提供计时资格，1431根因仍未证明。

原始P1冷skew单次验证 `20261026_0140_cold_skew_once` 已完整704/704通过，双机node及supervisor均rc0，26个自有PID已释放；此许可已消费，禁止重跑。原始P1 host-v2及原始DeepEP、无新增host trace，以skew,balanced,zero_gate,empty顺序执行EP16 B32 H128 I128 E64 K8 shared FP32固定门限检查。1431未复现，但去掉记录与改变顺序涉及多个因素，不构成单变量因果证明。释放时host28短暂显示4个非ledger PID且随后/proc已消失，故那个快照不是free16证明；不回填1431或本轮训练中的冲突因果。相关完整边界见本RUN `cold-skew-control-audit.json`。

P1目标BF16 routed balanced `20261026_0153_p1_main_routed_once` 已112/112通过，各rank65536行、四专家各16384，双机node/supervisor均rc0，26个自有PID-start已释放，禁止重跑。worker内实际maps路径/内容hash及40源pins已校验，但漏采完整live FD/fstat补充快照，此缺口明确保留，不作性能资格。主代理在 `p1_main_shared_parent_review.json` 中显式修订了前继FD条件：现有精度/身份/生命周期证据足以继续下一项shared精度验证，不能宣称漏采证据已满足。shared使用有界前台observer在原日志identity_after后尽早调用原collector，不改数学/同步或已钉远端manifest。

03:11:30/34 fresh free16/own0后已启动 `20261026_0310_p1_shared_once`，原shared许可已消费。03:13:51/55在途快照证实每机own8与foreign8同时存在；仅对本次RUN/PID-start核验的6个PGID发送TERM，无KILL、不干预外部，26个自有PID-start已释放。node和supervisor真实rc文件均未生成，保留null，不能用信号号补写rc。归档后的16份报告显示balanced的176项原始指标通过、case_close完成；这是中止后的证据校正，早期采样尚停在PG_init_before。由于人工中止了退出记录链、缺真实rc和完整live FD，该轮不授予资格或性能。node0的SignalException15出现在torchrun退出事件记录的socket.getfqdn/gethostbyaddr栈，不能冒称算子首错或将冲突视作其因果证明。见本RUN `operator-abort-release-audit.json`。下一归属检查不早于03:43:55；原样重试和自动设备后继均未授权，先复核新证据及改进后的具体诊断。

P2真实CPU import预检的basename误拒已修复并验证：deep_ep._C与torch_npu._C同名但属于不同Python包，原失败与双机原始FD证据保留在 `p2_qualification_deployment/`。新schema2使用实际包根相对路径的稳定逻辑身份，完整保留两份库的exact-path/FD绑定、未知同名冲突拒绝和替换/删除检查，原adapter及数学未改。86项CPU检查通过后，新独立目录的一次双机真实CPU预检均rc0，各6计划、34映射，serial/overlap身份不同而同schedule跨host相同；前后NPU/PG均未初始化且无SHMEM owner，见 `p2_qualification_schema2_deployment/cpu-preflight-summary.json`。本CPU许可已消费，不重跑。

P2 EP2串行具体入口已完成CPU父审，见 `p2_ep2_serial_parent_review.json`：仅host27卡0/1启动，仍先检查双机全16卡；56指标、固定门限。修正了复制双机控制器后误清理host28的问题，故障注入验证首错不被清理错误遮盖；启动前锁定49源文件和已审34个实际库映射。controller a7428940…78f57、manifest ccae3a06…fa993。仍无P2设备、物理重叠或性能资格；0310 shared已完成实测证据复核：176数值通过而退出链被中止，不授予完整资格；独立P2 EP2串行精度诊断仍是有效下一步，先父审保rc清理与外部launcher门禁的具体修订，再决定新单次启动，不等待1431根因证明，overlap没有自动后继许可。

P3全层资格入口的同名库问题已完成schema2修正，74项CPU门禁通过；当前全层库按实际包namespace识别，历史68项单算子证据继续按原绝对路径/FD验证，不伪造旧报告缺失的包根。P3 adapter/native/component/数学与误差门限未改。新独立目录的一次双机CPU import/实际FD预检已完成：两机rc0、各8计划和35映射、同配置跨机identity相同，before/after NPU与PG未初始化、owner为空、原68报告bytes不变。见 `p3_schema2_cpu_deployment/cpu-summary.json` 和 `p3_namespace_parent_review.json`；CPU许可已消费，不含P3设备/性能许可。

P1性能消费门禁的离线审查还确认：现有cold/main报告只有initial身份，无final身份。合法lazy新增会改变完整digest，旧gate可能误拒，但当前32份旧报告不足以证明发生了此误拒。独立副本已补worker内初始/最终FD绑定及host收敛、初始不可变项/旧依赖不变及仅允许新增、final精确匹配current的消费检查，升级p1-training-2，51组CPU检查通过。见 `p1_final_identity_parent_review.json`；新独立目录双机CPU import预检也已一次通过，两机rc0、各8配置/34初始与repeat-final FD映射、同配置跨机一致，前后NPU/PG false且owner为空；见 `p1_schema2_cpu_deployment/cpu-summary.json`，CPU许可已消费。旧报告不回填，native/数学未改，这不是1431或0310根因修复。旧数值证据保持，性能资格仍需匹配新合同。

03:37诊断补充：每机一次5秒有界CPU getfqdn测试中，27超时、确切自有child被回收rc=-9；28约38.35ms、rc0。见 `torch_elastic_exit_dns_review/one-shot-resolution/results.json`。没有修改Torch/hosts/resolver，不能回填0310总耗时。环境子代理正在独立P2控制目录实现保留记录parent的child-first清理及外部launcher门禁；reference正在准备仅本次launcher的elastic事件hostname局部wrapper，绑定已读安装源码SHA，不改worker/rendezvous/通信。这两项均须具体源码与CPU结果父审，当前没有新NPU许可。

既有 `megamoe-16` 保留每30分钟环境忙时复查、原目标聊天和全部验收目标；观察到空闲立即推进已复核诊断。出现报错停止本轮与后继阶段，分析证据并形成下一项有区分力的诊断/修复，不能无限只查占用。1431根因、瞬态冲突与device可见性仍未证实；所有历史资格和证据边界保持。

## 历史恢复入口（2026-10-25；当前启动策略以上文为准）

先读 `training_development_checkpoint.json`、`training_device_progress.json`，并检查现有子代理和本任务进程。它们记录当前阶段；下面的10月24日至早晨记录为历史快照，不能据此重复启动或判断训练仍未执行。

实时归属以 `training_development_checkpoint.json` 的 current_readonly_occupancy/current_occupancy_evidence 为准。18:14快照显示两机16卡再次被外部任务占用，每机8个foreign PID，本任务进程为0。1431失败原因仍未证实，继续保持error_hold和空启动队列；空闲不构成重试许可，当前占用也不能回填历史报错的因果。下列带时间的归属记录保留为历史证据。

- 冻结P0已通过16卡小规格、65536行完整前后向及shared精度；CANN ai_cpu已通过小规格和目标完整前后向。正式完整step基线见 `training_full_step_baseline.json`，原始逐rank证据保留。正式CANN AIV两原生API在当前950/CANN9.2上拒绝，见 `training_cann_aiv_support.json`；初始化加载AIV对象不推翻此结论。
- P1 pointwise原生63项、EP2及EP16 BF16小规格已通过。EP16 shared FP32小规格、目标BF16及目标shared尚待完成。0950失败发生在首次HCCL初始化，早于训练库加载；1357加日志纯HCCL通过不代表根因修复。14:31的shared FP32小规格在skew路由Dispatch复用同步失败，随后出现507035；两机实际rc1、22个自有PID全部退出。现为error_hold，原P1/P3条件队列已撤销，只排查现有证据。
- P3 grouped W1+SwiGLU构件已编译，单卡68项GU/A精度通过，含65536主规格两次全元素零差；训练接口16组CPU完整梯度通过。新全层入口53项离线门禁通过并已部署两机，仍未取得P3分布式整层精度或性能资格。详见 `p3_compute_component_status.json` 与 `p3_training_deployment/README.md`。
- 原版ep_gmm_fused EP8与独立EP16适配版均已CPU编译并保存不同wheel/源码哈希；尚未取得各自NPU完整梯度资格。不得把编译或ABI静态检查当EP16运行支持。
- 16:16只读复查：两机16卡均被外部任务占用，本任务设备及控制进程为0；该快照不能回填14:36的冲突证据。1431既有SHMEM日志在所有rank均有前置初始化成功记录，但仅支持host返回链，尚未观察到device context/table字节及descriptor实际写入，见 `p1_1431_initialization_log_review/README.md`。离线host记录器通过33项CPU检查，已修复可选plan处理和报错后禁止再次调用，见 `dispatch_host_trace_cpu_review.json`；仍未集成、部署或上板。error_hold和空设备队列保持不变。

当前冻结P0环境入口为 `/home/zzy/deploy/megamoe-20261024/training/activate-qualified-p0.sh`，源码根为 `training/qualified-p0-20261025/experiments/megamoe_training`。P1使用 `training/candidate-p1-host-v2/p1_pointwise`；P3使用 `training/candidate-p3-training/{p3_training,p3_up_swiglu}`。所有资格绑定各自源码、实际库和配置，不交叉沿用。部署/编译/测试继续由GPT-6.1-sol环境子代理执行，每阶段先核实双机16卡归属；存在foreign则不干预，交现有30分钟heartbeat复查。报错立即停止后续启动，保留首错、真实rc和PID归属/时间线；排查包含进程冲突在内的原因。空闲或30分钟到期不构成自动重跑许可，须经证据处置和主代理复核。1431仅有首尾空闲快照，不能据此证明或排除中途竞争；日志采集器轮转FileNotFound与算子报错分开记录。

## 历史初始交接（以下状态与命令不代表当前进展）

记录日期：2026-10-24，北京时间。状态：首版代码和构建完成，上板验证等待空闲窗口；训练正确性和性能均未验收。

**当前凌晨入口已升级为review版本**：使用 `training/activate-review.sh` 和 `scripts/run_training_review_checked.sh`，见 `training_static_review.md` / `training_review_runbook.md`。旧资格报告不能放行新版本；本轮修复了源码/实际库身份绑定，CPU数学4/4、门禁15/15通过，但仍不构成NPU训练资格。下面的旧13文件快照和构建证据继续保留，后续以新review源码16文件版本执行。

最新追加要求：对照纳入DeepEP `experiments/ep_gmm_fused`，先核查原版EP8限制并单独适配、验证EP16；精度通过后继续融合流水优化，分析实际通信带宽、通算掩盖、MoE层MFU，必要时加低扰动打点。具体约束与分析口径见同目录 `training_performance_protocol.md`；既有源码及构建证据快照保持原样，后续新版本独立归档。

## 已完成

- 新模块位于本地 `work/DeepEP/experiments/megamoe_training`，远端部署在两机的 `/home/zzy/deploy/megamoe-20261024/training/DeepEP/experiments/megamoe_training`。
- BF16 SHMEM HT dispatch/combine，前向保存精确路由，反向复用该路由派发 dY 并回传 dx/dp。强制 HT，避免小 batch 自动选入权重语义不同的 LL 分支。
- CATLASS 的两个前向 GEMM、两个输入梯度 GEMM、两个 FP32 权重梯度 GEMM；SwiGLU 的完整导数；零路由权重仍返回正确的非零 dp；可选共享专家及 FP32 master weights。
- 首批 CATLASS 原生库已在正式 CANN 9.2.0 下编译，返回码 0。原生库 SHA256：`f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32`。这只证明构建成功。
- 独立 HCCL 数学 oracle 与全元素 Y、dx、dp、dW1、dW2 检查，附小规格共享专家检查；所有 rank 的数值、finite 和接收行数均需通过固定门禁。
- CANN 正向基线和对比脚本准备完成。主比较计时包含两边每次动态路由准备和 pack/unpack；CANN prepared-plan、算子调用区间另列。NPU event 区间不能冒称纯 kernel 时间，后者需 profiler 证据。
- CANN 完整训练基线也已编码：前向和输入梯度使用 CANN 融合通信/GMM，两组权重梯度共用同一 CATLASS FP32 累加实现；通过 `--backend cann_ai_cpu` / `cann_ccu` 接入相同完整梯度 oracle。CANN 的转置反向和 `permute_out_flag=True` 路径尚待950实机验证，不能把自定义 autograd 组合称为官方算子已注册反向。

当前候选是分阶段训练实现，通信、GEMM 和激活尚未融合，不能称为已优化的融合训练 MegaMoE。正向比较脚本与完整训练脚本分别计时，不能将正向耗时与完整前后向耗时直接比较。所有训练后端必须使用独立结果目录并单独通过完整梯度验证；正向性能比较还强制读取所有16个 rank 的候选完整梯度合格报告及哈希。

## 65536 行验收配置

EP16、BF16、源 B=8192、H=6144、I=2048、E=64、TopK=8。每 rank 四个专家，各接收 16384 行，共 65536 行；每个 token 的路由同时触及八个 rank、两台服务器。

| 阶段 | 每 rank 的逻辑 shape |
|---|---|
| 派发后的 X | `[65536,6144]` |
| 第一组权重 | `[4,4096,6144]` |
| G/U | `[65536,4096]` |
| SwiGLU 输出 | `[65536,2048]` |
| 第二组权重 | `[4,6144,2048]` |
| 专家输出 | `[65536,6144]` |

E64 用于在现有16卡上保持用户给定的每卡四专家规格，不代表完整 GLM 的 E256 或64卡验证已经完成。dtype、第二个算子输出含义等此前未由 profiling 明确的信息仍按设计文件中的 BF16/派发 X 假设执行，后续按用户补充修订。

## 占用与恢复

20:03 两机全部16卡空闲；20:12:33/34 的启动前检查发现其他 EP64 Dispatch profiling 作业已占用全部卡。本次任务未启动任何 NPU 测试，未终止对方进程。详细归属与原始证据由环境报告保存。

已建立本聊天自动续接 `megamoe-16`，首次在 **2026-10-25 03:00（Asia/Shanghai）** 检查。首次唤醒后将同一自动任务调整为每10分钟；仍有外部作业就顺延检查。机器本地时区已核实为 China Standard Time。检查已有本任务进程和结果，避免重复启动；只有两机同时空闲才开始上板。

继续由 GPT-6.1-sol 执行部署、编译和测试，主代理负责算法与修复。认证使用本聊天用户提供的凭据，不在交接文件保存密码。

恢复顺序：

1. 激活两机 `/home/zzy/deploy/megamoe-20261024/training/activate-review.sh`，使用 `scripts/run_training_review_checked.sh`；核实代码、CATLASS、SHMEM、CANN 和候选库哈希相同，并执行空闲门禁。
2. 单卡 CATLASS NN/NT/TN，BF16/FP32 输出、非整 tile 和 zero-K 用例。使用 CPU FP32 独立参考。
3. EP2 完整梯度小用例，继而 EP16 balanced、zero_gate、empty、skew；另测 FP32 master weights 和共享专家。
4. 运行上述 EP16/65536 行完整精度检查。逐 rank 保留所有元素的误差统计；失败时停止性能测试并定位原因，不放宽阈值。
5. 精度通过后测试 CANN 与DeepEP ep_gmm_fused基线，确认实际backend、版本与profiler；原版EP8和适配版EP16分开验收，再比较相同范围的正向和完整训练耗时。
6. 依据通信带宽、通算重叠、MFU和必要打点定位瓶颈，继续改进grouped调度、激活融合与SHMEM/CATLASS分块流水；每次修改先回归完整精度，不以分阶段首版benchmark作为结束条件。
7. 最终确认本任务全部设备进程已退出，再归档完整数据和可复现源码。

## AIV 对比限制

当前正式 CANN 9.2.0 在 Ascend950 上的已安装源码与实际 `libopapi.so` 均显示只支持 `ai_cpu` / `ccu`；AIV 分支在其他架构会 trap，注释说明 Host 会先拒绝。实际 SO 中的错误字符串也明确限制 A5 的 commMode。V2 头文件的 `aicpu` 注释与实际 `ai_cpu` 拼写有差异，调用以现场 schema、动态脚本与运行结果为准。

因此仍需在空闲后保存 AIV 的运行时拒绝证据与 AICPU 成功路径的 profiler。CCU 可以作为补充对照，不能标成用户要求的 AIV。若必须获得官方 AIV 数值，需要支持该实现的硬件/CANN组合；本次16张950不应虚构该项结果。

## 数值和功能边界

BF16 leaf weights 的 `.grad` 按 PyTorch 要求返回 BF16，同时保留 FP32 中间权重梯度；FP32 master weights 的 `.grad` 为 FP32。生产通信在每服务器内累加后将 partial 转为 BF16，再返回源 rank 累加；独立 oracle 使用平坦 FP32 数学求和，二者归约树与部分舍入点不同，测试保留固定误差门限。

当前 SHMEM source B 支持1至16384、H<=7168且满足BF16对齐、K<=16；空专家可验收，源 rank 零 token 尚不支持。二阶梯度不支持。测试包含共享专家，但性能主表先限定相同 routed expert 范围。所有这些边界都需在最终报告保留。

## 2026-10-25 凌晨续接状态

同一 `megamoe-16` 已调整为每10分钟，目标聊天和提示词保留。03:11曾短暂空闲，但03:13启动前复查重新被外部EP64占用；03:21仍占用。以上检查均未执行训练NPU测试。空闲快照不是预约，每次设备阶段仍须检查两机；未变版本可复用CPU回归，启动时快速核对实际源码/库身份。

新增独立CPU检查 `outputs/ep16_abi_static_check.py` 和 `outputs/ep16_abi_static_results.json`，279项静态检查通过，未修改pinned原版或review生产身份。使用真实metadata packer验证协调后的EP16布局需25232字节，expertRows/segments偏移分别144/656；派生stage模式下AIC32、N6144、N0=256最小comm_interval为12。双stage、M0=128/256时H6144的stageRows上限1024；三stage的上限分别640/512。脚本还检查队列表索引与worker覆盖；这些结果不证明编译后ABI、队列完成、物理连通或设备精度。



