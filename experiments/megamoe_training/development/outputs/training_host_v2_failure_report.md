# host-v2 资格阶段3失败与释放

新validator SHA `841f663ee336bb3c342c83de9bc4d19bf5b41cc685329bef141dcf5fbc0ac93c`，旧P1/P0与pointwise SO未变。新资格前两阶段通过：0901 EP2 BF16四case、0902 EP2 shared FP32四case；全部指标、counts/shapes/dtypes、host case_cleanup、PG/host退出均通过且rc0/own退出，审计各在`outputs/training_results_<tag>/p1-stage-accuracy-audit.json`。

0904 EP16 BF16小规格在首个report-directory `collective_checked` 的int32 MIN allreduce失败。rank10(PID1798842/device2)报E39999，`RunAicpuIndOpThreadInit`执行返回0x2a；随后HCCL watchdog终止并SIGABRT。尚未完成identity/configuration，未创建P1引擎、未执行任何训练case，不能当成P1数学错误或0829 finalize复现。本次没有device日志确认threadArray=null，不能从旧故障沿用该结论。

rank10、同阶段peer rank8(PID1798840)、rank0(PID1722730)的精确plog/device日志已回收，`targeted-pid-<pid>-logs-final/manifest.json`有SHA，晚次副本与首份字节数相同。首因主要见rack05os8/logs/node1.log:70开始；watchdog后果见rack05os7/logs/node0.log:75开始。完整argv/env、双机source/library快hash见本次controller与各host/logs/launch-node*.json。失败worker退出前未保存maps，本次不能把预检的正式磁盘SHA冒充其实际mapped身份；P1正式identity尚未采集。

两node最终真实rc均1。所有own worker/torchrun/supervisor PID已退出；27仅本次已确认PGID1722184残留CPUtorchrun收到TERM，无KILL/reset。09:10:14/22 fresh双机快照显示各8卡foreign、own为空；并非当前16卡空闲，外部进程未被干预。PID归属证据在`outputs/training_occupancy_20261025_0910_host_v2_release/`。

只读资源快照：两机384CPU，load约14–15，MemAvailable约2.2TB，/dev/shm空闲。28的隔离ep_gmm构建nice10/parallel2，仍可能有未记录的历史压力，当前快照不能证明其与故障完全无关。`resource-kernel-readonly.json`保留资源/ulimit/cgroup信息，ulimit属于当前control进程而非已消失worker。完整内核窗口09:02–09:07:30共27机3281行/28机3388行；精确error/fail/oom/timeout/exception/reset/cqe筛选未截断，27无匹配，28仅5条09:02:01–02 UDMA -110/URMA TP -28，在本轮启动门禁前。内核PID1756720不属于本次run，已捕获历史own账本无匹配，历史归属仍未知，不能直接判foreign或ENOSPC因果。

全部NPU后续STOP，包含余下资格及ABBA/profile。保持新host-v2源与旧资格不变，下一设备动作必须由主代理根据证据重新授权。仅剩获准CPU隔离构建/回归可继续。

机器可读摘要：`outputs/training_results_20261025_0904/p1-host-v2-ep16-bf16-small/failure-release-audit.json`。
