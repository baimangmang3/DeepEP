# 0829 SHMEM finalize 只读诊断

本轮未初始化NPU、未重试设备阶段、未改库或系统。08:42:35/44双机快照均8卡空闲、own/foreign进程为空，证据在 `outputs/training_occupancy_20261025_0843_readonly/`。

0829全部16rank的实际加载内容身份均记录 `libshmem.so` SHA256 `9105bcc0a5022d70123291a55ac23cdf6b94e0a594ec296b709b43bb6cdbd8b3`，`libshmem_utils.so` 为 `2cb0f9b7c670be2917306d4619485695dc36ecddf90a8f64ffbe21b5694a06f9`。两机当前 `/home/zzy/deploy/megamoe-20261024/DeepEP/third-party/shmem/install/shmem/lib/` 文件与其一致。失败进程已退出，此处当前磁盘核验不替代存活maps；P1报告保留实际加载内容身份，未保存SHMEM绝对路径诊断字段。

两机SHMEM源码均clean，HEAD `4dd80e2129da42bd773982c2bb3758af1ed0bba8`，无本任务附加finalize patch。既有构建日志 `work/evidence7/logs/build.log:17,182,185`记录同revision构建安装。十份远端源码与该revision的git blob逐一SHA相等；Windows工作树CRLF不用于字节比对。远端副本与清单在 `work/shmem_finalize_0829/{rack05os7,rack05os8}/`，六份额外pinned blob在 `local-pinned-extra/`。

finalize已有同步：`src/host/init/shmem_init.cpp:1366`初始control barrier；1392 remove_heap；1394 collective status gate；1413末尾control barrier；1414 bootstrap finalize。`init/backends/shmem_init_backend.cpp:922`最终调用bootstrap barrier。`bootstrap/shmemi_bootstrap_config_store.cpp:129-160`在进程内store ref归零后DestroyStore并释放group/store。此版本有通用barrier，未发现独立的server-owner等待所有客户端完成回复读取的明确ack协议；不能据此直接定因为缺owner barrier。

rank12最早直接SHMEM错误：`store_tcp_config.cpp:306` response body Unpack返回-1；在此之前代码已取得非空response且Header.result==0。`store_net_group_engine.cpp:230,270`对应allgather序号19的wait key `SHM_(0)_S_0_19_GW`；随后 `shmem_init.cpp:1384`记录collective remove_heap status gate失败-3。旧日志未区分空body/截断/格式/长度检查分支，不能认定socket关闭或store提前销毁。

限定搜索 `/root/shmem/log` 中rank0 PID1670834及本次父PID1669834/1669833/1669725，未找到对应日志；rank12仅找到已保存6行日志。源码 `bootstrap_config_store.cpp:243`由rank0创建store server，`store_tcp_config.cpp:185-198`在当前进程构造AccStoreServer，server timer/worker为线程，并无证实的独立服务进程可另抓。stdout已保留node0/node1日志。logger默认WARN，server Shutdown为INFO（`store_tcp_config_server.cpp:115`）；没有该行不能证明未关闭。未扩大到其它PID日志。

实际库 `nm -D -C` rc0，但StoreFactory::CreateStore/DestroyStore和SmemNetGroupEngine::Create/GroupAllGather/GroupBarrier均无动态导出。现有build无tests/test/unittest目录，bin仅root_info_generate；未重编整库或绕过符号可见性。原始核查在 `work/shmem_finalize_0829/cpu-symbols-output.txt`。

rank0设备错误仍为HCCL AllGather op59/BF16 count4096/远端rank12 UB inline write CQE status3，随后异步Nonzero507057。跨机时钟未归一，不据墙钟断言其与rank12 store错误先后。根因未确认，P1完整EP16资格与后续ABBA/profile仍停止。

机器可读证据：`outputs/shmem_finalize_0829_readonly_evidence.json`。失败与释放摘要：`outputs/training_results_20261025_0829/p1-ep16-bf16/failure-and-release-summary.json`。
