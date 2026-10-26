# One-shot native编排入口

已准备 `work/training_native_round.py`，本轮仅做本地CPU语法/模拟检查，未连接主机、查询设备或初始化NPU。6项模拟回归全部通过，涵盖哈希必须先于所有门禁、foreign任一主机占用、own残留、已有阶段防重复、哈希失败止步及非法RUN_TAG。控制器不导入torch；模拟测试也不导入paramiko。

03:41下一轮使用已有认证机制，将两机密码仅放进控制进程的 `DEPLOY_PASSWORD_27` / `DEPLOY_PASSWORD_28` 环境变量。脚本及文档不保存密码。入口（Windows当前workspace）：

```powershell
& 'C:\Users\z30064827\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  work/training_native_round.py --execute --run-tag 20261025_0341 `
  --output outputs/training_round_20261025_0341
```

顺序固定在同一控制进程中：并行建立并保留两条SSH连接 → 并行CPU核对冻结16源码/48身份文件及3个部署helper哈希（即便foreign占用也只读文件）→ 并行两机16卡状态/逐PID归属与本任务残留/结果检查 → 任一非空或不确定立即保存证据退出；两机均空闲则立即复用27的同一SSH连接启动native supervisor。门禁之后没有模型往返、新SSH握手、SFTP或CPU回归；记录 gate_to_launch_call_ms 供核验。

控制器当前固定要求主规格shared新版CPU身份 `04c7371f9d9c78623e54ad823c224dddffbb6f9e63b35c665f3f379e10a0fba6`（两机实际CPU采集一致；旧a211身份仅历史保留），任何CPU失败都不会触发设备门禁/启动。没有参数默认执行行为；必须明确 `--execute`。

远端native helper再次本地8卡门禁；通过后才执行实际ACL核数查询，再运行原16case native GEMM。supervisor总体timeout360秒，TERM后30秒KILL；内部核数查询90秒、native180秒各有自己的timeout。仅控制本次timeout进程组，无foreign信号/复位。native输出 `runs/<RUN_TAG>/native-gemm/{rc,core-query.log,native.log,results.json,npu-before.txt,npu-after.txt}`。

native-gemm目录的启动标记只用于同RUN_TAG防重复与审计，不占卡，不代表设备预约。控制器不常驻、不轮询、没有占卡进程；日志状态 `native_supervisor_started` 仅表示监督进程创建，实际设备执行须从local gate和结果日志确认。

native通过并确认释放后，继续既有review版EP2→shared/FP32→EP16小规格→65536行完整梯度及对照/性能流程；每阶段仍执行双机门禁，阈值不改，失败不计时。此控制器不把一次native启动当作完整任务完成。

当前native16/16和旧身份EP2两套均已通过；以后同native/SO不要调用native入口重跑。统一从training_device_progress.json的未通过检查点续测，使用work/training_stage_round.py（相同内存环境认证，双机门禁后同连接即时启动）。新版待EP16；具体版本及foreign检查点见training_shared_main_deployment/README.md。旧EP2报告只作历史证据，不能替代新身份的完整资格。
