import json,pathlib,sys,datetime
p=pathlib.Path('outputs')/('training_occupancy_'+sys.argv[1]);hosts=[]
for host in ('rack05os7','rack05os8'):
 r=json.loads((p/(host+'-summary.json')).read_text());assert r['foreign_count']==8 and r['own_device_processes']==[] and not r['existing_rank_results'] and not r['existing_test_processes']
 r['time_beijing']=datetime.datetime.strptime(r['time_utc'],'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
 hosts.append(r)
def arg(row,key):return row['parent_args'][row['parent_args'].index(key)+1]
rows='\n'.join(f"|{r['host']}|{r['time_beijing']}|{r['npu_pids'][0]}–{r['npu_pids'][-1]}|{r['parent_pid']}|" for r in hosts)
text=f'''# 本轮门禁：16卡外部任务占用

|主机|北京时间|NPU PID|父PID|
|---|---|---|---|
{rows}

两机各8/8卡foreign，own_device_processes=[]，本任务已有rank结果=[]，未发现本任务测试残留。

归属：cwd `{hosts[0]['cwd']}`；父脚本 `tests/elastic/test_ep_low_latency.py`，EP64 node6/7，master `{arg(hosts[0],'--master-addr')}:{arg(hosts[0],'--master-port')}`。case tokens{arg(hosts[0],'--num-tokens')}/H{arg(hosts[0],'--hidden')}/K{arg(hosts[0],'--num-topk')}/E{arg(hosts[0],'--num-experts')}/FP8，输出 `{arg(hosts[0],'--output-dir')}`。

本轮仅一次双机npu-smi查询，随后从已存快照只读核对/proc归属。native GEMM、ACL核数查询和训练设备阶段均未启动；没有干预、终止或发信号给外部进程。没有持续轮询，由既有10分钟heartbeat续查。

原始npu-smi、检查JSON、逐PID归属JSON及精简summary均保留此目录；此前轮次证据不覆盖。
'''
(p/'README.md').write_text(text,encoding='utf-8');(p/'summary.json').write_text(json.dumps(dict(status='foreign_occupied_deferred',hosts=hosts,native_gemm_executed=False,acl_core_query_executed=False,npu_tests_started=False,external_processes_signaled=False),indent=2));print('saved',p,'16 foreign, zero own; no device tests')
