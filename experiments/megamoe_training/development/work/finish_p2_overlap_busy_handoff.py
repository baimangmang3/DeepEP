import pathlib,json,hashlib,datetime
ROOT=pathlib.Path('outputs/p2_ep2_overlap_hostname_preparation');TAG='20261026_0354_p2_overlap_once'
REPORT=pathlib.Path('outputs/training_round_'+TAG)/'p2-ep2-bf16-overlap-controller/summary.json';r=json.loads(REPORT.read_text());assert not r.get('launches') and r['status']=='blocked_cpu_own_listener_or_prior'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
hostrows={};latest=0
for host,c in r['checks'].items():
 g=c['external_launcher_gate'];latest=max(latest,g['time_ns']);items=g['controllers']
 hostrows[host]={'recorded_wall_ns':g['time_ns'],'recognized_foreign_processes':len(items),'torchrun_controllers':[v for v in items if 'torch.distributed.run' in v['argv']],'own_processes':c['own_processes'],'listeners':c['listeners'],'prior_run_evidence':c['prior_run_evidence'],'manifest_passed':c['manifest_check']['passed']}
deadline=latest+30*60*10**9
utc=lambda ns:datetime.datetime.fromtimestamp(ns/1e9,datetime.timezone.utc).isoformat()
cst=lambda ns:datetime.datetime.fromtimestamp(ns/1e9,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
summary={'classification':'Blocked by recognized active foreign launcher/benchmark before NPU query','host_evidence':hostrows,'device_started':False,'permission_consumed':False,'pending_single_control':True,'card_occupancy_at_this_gate':'Not queried; CPU guard stopped first. Do not relabel prior free16 as current.','latest_busy_observation_CST':cst(latest),'ownership_recheck_not_before_CST':cst(deadline),'ownership_recheck_not_before_UTC':utc(deadline),'no_cleanup_signal_sent':True,'no_current_own_control_or_training_job':True,'raw_controller_summary':str(REPORT),'raw_summary_sha256':sha(REPORT)}
(ROOT/'busy-guard-summary.json').write_text(json.dumps(summary,indent=2))
(ROOT/'README.md').write_text(f'''P2 EP2 overlap isolated preparation and one busy gate

Both hosts deployed independently to /home/zzy/deploy/megamoe-20261024/training/validation-p2-ep2-overlap-hostname-20261026. Exact reviewed manifest e0c8aafefab26d353884e8d24815290466bd38f2c00d180963e4f6e4b6a45a15; controller53af4288397a65f7132e845a88fb3b12be82588935834045f0061d7738b4cf60; shell6b4a92ae088d1f5bf7bbd12404828f9d0a4d568a71e0739340ab9d538f062e05. All50 source/34 dependency pins, full helper/probe bytes, shell syntax and no-Torch overlap plan passed. The missing local probe was copied byte-exact from the approved serial remote deployment and checked against its e875c057 pin. No wrapper CPU smoke/matrix was repeated.

Parent approved one separate EP2 BF16 routed overlap56-metric run. The current single controller attempt was blocked by CPU launcher census on both hosts: external EP64 torchrun plus8 benchmark processes each, case_0029 FP8/bs256, master177.22.23.1:31018. No own job, reserved listener or prior RUN. This is an occupancy guard block, not a training failure. No NPU query or stage was started, no signal sent, no successor executed. Permission remains unconsumed. Raw PID/start/PPID/argv and timestamps are in busy-guard-summary.json and linked controller summary.

Latest busy observation {cst(latest)}; earliest next ownership check {cst(deadline)} under the existing30minute heartbeat. No loop, reservation or new automation. Future execution must still pass a fresh two-host/all16/own/source/native/external-launcher/port gate; no overlap performance claim follows from preparation or serial accuracy.
''',encoding='utf-8')
p=pathlib.Path('outputs/training_device_progress.json');s=json.loads(p.read_text(encoding='utf-8-sig'))
s['p2_ep2_overlap_hostname_control'].update(device_authorized=True,pending_parent_final_review=False,pending_single_control=True,permission_consumed=False,last_busy_guard=str(ROOT/'busy-guard-summary.json'),ownership_recheck_not_before_UTC=utc(deadline),ownership_recheck_not_before_CST=cst(deadline),last_busy_run_tag=TAG)
s['active']=None;s['device_tasks_currently_running']=False;s['automatic_device_queue']=[];s['status']='p2_overlap_pending_active_foreign_launcher_30min'
s['latest_external_launcher_ownership']={'evidence':str(ROOT/'busy-guard-summary.json'),'recorded_CST':cst(latest),'recognized_foreign_processes':{h:v['recognized_foreign_processes'] for h,v in hostrows.items()},'own':0,'NPU_card_table_queried':False}
s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(s,indent=2))
for root in (ROOT,pathlib.Path('outputs/training_results_20261026_0348_p2_serial_once/p2-ep2-bf16-serial')):
 (root/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'},indent=2))
print(json.dumps({'permission_consumed':False,'foreign_processes':{h:v['recognized_foreign_processes'] for h,v in hostrows.items()},'not_before_CST':cst(deadline),'no_device_started':True}))
