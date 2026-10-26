"""Preserve a new busy CPU guard separately; no remote/occupancy action."""
import pathlib,json,hashlib,datetime
TAG='20261026_0425_p2_overlap_once';ROOT=pathlib.Path('outputs/p2_ep2_overlap_hostname_preparation');OUT=ROOT/'gates'/TAG;OUT.mkdir(parents=True,exist_ok=False)
P=pathlib.Path('outputs/training_round_'+TAG)/'p2-ep2-bf16-overlap-controller/summary.json';r=json.loads(P.read_text())
assert r['status']=='blocked_cpu_own_listener_or_prior' and not r.get('launches')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
hosts={}
for host,v in r['checks'].items():
 g=v['external_launcher_gate'];assert not v['own_processes'] and not v['listeners'] and not v['prior_run_evidence'] and g['busy_or_uncertain'] and len(g['controllers'])==9
 hosts[host]={'foreign_recognized_processes':9,'torchrun':[p for p in g['controllers'] if 'torch.distributed.run' in p['argv']],'recorded_host_time_ns':g['time_ns'],'own':0,'reserved_listeners':[],'prior_RUN':False,'pins_passed':v['manifest_check']['passed']}
deadline=P.stat().st_mtime_ns+1800*10**9
utc=lambda ns:datetime.datetime.fromtimestamp(ns/1e9,datetime.timezone.utc).isoformat()
cst=lambda ns:datetime.datetime.fromtimestamp(ns/1e9,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
summary={'run_tag':TAG,'classification':'Active known external EP64 launcher/benchmark blocked CPU guard before any NPU query','device_started':False,'permission_consumed':False,'pending_single_control':True,'hosts':hosts,'raw_summary':str(P),'raw_summary_sha256':sha(P),'controller_summary_local_receipt_mtime_ns':P.stat().st_mtime_ns,'recheck_deadline_basis':'Local receipt/write timestamp of current controller summary plus30minutes; host wall times retained separately without calibrated cross-host ordering','ownership_recheck_not_before_UTC':utc(deadline),'ownership_recheck_not_before_CST':cst(deadline),'NPU_table_queried':False,'signals_sent':False,'EP16_conditional_successor_triggered':False}
(OUT/'busy-guard-summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'README.md').write_text(f'''EP2 overlap new single guard {TAG}

Both hosts identified1 external torchrun+8 benchmark workers, EP64 case_0042 FP8/bs1, master177.22.23.1:31024.27 controller3989274/start23438157;28 controller4126524/start23440402. Own0, reserved ports empty, no existing RUN, exact pins pass. CPU guard blocked before npu-smi; current card occupancy is not asserted. No device started, no signal or retry. Existing single overlap permission remains unconsumed; EP16 serial conditional permission has not triggered.

Earliest next check {cst(deadline)}, conservatively30min after the local saved controller result receipt; per-host times are separate uncalibrated evidence.0354 summary and earlier busy-guard-summary were not overwritten. This new raw summary and hash are linked from busy-guard-summary.json. No new timer/job/automation created.
''')
p=pathlib.Path('outputs/training_device_progress.json');s=json.loads(p.read_text(encoding='utf-8-sig'))
s['p2_ep2_overlap_hostname_control'].update(device_authorized=True,pending_single_control=True,permission_consumed=False,last_busy_run_tag=TAG,last_busy_guard=str(OUT/'busy-guard-summary.json'),ownership_recheck_not_before_UTC=utc(deadline),ownership_recheck_not_before_CST=cst(deadline))
s['status']='p2_overlap_pending_active_foreign_launcher_30min';s['active']=None;s['device_tasks_currently_running']=False;s['automatic_device_queue']=[]
s['latest_external_launcher_ownership']={'evidence':str(OUT/'busy-guard-summary.json'),'recognized_foreign_processes':{h:9 for h in hosts},'own':0,'NPU_table_queried':False}
s['p2_ep16_serial_conditional_authorization']={'parent_review':'outputs/p2_ep16_parent_review.json','conditional_on':'EP2 overlap exact56/internal FD/caseclose/PG/true rc0/no abort or collision/own released, followed by fresh all16/own/foreignlauncher/ports/pins','triggered':False,'device_authorized_now':False,'performance_authorized':False,'no_EP16_overlap_or_target_successor':True}
s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(s,indent=2))
(ROOT/'README.md').write_text((ROOT/'README.md').read_text()+f'\nNew guard {TAG} also blocked by known external launchers; see gates/{TAG}/README.md. New not-before {cst(deadline)}. Previous0354 artifacts preserved.\n')
(OUT/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'},indent=2))
print(json.dumps({'foreign_recognized_each_host':9,'own':0,'device_started':False,'permission_consumed':False,'not_before_CST':cst(deadline),'evidence':str(OUT/'busy-guard-summary.json')}))
