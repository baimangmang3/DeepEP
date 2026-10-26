"""Local-only per-RUN guard preservation, no occupancy recheck."""
import pathlib,json,hashlib,datetime,sys
tag=sys.argv[1];root=pathlib.Path('outputs/p2_ep2_overlap_hostname_preparation');out=root/'gates'/tag;out.mkdir(parents=True,exist_ok=False)
p=pathlib.Path('outputs/training_round_'+tag)/'p2-ep2-bf16-overlap-controller/summary.json';r=json.loads(p.read_text());assert r['status']=='blocked_cpu_own_listener_or_prior' and not r.get('launches')
hosts={}
for host,c in r['checks'].items():
 g=c['external_launcher_gate'];assert not c['own_processes'] and not c['listeners'] and not c['prior_run_evidence'];controllers=[]
 for row in g['controllers']:
  if 'torch.distributed.run' not in row['argv']:continue
  args=row['argv'];entry={k:row[k] for k in ('pid','start','ppid','argv','visible')}
  entry['selected_args']={key:args[args.index(key)+1] for key in ('--master-addr','--master-port','--config','--bs-values','--input') if key in args};controllers.append(entry)
 hosts[host]={'observed_host_time_ns':g['time_ns'],'foreign_recognized':len(g['controllers']),'torchrun':controllers,'own':0,'listeners':[],'prior':False,'read_errors':g['recognized_launcher_read_errors'],'pins_passed':c['manifest_check']['passed']}
assert any(h['foreign_recognized'] or h['read_errors'] for h in hosts.values())
deadline=p.stat().st_mtime_ns+1800*10**9
utc=lambda t:datetime.datetime.fromtimestamp(t/1e9,datetime.timezone.utc).isoformat()
cst=lambda t:datetime.datetime.fromtimestamp(t/1e9,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
summary={'tag':tag,'classification':'Known active external launcher/benchmark guard; no NPU query or device launch','hosts':hosts,'permission_consumed':False,'pending_single_control':True,'signals_sent':False,'raw_controller_summary':str(p),'raw_summary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'controller_receipt_mtime_ns':p.stat().st_mtime_ns,'not_before_UTC':utc(deadline),'not_before_CST':cst(deadline),'deadline_basis':'Local summary receipt timestamp plus30min; per-host wall clocks not calibrated for cross-host ordering','EP16_serial_condition_triggered':False}
(out/'busy-guard-summary.json').write_text(json.dumps(summary,indent=2))
(out/'README.md').write_text(f'''{tag}: one due EP2 overlap guard blocked before NPU query. Both hosts own0, no reserved listener or priorRUN; recognized external launcher details and source/native pins retained in busy-guard-summary.json. No training stage, signal, cleanup or retry. Pending single permission unconsumed; EP16 condition untriggered. Earliest next check {cst(deadline)}, based on local result receipt+30minutes. Earlier guards preserved.\n''')
progress=pathlib.Path('outputs/training_device_progress.json');s=json.loads(progress.read_text(encoding='utf-8-sig'));key=s['p2_ep2_overlap_hostname_control']
key.update(device_authorized=True,pending_single_control=True,permission_consumed=False,last_busy_run_tag=tag,last_busy_guard=str(out/'busy-guard-summary.json'),ownership_recheck_not_before_UTC=utc(deadline),ownership_recheck_not_before_CST=cst(deadline))
s['active']=None;s['device_tasks_currently_running']=False;s['automatic_device_queue']=[];s['status']='p2_overlap_pending_active_foreign_launcher_30min'
s['latest_external_launcher_ownership']={'evidence':str(out/'busy-guard-summary.json'),'recognized_foreign_processes':{h:v['foreign_recognized'] for h,v in hosts.items()},'own_at_guard':0,'NPU_table_queried':False}
s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();progress.write_text(json.dumps(s,indent=2))
print(json.dumps({'not_before_CST':cst(deadline),'hosts':{h:{'foreign_recognized':v['foreign_recognized'],'torchrun':[(p['pid'],p['start'],p['selected_args']) for p in v['torchrun']]} for h,v in hosts.items()},'device_started':False}))
