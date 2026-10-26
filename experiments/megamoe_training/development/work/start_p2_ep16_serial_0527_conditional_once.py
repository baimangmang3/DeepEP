"""Specific parent conditional successor; refuses missing/failed EP2 evidence."""
import pathlib,json,hashlib,subprocess,sys,datetime
PRIOR=pathlib.Path('outputs/training_results_20261026_0526_p2_overlap_once/p2-ep2-bf16-overlap/strict-accuracy-release-audit.json')
audit=json.loads(PRIOR.read_text())
assert audit['complete'] and audit['all_56_numerical_metrics_passed'] and audit['node_rc']==audit['supervisor_rc']==0 and audit['no_cleanup_signals_sent']
assert audit['observed_own_processes_after']==0 and audit['observed_foreign_processes_after']==0 and audit['all16_free_after']
assert audit['backward_schedule']=='overlap' and len(audit['ranks'])==2 and all(v['initial_digest']==v['final_digest']==audit['ranks'][0]['initial_digest'] for v in audit['ranks'])
TAG='20261026_0527_p2_ep16_serial_once';STAGE='p2-ep16-bf16-serial'
controller=pathlib.Path('work/training_diagnostics/p2_ep16_serial_hostname_control/controller.py')
assert hashlib.sha256(controller.read_bytes()).hexdigest()=='7c0e656a196162716b0bca6fae10b0d5fa66127b6883af0d78002bfc6cf5f58a'
p=pathlib.Path('outputs/training_device_progress.json');d=json.loads(p.read_text(encoding='utf-8-sig'))
assert d.get('active') is None
key=d.setdefault('p2_ep16_serial_hostname_control',{});assert not key.get('consumed_run_tag')
key.update(device_authorized=True,pending_parent_review=False,parent_review='outputs/p2_ep16_parent_review.json',conditional_EP2_success_audit=str(PRIOR),approved_single_run_tag=TAG)
p.write_text(json.dumps(d,indent=2))
record=pathlib.Path('outputs/training_round_'+TAG);record.mkdir(exist_ok=False)
with (record/'controller-local.stdout').open('w') as stdout,(record/'controller-local.stderr').open('w') as stderr:
 rc=subprocess.run([sys.executable,str(controller),'--execute','--run-tag',TAG],stdout=stdout,stderr=stderr).returncode
summary=record/(STAGE+'-controller')/'summary.json'
if summary.exists():
 r=json.loads(summary.read_text());d=json.loads(p.read_text());key=d['p2_ep16_serial_hostname_control'];key['latest_controller_report']=str(summary)
 if r.get('launch_attempts') or r.get('launches'):key.update(device_authorized=False,pending_single_control=False,consumed_run_tag=TAG,run_status=r['status'])
 else:key.update(pending_single_control=True,run_status=r['status'])
 d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(d,indent=2))
 print(json.dumps({'controller_process_rc':rc,'status':r['status'],'launches':r.get('launches',{}),'summary':str(summary)}))
 if r['status']=='supervisors_started':subprocess.run([sys.executable,'work/observe_p2_ep16_serial.py',TAG,'early','--capture','--gate'],check=True)
raise SystemExit(rc)
