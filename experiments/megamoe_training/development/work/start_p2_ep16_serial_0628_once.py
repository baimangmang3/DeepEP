"""One already-approved EP16 serial execution; uses unchanged reviewed controller."""
import pathlib,json,hashlib,subprocess,sys,datetime
TAG='20261026_0628_p2_ep16_serial_once';STAGE='p2-ep16-bf16-serial'
ROOT_REVIEW=pathlib.Path('outputs/p2_ep2_overlap_0526_parent_review.json');review=json.loads(ROOT_REVIEW.read_text())
prior=pathlib.Path(review['strict_audit']);assert hashlib.sha256(prior.read_bytes()).hexdigest()==review['strict_audit_sha256']
audit=json.loads(prior.read_text());assert audit['complete'] and audit['all_56_numerical_metrics_passed'] and audit['host_case_close_final_identity_PG_passed'] and audit['node_rc']==audit['supervisor_rc']==0 and audit['no_cleanup_signals_sent'] and audit['observed_own_processes_after']==0
assert len(audit['ranks'])==2 and all(v['initial_digest']==v['final_digest']==audit['ranks'][0]['initial_digest'] for v in audit['ranks'])
not_before=datetime.datetime.fromisoformat(review['post_run_occupancy']['next_gate_not_before_utc']);assert datetime.datetime.now(datetime.timezone.utc)>=not_before,'No ownership/device action before agreed30min deadline'
controller=pathlib.Path('work/training_diagnostics/p2_ep16_serial_hostname_control/controller.py')
manifest=pathlib.Path('outputs/p2_ep16_serial_hostname_preparation/rack05os7-diagnostic-deployment-manifest.json')
assert hashlib.sha256(controller.read_bytes()).hexdigest()==review['next_stage']['controller_sha256']=='7c0e656a196162716b0bca6fae10b0d5fa66127b6883af0d78002bfc6cf5f58a'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==review['next_stage']['manifest_sha256']=='4908ff82d5b7396d9a6174ab2ad80c33ae8033ac4410ca893ad02a2b2b6a12d7'
p=pathlib.Path('outputs/training_device_progress.json');state=json.loads(p.read_text(encoding='utf-8-sig'));assert state.get('active') is None and not state.get('device_tasks_currently_running')
key=state.setdefault('p2_ep16_serial_hostname_control',{});assert not key.get('consumed_run_tag') and not key.get('permission_consumed')
assert key.get('pending_single_control') is True
latest_deadline=datetime.datetime.fromisoformat(key['ownership_recheck_not_before_UTC']);assert datetime.datetime.now(datetime.timezone.utc)>=latest_deadline,'Latest30min guard deadline not reached'
key.update(device_authorized=True,pending_single_control=True,permission_consumed=False,parent_acceptance=str(ROOT_REVIEW),predecessor_audit=str(prior),ownership_recheck_not_before_UTC=not_before.isoformat(),approved_single_run_tag=TAG)
p.write_text(json.dumps(state,indent=2))
record=pathlib.Path('outputs/training_round_'+TAG);record.mkdir(exist_ok=False)
with (record/'controller-local.stdout').open('w') as stdout,(record/'controller-local.stderr').open('w') as stderr:
 rc=subprocess.run([sys.executable,str(controller),'--execute','--run-tag',TAG],stdout=stdout,stderr=stderr).returncode
summary=record/(STAGE+'-controller')/'summary.json'
if summary.exists():
 r=json.loads(summary.read_text());state=json.loads(p.read_text());key=state['p2_ep16_serial_hostname_control'];key['latest_controller_report']=str(summary)
 if r.get('launch_attempts') or r.get('launches'):key.update(device_authorized=False,pending_single_control=False,permission_consumed=True,consumed_run_tag=TAG,run_status=r['status'])
 else:key.update(pending_single_control=True,permission_consumed=False,run_status=r['status'])
 state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(state,indent=2))
 print(json.dumps({'controller_process_rc':rc,'status':r['status'],'launches':r.get('launches',{}),'summary':str(summary)}))
 if r['status']=='supervisors_started':subprocess.run([sys.executable,'work/observe_p2_ep16_serial.py',TAG,'early','--capture','--gate'],check=True)
raise SystemExit(rc)
