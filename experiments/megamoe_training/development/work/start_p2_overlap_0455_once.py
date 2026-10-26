import pathlib,json,hashlib,subprocess,sys,datetime
TAG='20261026_0455_p2_overlap_once';STAGE='p2-ep2-bf16-overlap'
controller=pathlib.Path('work/training_diagnostics/p2_ep2_overlap_hostname_control/controller.py')
assert hashlib.sha256(controller.read_bytes()).hexdigest()=='53af4288397a65f7132e845a88fb3b12be82588935834045f0061d7738b4cf60'
p=pathlib.Path('outputs/training_device_progress.json');d=json.loads(p.read_text(encoding='utf-8-sig'));key=d['p2_ep2_overlap_hostname_control']
assert not key.get('consumed_run_tag') and d.get('active') is None
key.update(device_authorized=True,pending_parent_final_review=False,parent_review='outputs/p2_ep2_overlap_parent_review.json',approved_single_run_tag=TAG)
p.write_text(json.dumps(d,indent=2))
record=pathlib.Path('outputs/training_round_'+TAG);record.mkdir(exist_ok=False)
with (record/'controller-local.stdout').open('w') as stdout,(record/'controller-local.stderr').open('w') as stderr:
 rc=subprocess.run([sys.executable,str(controller),'--execute','--run-tag',TAG],stdout=stdout,stderr=stderr).returncode
report_path=record/(STAGE+'-controller')/'summary.json'
if report_path.exists():
 r=json.loads(report_path.read_text());d=json.loads(p.read_text());key=d['p2_ep2_overlap_hostname_control'];key['last_controller_report']=str(report_path)
 if r.get('launches'):key.update(device_authorized=False,pending_single_control=False,consumed_run_tag=TAG,run_status=r['status'])
 else:key.update(pending_single_control=True,run_status=r['status'])
 d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(d,indent=2))
 print(json.dumps({'controller_process_rc':rc,'status':r['status'],'launches':r.get('launches',{}),'report':str(report_path)}))
 if r['status']=='supervisors_started':
  monitor=pathlib.Path('work/collect_p2_hostname_state.py').read_text().replace('20261026_0348_p2_serial_once',TAG).replace('p2-ep2-bf16-serial',STAGE).replace('validation-p2-ep2-serial-hostname-20261026','validation-p2-ep2-overlap-hostname-20261026')
  path=pathlib.Path('work/collect_p2_overlap_state.py');path.write_text(monitor)
  subprocess.run([sys.executable,str(path),'early','--capture','--gate'],check=True)
raise SystemExit(rc)
