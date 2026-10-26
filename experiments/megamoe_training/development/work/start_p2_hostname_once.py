import pathlib,json,hashlib,subprocess,sys,datetime
TAG='20261026_0348_p2_serial_once'
controller=pathlib.Path('work/training_diagnostics/p2_ep2_serial_hostname_control/controller.py')
assert hashlib.sha256(controller.read_bytes()).hexdigest()=='79244132decd43be482d391a6d5681f99c6c7bdb720587ee2fa2813f578996af'
p=pathlib.Path('outputs/training_device_progress.json');d=json.loads(p.read_text(encoding='utf-8-sig'));key=d['p2_ep2_serial_hostname_control']
assert not key.get('consumed_run_tag') and d.get('active') is None
key.update(device_authorized=True,pending_parent_final_review=False,parent_review='outputs/p2_ep2_serial_parent_review.json',approved_single_run_tag=TAG)
p.write_text(json.dumps(d,indent=2))
rc=subprocess.run([sys.executable,str(controller),'--execute','--run-tag',TAG]).returncode
report_path=pathlib.Path('outputs/training_round_'+TAG)/'p2-ep2-bf16-serial-controller/summary.json'
if report_path.exists():
 r=json.loads(report_path.read_text());d=json.loads(p.read_text());key=d['p2_ep2_serial_hostname_control'];key['last_controller_report']=str(report_path)
 if r.get('launches'):
  key.update(device_authorized=False,pending_single_control=False,consumed_run_tag=TAG,run_status=r['status'])
 else:key.update(pending_single_control=True,run_status=r['status'])
 d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(d,indent=2))
print(json.dumps({'controller_process_rc':rc,'report':str(report_path)}));raise SystemExit(rc)
