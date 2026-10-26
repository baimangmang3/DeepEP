"""One local fault injection into actual controller main; no SSH/signals/device."""
import pathlib,json,importlib.util,sys,tempfile,os,contextlib,io
P=pathlib.Path('work/training_diagnostics/p2_ep2_serial_child_first_control/controller.py').resolve()
spec=importlib.util.spec_from_file_location('p2_control_cleanup_test',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
calls={'verify':[],'gates':[],'launch':[],'cleanup':[],'closed':False}
class FakeBackend:
 def connect(self):pass
 def close(self):calls['closed']=True
 def python(self,host,tag,code):
  if 'subprocess.Popen' in code:
   calls['launch'].append(host);raise RuntimeError('ORIGINAL launch failure injected')
  calls['verify'].append(host);return {'passed':True,'own_processes':[],'listeners':[],'prior_run_evidence':False,'manifest_check':{'passed':True}}
 def gate(self,host,tag):
  calls['gates'].append(host);return {'observation':{'npu_smi_rc':0,'all_eight_free':True,'processes':[]},'owners':{'processes':[]}}
def failing_stop(backend,host,tag):
 calls['cleanup'].append(host);raise OSError('CLEANUP evidence failure injected')
m.SshBackend=FakeBackend;m.stop_owned=failing_stop
cwd=pathlib.Path.cwd();argv=sys.argv[:]
try:
 with tempfile.TemporaryDirectory(prefix='p2-control-CPU-') as temp:
  try:
   os.chdir(temp);root=pathlib.Path(temp);m.OUT=root/'manifest';m.OUT.mkdir();(m.OUT/'rack05os7-diagnostic-deployment-manifest.json').write_text('{}')
   p=root/'outputs/training_device_progress.json';p.parent.mkdir();p.write_text(json.dumps({'p2_ep2_serial_child_first_control':{'device_authorized':True}}))
   sys.argv=[str(P),'--execute','--run-tag','CPU_fault_injection_only']
   with contextlib.redirect_stdout(io.StringIO()):m.main()
   summary_path=root/'outputs/training_round_CPU_fault_injection_only/p2-ep2-bf16-serial-controller/summary.json'
   summary=json.loads(summary_path.read_text())
   assert calls['cleanup']==[m.HOSTS[0]] and calls['launch']==[m.HOSTS[0]]
   assert set(calls['gates'])==set(m.HOSTS) and set(calls['verify'])==set(m.HOSTS)
   assert summary['status']=='controller_error' and 'ORIGINAL launch failure injected' in summary['error'] and summary['first_error']==summary['error']
   assert 'CLEANUP evidence failure injected' in summary['cleanup_errors'][m.HOSTS[0]] and m.HOSTS[1] not in summary['cleanup_errors']
   assert calls['closed'] and summary_path.exists()
   result={'passed':True,'fault_case':'actual controller launch raises; stop_owned also raises','checks':['only_host27_launch_and_cleanup','all16_checks_remain_two_hosts','original_first_error_preserved','cleanup_error_separate','finally_summary_saved','backend_closed'],'actual_summary':summary,'calls':calls,'SSH_used':False,'signals_used':False,'NPU_used':False}
  finally:os.chdir(cwd)
finally:os.chdir(cwd);sys.argv=argv
out=pathlib.Path('outputs/p2_ep2_serial_child_first_preparation/cleanup-fault-injection-CPU.json');out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:result[k] for k in ('passed','checks','SSH_used','signals_used','NPU_used')}))
