"""Reviewed control-byte deployment and27 default plans only; never controller/execute."""
import pathlib,json,hashlib,datetime
from training_native_round import SshBackend,T,HOSTS,parallel
SRC=pathlib.Path('work/training_diagnostics/p4_component_v2_control');M=SRC/'manifest.json';MH='04d326db4f19b535231ce15e4360658810003b25c06251b8f2d885f2ea7ce363'
assert hashlib.sha256(M.read_bytes()).hexdigest()==MH
m=json.loads(M.read_text());blobs={}
for rel,pin in m['files'].items():
 raw=(SRC/pathlib.PurePosixPath(rel).name).read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'];blobs[rel]=raw
R=T+'/validation-p4-component-v2-control-20261026';OUT=pathlib.Path('outputs/p4_component_v2_control_deployment');OUT.mkdir(exist_ok=False)
x=SshBackend();x.connect()
try:
 def one(host):
  node=HOSTS.index(host);machine='rack05os7' if node==0 else 'rack05os8';d=OUT/machine;d.mkdir()
  x.python(host,'P4_control_CPU_setup',f'''import pathlib,json
r=pathlib.Path({R!r});assert not r.exists(),'Refuse overwrite'
for name in ('scripts','evidence'):r.joinpath(name).mkdir(parents=True,exist_ok=True)
print(json.dumps({{'created':str(r)}}))''')
  with x.clients[host].open_sftp() as ftp:
   for rel,raw in blobs.items():
    with ftp.open(R+'/'+rel,'wb') as f:f.write(raw)
   ftp.put(str(M),R+'/manifest.json')
  report=x.python(host,'P4_control_CPU_verify',f'''import pathlib,json,subprocess
r=pathlib.Path({R!r})
cmd=['/root/miniconda3/envs/zzy/bin/python',str(r/'scripts/verify_deployment.py'),str(r/'manifest.json'),{MH!r}]
if {node==0!r}:cmd+=['--worker-source']
p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stdout+p.stderr
b=subprocess.run(['bash','-n',str(r/'scripts/start_supervisor.sh')],capture_output=True,text=True,timeout=10);assert b.returncode==0,b.stderr
print(json.dumps({{'verify_argv':cmd,'verify_rc':p.returncode,'verify_stdout':p.stdout,'verify_stderr':p.stderr,'bash_n_rc':b.returncode,'remote_controller_executed':False,'credentials_or_backend_deployed':False,'device_authorization_file_created':False}}))''')
  (d/'CPU-verification.json').write_text(json.dumps(report,indent=2));return report
 results=parallel(one)
 # Each default-plan entry is invoked once, only on27. No --execute or controller.
 state=x.python(HOSTS[0],'P4_control_CPU_plans',f'''import pathlib,json,subprocess
r=pathlib.Path({R!r});rows=[]
for name in ('worker_recorder.py','supervisor.py'):
 entry=str(r/'scripts'/name)
 bootstrap="import sys,runpy,pathlib,json; sys.argv=[sys.argv[1]]; before={{'Torch':'torch' in sys.modules,'PTA':'torch_npu' in sys.modules,'P4_native':any('libmegamoe_wave_compute.so' in v for v in pathlib.Path('/proc/self/maps').read_text().splitlines())}}; assert not any(before.values()); code=0\\ntry:runpy.run_path(sys.argv[0],run_name='__main__')\\nexcept SystemExit as e:code=e.code\\nafter={{'Torch':'torch' in sys.modules,'PTA':'torch_npu' in sys.modules,'P4_native':any('libmegamoe_wave_compute.so' in v for v in pathlib.Path('/proc/self/maps').read_text().splitlines())}}; assert code in (0,None) and not any(after.values()); print(json.dumps({{'CPU_state_passed':True,'before':before,'after':after}}))"
 bootstrap=bootstrap.replace('\\\\n','\\n')
 cmd=['/root/miniconda3/envs/zzy/bin/python','-c',bootstrap,entry]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);row={{'entry':name,'argv':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}};rows.append(row)
 (r/'evidence'/(name+'.CPU-plan.json')).write_text(json.dumps(row,indent=2))
 if p.returncode!=0:print(json.dumps({{'passed':False,'rows':rows}}));raise SystemExit(1)
 values=[json.loads(line) for line in p.stdout.splitlines() if line.startswith('{{')];assert len(values)==2 and values[0]['action']=='plan only' and values[1]['CPU_state_passed']
print(json.dumps({{'passed':True,'rows':rows,'controller_not_run':True,'occupancy_not_queried':True,'device_not_started':True}}))''')
  # Prior helper preserves actual per-entry process codes/stdout/stderr in remote evidence.
 (OUT/'rack05os7/default-plans.json').write_text(json.dumps(state,indent=2))
 (OUT/'CPU-deployment-summary.json').write_text(json.dumps({'passed':True,'manifest_sha256':MH,'hosts':results,'plans':state},indent=2))
 p=pathlib.Path('outputs/training_device_progress.json');s=json.loads(p.read_text(encoding='utf-8-sig'));s['p4_component_control_CPU_deployment']={'passed':True,'CPU_permission_consumed':True,'root':R,'manifest_sha256':MH,'device_authorized':False,'controller_not_run':True,'evidence':str(OUT)};s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(s,indent=2));print(json.dumps({'both_verified':True,'host27_plans':2,'actual_rcs':[v['returncode'] for v in state['rows']],'device_started':False}))
finally:x.close()
