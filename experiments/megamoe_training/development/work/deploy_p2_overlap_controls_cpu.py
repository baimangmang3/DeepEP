"""Approved isolated overlap control deployment; verify/plan only, no device."""
import pathlib,json,hashlib
from training_native_round import SshBackend,T,HOSTS,parallel
SRC=pathlib.Path('work/training_diagnostics/p2_ep2_overlap_hostname_control');OUT=pathlib.Path('outputs/p2_ep2_overlap_hostname_preparation')
R=T+'/validation-p2-ep2-overlap-hostname-20261026';OLD=T+'/validation-p2-ep2-serial-hostname-20261026';sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((SRC/'controller.py').read_bytes())=='53af4288397a65f7132e845a88fb3b12be82588935834045f0061d7738b4cf60'
x=SshBackend();x.connect()
try:
 def one(h):
  machine='rack05os7' if h==HOSTS[0] else 'rack05os8';d=OUT/machine;d.mkdir(exist_ok=False)
  manifest=OUT/f'{machine}-diagnostic-deployment-manifest.json';m=json.loads(manifest.read_text());mh=sha(manifest.read_bytes())
  blobs={}
  for rel,v in m['files'].items():
   if rel=='contract.json':raw=pathlib.Path('work/p2_ep2_bf16_device_contract.json').read_bytes()
   elif (SRC/pathlib.PurePosixPath(rel).name).exists():raw=(SRC/pathlib.PurePosixPath(rel).name).read_bytes()
   else:
    assert rel=='scripts/probe_wrapper_cpu.py'
    with x.clients[h].open_sftp() as ftp:
     with ftp.open(OLD+'/'+rel,'rb') as f:raw=f.read()
   assert sha(raw)==v['sha256'] and len(raw)==v['bytes'],rel;blobs[rel]=raw
  x.python(h,'CPU_overlap_setup',f'''import pathlib,json
r=pathlib.Path({R!r});assert not r.exists(),'Refuse overwrite'
for name in ('scripts','evidence'):r.joinpath(name).mkdir(parents=True,exist_ok=True)
print(json.dumps({{'created':str(r)}}))''')
  with x.clients[h].open_sftp() as ftp:
   for rel,raw in blobs.items():
    with ftp.open(R+'/'+rel,'wb') as f:f.write(raw)
   ftp.put(str(manifest),R+'/diagnostic-deployment-manifest.json')
  result=x.python(h,'CPU_overlap_verify',f'''import pathlib,json,subprocess
r=pathlib.Path({R!r})
v=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(r/'scripts/verify_deployment.py'),str(r/'diagnostic-deployment-manifest.json'),{mh!r}],capture_output=True,text=True,timeout=45);assert v.returncode==0,v.stdout+v.stderr
b=subprocess.run(['bash','-n',str(r/'scripts/run_dispatch_host_trace_checked.sh')],capture_output=True,text=True,timeout=10);assert b.returncode==0,b.stderr
args=['/root/miniconda3/envs/zzy/bin/python',{T!r}+'/candidate-p2-qualification-schema2-20261026/p2_backward_overlap/validate_training_p2.py','--training-source-root',{T!r}+'/qualified-p0-20261025/experiments/megamoe_training','--candidate-source-root',{T!r}+'/candidate-p2-qualification-schema2-20261026/p2_backward_overlap','--p1-source-root',{T!r}+'/candidate-p1-host-v2/p1_pointwise','--pointwise-library',{T!r}+'/candidate-p1/build/libmegamoe_pointwise.so','--backward-schedule','overlap','--world-size','2','--B','32','--H','128','--I','128','--E','8','--K','4','--cases','balanced,zero_gate,empty,skew','--weight-dtype','bf16','--seed','4317','--output',str(r/'evidence/plan')]
p=subprocess.run(args,capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stdout+p.stderr
print(json.dumps({{'verify':json.loads(v.stdout),'bash_n_rc':b.returncode,'plan_argv':args,'plan_rc':p.returncode,'plan':json.loads(p.stdout),'plan_stderr':p.stderr,'wrapper_CPU_not_rerun':True,'device_started':False}}))''')
  (d/'deployment-CPU-verification.json').write_text(json.dumps(result,indent=2));return result
 results=parallel(one);(OUT/'deployment-summary.json').write_text(json.dumps(results,indent=2))
 p=pathlib.Path('outputs/training_device_progress.json');state=json.loads(p.read_text(encoding='utf-8-sig'));state['p2_ep2_overlap_hostname_control']={'CPU_deployed':True,'CPU_verify_shell_plan_passed':True,'device_authorized':False,'pending_parent_final_review':True,'artifact_root':str(OUT),'remote_root':R};p.write_text(json.dumps(state,indent=2))
 print(json.dumps({'both_CPU_verify_plan_passed':True,'device_started':False,'root':R}))
finally:x.close()
