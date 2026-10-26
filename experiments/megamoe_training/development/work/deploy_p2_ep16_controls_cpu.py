"""Two parent-reviewed isolated EP16 control CPU deployments, never device/gate."""
import pathlib,json,hashlib
from training_native_round import SshBackend,T,HOSTS,parallel
review=json.loads(pathlib.Path('outputs/p2_ep16_parent_review.json').read_text())
assert review['device_authorized'] is False
sha=lambda b:hashlib.sha256(b).hexdigest();x=SshBackend();x.connect()
try:
 summaries={}
 for mode in ('serial','overlap'):
  item=review[mode];SRC=pathlib.Path(item['local']);OUT=pathlib.Path('outputs/p2_ep16_'+mode+'_hostname_preparation');R=item['remote']
  assert sha((SRC/'controller.py').read_bytes())==item['controller_sha256']
  def one(host):
   machine='rack05os7' if host==HOSTS[0] else 'rack05os8';d=OUT/machine;d.mkdir(exist_ok=False)
   manifest=OUT/f'{machine}-diagnostic-deployment-manifest.json';assert sha(manifest.read_bytes())==item['manifest_sha256'];m=json.loads(manifest.read_text());blobs={}
   for rel,pin in m['files'].items():
    candidate=SRC/pathlib.PurePosixPath(rel).name
    if candidate.exists():raw=candidate.read_bytes()
    else:
     assert rel=='scripts/probe_wrapper_cpu.py'
     with x.clients[host].open_sftp() as ftp:
      with ftp.open(T+'/validation-p2-ep2-serial-hostname-20261026/'+rel,'rb') as f:raw=f.read()
    assert sha(raw)==pin['sha256'] and len(raw)==pin['bytes'],rel;blobs[rel]=raw
   x.python(host,'CPU_ep16_'+mode+'_setup',f'''import pathlib,json
r=pathlib.Path({R!r});assert not r.exists(),'Refuse overwrite'
for part in ('scripts','evidence'):r.joinpath(part).mkdir(parents=True,exist_ok=True)
print(json.dumps({{'created':str(r)}}))''')
   with x.clients[host].open_sftp() as ftp:
    for rel,raw in blobs.items():
     with ftp.open(R+'/'+rel,'wb') as f:f.write(raw)
    ftp.put(str(manifest),R+'/diagnostic-deployment-manifest.json')
   result=x.python(host,'CPU_ep16_'+mode+'_verify',f'''import subprocess,pathlib,json
r=pathlib.Path({R!r})
v=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(r/'scripts/verify_deployment.py'),str(r/'diagnostic-deployment-manifest.json'),{item['manifest_sha256']!r}],capture_output=True,text=True,timeout=45);assert v.returncode==0,v.stdout+v.stderr
b=subprocess.run(['bash','-n',str(r/'scripts/run_dispatch_host_trace_checked.sh')],capture_output=True,text=True,timeout=10);assert b.returncode==0,b.stderr
args=['/root/miniconda3/envs/zzy/bin/python',{T!r}+'/candidate-p2-qualification-schema2-20261026/p2_backward_overlap/validate_training_p2.py','--training-source-root',{T!r}+'/qualified-p0-20261025/experiments/megamoe_training','--candidate-source-root',{T!r}+'/candidate-p2-qualification-schema2-20261026/p2_backward_overlap','--p1-source-root',{T!r}+'/candidate-p1-host-v2/p1_pointwise','--pointwise-library',{T!r}+'/candidate-p1/build/libmegamoe_pointwise.so','--backward-schedule',{mode!r},'--world-size','16','--B','32','--H','128','--I','128','--E','64','--K','8','--cases','balanced,zero_gate,empty,skew','--weight-dtype','bf16','--seed','4317','--output',str(r/'evidence/plan')]
p=subprocess.run(args,capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stdout+p.stderr
print(json.dumps({{'verification':json.loads(v.stdout),'bash_n_rc':b.returncode,'plan_argv':args,'plan_rc':p.returncode,'plan':json.loads(p.stdout),'plan_stderr':p.stderr,'wrapper_import_not_repeated':True,'NPU_query_or_initialization':False,'device_started':False}}))''')
   (d/'CPU-deployment-verification.json').write_text(json.dumps(result,indent=2));return {'passed':True,'manifest_sha256':item['manifest_sha256'],'bash_n_rc':result['bash_n_rc'],'plan_rc':result['plan_rc'],'device_started':False}
  summaries[mode]=parallel(one);(OUT/'CPU-deployment-summary.json').write_text(json.dumps(summaries[mode],indent=2))
  (OUT/'README.md').write_text(f'''EP16 {mode} isolated CPU control deployment

Both hosts match parent-reviewed manifest {item['manifest_sha256']}; all manifest-pinned helper/contract/probe/source/native bytes verified, bash-n and explicit world16/E64K8/B32 routed BF16 four-case no-Torch plan pass. Original serial wrapper probe copied raw from approved EP2 deployment and pin matched. No wrapper import/smoke or prior CPU matrix repeated. No NPU query, tensor, PG/session or device stage. CPU-only preparation; EP16 device_authorized remains false.

Remote root {R}. Controller {item['controller_sha256']}. Shared collector6690c393 fixesworld16/local0..7 binding; lifecycle4f465926 retained. Old controls untouched. EP2 overlap due04:24:39 remains the sole pending authorized device stage.
''')
  (OUT/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p.read_bytes()) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'},indent=2))
 p=pathlib.Path('outputs/training_device_progress.json');s=json.loads(p.read_text(encoding='utf-8-sig'));s['p2_ep16_controls_CPU_deployment']={'serial':summaries['serial'],'overlap':summaries['overlap'],'device_authorized':False,'parent_review':'outputs/p2_ep16_parent_review.json'};p.write_text(json.dumps(s,indent=2));print(json.dumps(summaries))
finally:x.close()
