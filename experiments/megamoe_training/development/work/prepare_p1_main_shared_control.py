"""CPU-only next-stage contract; never launches an NPU job."""
import pathlib,json,hashlib,difflib,shlex
from training_native_round import SshBackend,T,parallel,HOSTS
OLD=pathlib.Path('work/training_diagnostics/p1_main_control');SRC=OLD.with_name('p1_main_shared_control');SRC.mkdir(exist_ok=True)
OUT=pathlib.Path('outputs/p1_main_shared_preparation');OUT.mkdir(exist_ok=True)
R=T+'/validation-p1-main-shared-20261026'
base=json.loads(pathlib.Path('outputs/p1_cold_skew_preparation/rack05os7-diagnostic-deployment-manifest.json').read_text(encoding='utf-8'))
for name in ('run_dispatch_host_trace_checked.sh','controller.py','capture_live_workers.py','verify_deployment.py'):
 s=(OLD/name).read_text(encoding='utf-8').replace('validation-p1-main-routed-20261026','validation-p1-main-shared-20261026').replace('p1-host-v2-main-bf16-balanced','p1-host-v2-main-shared-bf16-balanced').replace('outputs/p1_main_control_preparation','outputs/p1_main_shared_preparation').replace("d['p1_main_control']","d['p1_main_shared_control']")
 if name.endswith('.sh'):
  s=s.replace('--B 32 --H 128 --I 128 --E 64 --K 8 --cases skew,balanced,zero_gate,empty','--B 8192 --H 6144 --I 2048 --E 64 --K 8 --cases balanced').replace('--weight-dtype bf16 --output','--weight-dtype bf16 --shared --output')
 (SRC/name).write_text(s,encoding='utf-8',newline='\n')
diff=[]
for n in ('run_dispatch_host_trace_checked.sh','controller.py'):
 diff.extend(difflib.unified_diff((OLD/n).read_text(encoding='utf-8').splitlines(keepends=True),(SRC/n).read_text(encoding='utf-8').splitlines(keepends=True),fromfile='approved-cold-skew/'+n,tofile='pending-main/'+n))
(OUT/'minimal-entry-diff.patch').write_text(''.join(diff),encoding='utf-8')
x=SshBackend();x.connect()
try:
 def one(h):
  x.python(h,'cpu_main_preparation',f'''import pathlib,json
p=pathlib.Path({R!r});assert not p.exists(),'Do not overwrite';(p/'scripts').mkdir(parents=True);print(json.dumps({{'created':str(p)}}))''')
  with x.clients[h].open_sftp() as s:
   for n in ('run_dispatch_host_trace_checked.sh','capture_live_workers.py','verify_deployment.py'):s.put(str(SRC/n),R+'/scripts/'+n)
  code=f'''import pathlib,json,hashlib,subprocess,ast,shlex
R=pathlib.Path({R!r});m={base!r}
for n,v in m['source_files'].items():assert hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==v['sha256'],n
for n,v in m['external_native'].items():assert hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==v['sha256'],n
for n in m['files']:
 p=R/n;b=p.read_bytes();m['files'][n]={{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}}
 if n.endswith('.py'):ast.parse(b.decode(),filename=n)
m.update(schema='original-p1-main-shared-control-1',purpose='P1 target shared BF16 full-gradient accuracy only, pending explicit parent approval',shape={{'B':8192,'H':6144,'I':2048,'E':64,'K':8}},cases=['balanced'],shared=True,weight_dtype='torch.bfloat16',expected_recv_rows=65536,expected_rows_per_expert=[16384]*4,NPU_executed=False)
p=R/'diagnostic-deployment-manifest.json';p.write_text(json.dumps(m,indent=2));sha=hashlib.sha256(p.read_bytes()).hexdigest()
q=subprocess.run(['bash','-n',str(R/'scripts/run_dispatch_host_trace_checked.sh')],capture_output=True,text=True);assert q.returncode==0,q.stderr
v=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(R/'scripts/verify_deployment.py'),str(p),sha],capture_output=True,text=True);assert v.returncode==0,v.stdout+v.stderr
T={T!r};C=T+'/candidate-p1-host-v2/p1_pointwise'
args=[C+'/validate_training_p1.py','--training-source-root',T+'/qualified-p0-20261025/experiments/megamoe_training','--candidate-source-root',C,'--pointwise-library',T+'/candidate-p1/build/libmegamoe_pointwise.so','--world-size','16','--B','8192','--H','6144','--I','2048','--E','64','--K','8','--cases','balanced','--weight-dtype','bf16','--shared','--seed','4317','--oracle-chunk','1024','--max-reference-gib','12','--output',str(R/'plan-output')]
q=subprocess.run(['bash','-lc','source '+T+'/activate-qualified-p0.sh\\n'+'timeout 30 python '+shlex.join(args)],capture_output=True,text=True,timeout=45);assert q.returncode==0,q.stdout+q.stderr
(R/'no-torch-plan.json').write_text(q.stdout)
print(json.dumps({{'manifest_sha256':sha,'plan':json.loads(q.stdout),'source_files_verified':len(m['source_files']),'bash_n_rc':0,'Torch_imported':False,'NPU_started':False}}))'''
  r=x.python(h,'cpu_main_preparation',code);host='rack05os7' if h==HOSTS[0] else 'rack05os8'
  with x.clients[h].open_sftp() as s:
   for n in ('diagnostic-deployment-manifest.json','no-torch-plan.json'):s.get(R+'/'+n,str(OUT/(host+'-'+n)))
  return r
 rows=parallel(one);(OUT/'cpu-deployment.json').write_text(json.dumps(rows,indent=2));assert len({r['manifest_sha256'] for r in rows.values()})==1
 print(json.dumps(rows))
finally:x.close()
p=pathlib.Path('outputs/training_device_progress.json');d=json.loads(p.read_text(encoding='utf-8-sig'));d['p1_main_shared_control']={'cpu_preparation_complete':True,'pending_final_review':True,'NPU_started':False,'report':str(OUT),'manifest_sha256':next(iter(rows.values()))['manifest_sha256'],'no_successor_device_authorization':True};p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
