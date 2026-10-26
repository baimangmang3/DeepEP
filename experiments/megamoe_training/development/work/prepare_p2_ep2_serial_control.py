"""Concrete EP2 serial controls: CPU preparation only, device authorization false."""
import pathlib,json,hashlib,difflib,shlex
from training_native_round import SshBackend,T,parallel,HOSTS
CONTRACT=pathlib.Path('work/p2_ep2_bf16_device_contract.json')
CSHA='07cfad0e860ffdfd49b515774e7f8196f661ee58ecc57e39fd42ee7839ae9496'
assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest()==CSHA
SRC=pathlib.Path('work/training_diagnostics/p2_ep2_serial_control');SRC.mkdir(exist_ok=True)
OUT=pathlib.Path('outputs/p2_ep2_serial_preparation');OUT.mkdir(exist_ok=True)
OLD=pathlib.Path('work/training_diagnostics/p1_main_control')
R=T+'/validation-p2-ep2-serial-schema2-20261026';P2=T+'/candidate-p2-qualification-schema2-20261026/p2_backward_overlap';P1=T+'/candidate-p1-host-v2/p1_pointwise'
s=(OLD/'run_dispatch_host_trace_checked.sh').read_text(encoding='utf-8').replace('validation-p1-main-routed-20261026','validation-p2-ep2-serial-schema2-20261026').replace('p1-host-v2-main-bf16-balanced','p2-ep2-bf16-serial')
s=s.replace('export CANDIDATE_SOURCE_ROOT="'+P1+'"','export CANDIDATE_SOURCE_ROOT="'+P2+'"\nexport P1_SOURCE_ROOT="'+P1+'"\nexport ASCEND_RT_VISIBLE_DEVICES=0,1\nexport PYTHONPATH="'+str(pathlib.PurePosixPath(P2).parent)+':'+str(pathlib.PurePosixPath(P1).parent)+':${PYTHONPATH:-}"')
s=s.replace('[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]]','[[ "$NODE_RANK" == 0 ]]')
s=s.replace('tcp://177.22.27.1:27774','tcp://177.22.27.1:27794').replace('TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-29926}','TRAIN_MASTER_PORT=29946')
s=s.replace('--seed 4317 --oracle-chunk 1024 --max-reference-gib 12 --world-size 16','--p1-source-root "$P1_SOURCE_ROOT" --backward-schedule serial\n --seed 4317 --oracle-chunk 1024 --max-reference-gib 12 --world-size 2')
s=s.replace('--B 8192 --H 6144 --I 2048 --E 64 --K 8 --cases balanced','--B 32 --H 128 --I 128 --E 8 --K 4 --cases balanced,zero_gate,empty,skew')
s=s.replace('python -m torch.distributed.run --nnodes=2 --nproc-per-node=8','python -m torch.distributed.run --nnodes=1 --nproc-per-node=2 --max-restarts=0').replace('--node-rank="$NODE_RANK" --rdzv-backend=static --master-addr=177.22.27.1','--node-rank=0 --rdzv-backend=static --master-addr=127.0.0.1').replace('/validate_training_p1.py','/validate_training_p2.py')
s=s.replace('"RUN_TAG","TRAINING_SOURCE_ROOT"','"ASCEND_RT_VISIBLE_DEVICES","P1_SOURCE_ROOT","SHMEM_UID_SOCK_IFNAME","GLOO_SOCKET_IFNAME","RUN_TAG","TRAINING_SOURCE_ROOT"')
(SRC/'run_dispatch_host_trace_checked.sh').write_text(s,encoding='utf-8',newline='\n')
s=(OLD/'controller.py').read_text(encoding='utf-8').replace('validation-p1-main-routed-20261026','validation-p2-ep2-serial-schema2-20261026').replace('p1-host-v2-main-bf16-balanced','p2-ep2-bf16-serial').replace('outputs/p1_main_control_preparation','outputs/p2_ep2_serial_preparation').replace("d['p1_main_control']","d['p2_ep2_serial_control']")
s=s.replace(P1+'/validate_training_p1.py',P2+'/validate_training_p2.py').replace('29926','29946').replace('27774','27794')
s=s.replace('launches=parallel(launch)','launches={HOSTS[0]:launch(HOSTS[0])}').replace('states=parallel(state)','states={HOSTS[0]:state(HOSTS[0])}')
s=s.replace("'<node0-or-1>'","'0'").replace("prior=run.exists() and (any(run.glob('.node*-*')) or any(run.glob('rank*.json')) or any(run.glob('host-trace/*.jsonl')))",'prior=run.exists()')
anchor="if not a.execute:print(json.dumps(report,indent=2));return"
s=s.replace(anchor,anchor+"\n state=json.loads(pathlib.Path('outputs/training_device_progress.json').read_text(encoding='utf-8-sig'));assert state.get('p2_ep2_serial_control',{}).get('device_authorized') is True,'P2 serial device execution is not authorized'")
s=s.replace("(marker/'identity.json').write_text(json.dumps(row));print(json.dumps(row))","row.update(run_tag={a.run_tag!r},contract_sha256='"+CSHA+"',deployment_manifest_sha256={mh!r},ppid=int((proc/'stat').read_text().rsplit(')',1)[1].split()[1]),cwd=os.readlink(proc/'cwd'),uid=(proc/'status').read_text().split('Uid:',1)[1].splitlines()[0].strip())\n(marker/'identity.json').write_text(json.dumps(row));print(json.dumps(row))")
(SRC/'controller.py').write_text(s,encoding='utf-8',newline='\n')
s=(OLD/'capture_live_workers.py').read_text(encoding='utf-8').replace(P1+'/validate_training_p1.py',P2+'/validate_training_p2.py')
s=s.replace("b'RANK',b'LOCAL_RANK',b'WORLD_SIZE',b'RUN_TAG'","b'ASCEND_RT_VISIBLE_DEVICES',b'MASTER_ADDR',b'MASTER_PORT',b'TORCHELASTIC_RESTART_COUNT',b'RANK',b'LOCAL_RANK',b'WORLD_SIZE',b'RUN_TAG'")
s=s.replace("if 'RANK' not in env:continue","if 'RANK' not in env or env.get('RUN_TAG')!=run.parent.name or env.get('WORLD_SIZE')!='2' or env['RANK'] not in ('0','1'):continue")
s=s.replace("'process_start_ticks':proc_start,'stat':","'process_start_ticks':proc_start,'uid':p.stat().st_uid,'cwd':os.readlink(p/'cwd'),'stat':")
(SRC/'capture_live_workers.py').write_text(s,encoding='utf-8',newline='\n')
(SRC/'verify_deployment.py').write_bytes((OLD/'verify_deployment.py').read_bytes())
diff=[]
for n in ('run_dispatch_host_trace_checked.sh','controller.py','capture_live_workers.py'):
 diff.extend(difflib.unified_diff((OLD/n).read_text(encoding='utf-8').splitlines(keepends=True),(SRC/n).read_text(encoding='utf-8').splitlines(keepends=True),fromfile='approved-P1-controls/'+n,tofile='CPU-only-P2-serial/'+n))
(OUT/'minimal-control-diff.patch').write_text(''.join(diff),encoding='utf-8')
base=json.loads(pathlib.Path('outputs/p1_main_control_preparation/rack05os7-diagnostic-deployment-manifest.json').read_text(encoding='utf-8'))
contract=json.loads(CONTRACT.read_text(encoding='utf-8'));expected=dict(base['source_files'])
for n,h in contract['source_lock']['files'].items():expected[P2+'/'+n]={'sha256':h}
expected[P2+'/qualification_manifest.json']={'sha256':contract['source_lock']['manifest_sha256']}
x=SshBackend();x.connect()
try:
 def one(h):
  x.python(h,'cpu_p2_serial_prepare',f'''import pathlib,json
p=pathlib.Path({R!r});assert not p.exists(),'Refuse overwrite';(p/'scripts').mkdir(parents=True);print(json.dumps({{'created':str(p)}}))''')
  with x.clients[h].open_sftp() as s:
   for n in ('run_dispatch_host_trace_checked.sh','capture_live_workers.py','verify_deployment.py'):s.put(str(SRC/n),R+'/scripts/'+n)
   s.put(str(CONTRACT),R+'/contract.json')
  code=f'''import pathlib,json,hashlib,ast,subprocess,shlex
R=pathlib.Path({R!r});files={{}};sources={{}}
for n,v in {expected!r}.items():
 p=pathlib.Path(n);b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha==v['sha256'],n;sources[n]={{'sha256':sha,'bytes':len(b)}}
assert hashlib.sha256((R/'contract.json').read_bytes()).hexdigest()=={CSHA!r}
for p in sorted((R/'scripts').iterdir()):
 b=p.read_bytes();files[str(p.relative_to(R))]={{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}}
 if p.suffix=='.py':ast.parse(b.decode(),filename=str(p))
files['contract.json']={{'sha256':{CSHA!r},'bytes':(R/'contract.json').stat().st_size}}
for n,v in {base['external_native']!r}.items():assert hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==v['sha256'],n
m={{'schema':'p2-ep2-serial-controls-CPU-only-1','contract_sha256':{CSHA!r},'device_authorized':False,'files':files,'source_files':sources,'external_native':{base['external_native']!r},'execution':{contract['stages'][0]['identity_execution']!r},'host':'177.22.27.1','physical_devices':[0,1],'all16_gate_required':True,'ports':[29946,27794],'host_store_prefix':'pure-hccl-probe/session/<TORCHELASTIC_RESTART_COUNT> then UUID; unchanged f7c995d83','watchdog_seconds':600,'kill_after_seconds':30}}
p=R/'diagnostic-deployment-manifest.json';p.write_text(json.dumps(m,indent=2));sha=hashlib.sha256(p.read_bytes()).hexdigest()
r=subprocess.run(['bash','-n',str(R/'scripts/run_dispatch_host_trace_checked.sh')],capture_output=True,text=True);assert r.returncode==0,r.stderr
r=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(R/'scripts/verify_deployment.py'),str(p),sha],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
args=[{P2!r}+'/validate_training_p2.py','--world-size','2','--B','32','--H','128','--I','128','--E','8','--K','4','--backward-schedule','serial','--cases','balanced,zero_gate,empty,skew','--weight-dtype','bf16','--seed','4317','--oracle-chunk','1024','--max-reference-gib','12','--training-source-root',{T!r}+'/qualified-p0-20261025/experiments/megamoe_training','--candidate-source-root',{P2!r},'--p1-source-root',{P1!r},'--pointwise-library',{T!r}+'/candidate-p1/build/libmegamoe_pointwise.so','--output',str(R/'plan-only-output')]
r=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',*args],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stdout+r.stderr
(R/'no-torch-plan.json').write_text(r.stdout)
print(json.dumps({{'manifest_sha256':sha,'files':files,'source_count':len(sources),'plan':json.loads(r.stdout),'bash_n_rc':0,'NPU_started':False}}))'''
  result=x.python(h,'cpu_p2_serial_prepare',code);host='rack05os7' if h==HOSTS[0] else 'rack05os8'
  with x.clients[h].open_sftp() as s:
   for n in ('diagnostic-deployment-manifest.json','no-torch-plan.json'):s.get(R+'/'+n,str(OUT/(host+'-'+n)))
  return result
 rows=parallel(one);assert len({v['manifest_sha256'] for v in rows.values()})==1;(OUT/'cpu-preparation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows))
finally:x.close()
p=pathlib.Path('outputs/training_device_progress.json');d=json.loads(p.read_text(encoding='utf-8-sig'));d['p2_ep2_serial_control']={'cpu_preparation_complete':True,'pending_parent_review':True,'device_authorized':False,'NPU_started':False,'report':str(OUT),'contract_sha256':CSHA,'manifest_sha256':next(iter(rows.values()))['manifest_sha256'],'overlap_auto_successor':False};p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
