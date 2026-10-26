import pathlib,shutil,json,hashlib,py_compile,subprocess,sys
src=pathlib.Path('work/training_candidates/p1_pointwise');dst=pathlib.Path('work/training_candidates/p1_host_v2/p1_pointwise');dst.mkdir(parents=True,exist_ok=True);m=json.loads((src/'training_manifest.json').read_text());files=m['sources_sha256'].copy();v=pathlib.Path('work/p1_case_lifecycle_review/validate_training_p1_host_candidate.py');assert hashlib.sha256(v.read_bytes()).hexdigest()=='841f663ee336bb3c342c83de9bc4d19bf5b41cc685329bef141dcf5fbc0ac93c'
for n in files:
 p=dst/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v.read_bytes() if n=='validate_training_p1.py' else (src/n).read_bytes())
files['validate_training_p1.py']=hashlib.sha256(v.read_bytes()).hexdigest();m['sources_sha256']=files;m['status']='host-v2 explicit host cleanup; new qualification required';m['remote_root']='/home/zzy/deploy/megamoe-20261024/training/candidate-p1-host-v2/p1_pointwise';(dst/'training_manifest.json').write_text(json.dumps(m,indent=2))
for n in files:
 assert hashlib.sha256((dst/n).read_bytes()).hexdigest()==files[n]
 if n.endswith('.py'):py_compile.compile(str(dst/n),doraise=True)
text=pathlib.Path('work/run_p1_training_round.py').read_text();text=text.replace("Path('work/training_candidates/p1_pointwise')","Path('work/training_candidates/p1_host_v2/p1_pointwise')").replace('4020cc4f0a7ee3145be315168f07bd0a27c6172cfd6e95c27a0f766e2814bd54','841f663ee336bb3c342c83de9bc4d19bf5b41cc685329bef141dcf5fbc0ac93c').replace("CP=T+'/candidate-p1-training/p1_pointwise'","CP=T+'/candidate-p1-host-v2/p1_pointwise'").replace('run_training_p1_checked.sh','run_training_p1_host_v2_checked.sh').replace('outputs/p1_training_deployment','outputs/p1_host_v2_deployment')
text=text.replace("if p.exists() and any(p.iterdir()) and not (p/'training_manifest.json').exists():", "if p.exists() and any(p.iterdir()) and not ((p/'training_manifest.json').exists() or (p/'cpu-staging.json').exists()):")
text=text.replace("if q.exists():assert hashlib.sha256(q.read_bytes()).hexdigest()==sha,'Modified existing candidate: '+name", """if q.exists():
  got=hashlib.sha256(q.read_bytes()).hexdigest()
  if name=='validate_training_p1.py' and got=='4020cc4f0a7ee3145be315168f07bd0a27c6172cfd6e95c27a0f766e2814bd54':
   backup=p.parent/'staged-validator-before-host-v2.py'
   if not backup.exists():backup.write_bytes(q.read_bytes())
  else:assert got==sha,'Modified existing candidate: '+name""")
pathlib.Path('work/run_p1_host_v2_round.py').write_text(text)
launch=pathlib.Path('work/run_training_p1_checked.sh').read_text().replace('candidate-p1-training/p1_pointwise','candidate-p1-host-v2/p1_pointwise').replace('LIMIT_SECONDS:-900','LIMIT_SECONDS:-600').replace('mkdir -p "$RUN/logs"','''mkdir -p "$RUN/logs"
case "$STAGE" in
 *-main*) export SHMEM_LOG_LEVEL=INFO SHMEM_LOG_PATH="$RUN/shmem-info" ;;
 *) export SHMEM_LOG_LEVEL=DEBUG SHMEM_LOG_PATH="$RUN/shmem-debug" ;;
esac
mkdir -p "$SHMEM_LOG_PATH"''').replace('npu-smi info >','timeout 30 npu-smi info >')
launch=launch.replace('set +e\ntimeout','''python -c 'import os,json,sys; print(json.dumps({"argv":sys.argv[1:],"environment":{k:os.getenv(k) for k in ("RUN_TAG","TRAINING_SOURCE_ROOT","CANDIDATE_SOURCE_ROOT","POINTWISE_SO","SHMEM_LOG_LEVEL","SHMEM_LOG_PATH","HCCL_SOCKET_IFNAME","DEEPEP_SHMEM_ENDPOINT","NNODES","NPROC_PER_NODE","LIMIT_SECONDS")}},indent=2))' "$SCRIPT" --output "$RUN" "$@" > "$RUN/logs/launch-node$NODE_RANK.json"
set +e
timeout''')
pathlib.Path('work/run_training_p1_host_v2_checked.sh').write_text(launch,newline='\n')
for name in ('work/run_p1_host_v2_round.py',):py_compile.compile(name,doraise=True)
p=pathlib.Path('work/collect_training_stage.py');p.write_text(p.read_text().replace("('logs','shmem-debug')","('logs','shmem-debug','shmem-info')"))
p=pathlib.Path('work/training_occupancy_gate.py');p.write_text(p.read_text().replace("'run_training_lifecycle_checked.sh'","'run_training_lifecycle_checked.sh','run_training_p1_host_v2_checked.sh'"))
plans=[]
for world,e,k,shared,dtype,b,h,i in [(2,8,4,False,'bf16',32,128,128),(2,8,4,True,'fp32',32,128,128),(16,64,8,False,'bf16',32,128,128),(16,64,8,True,'fp32',32,128,128),(16,64,8,False,'bf16',8192,6144,2048),(16,64,8,True,'bf16',8192,6144,2048)]:
 args=['--world-size',str(world),'--E',str(e),'--K',str(k),'--B',str(b),'--H',str(h),'--I',str(i),'--cases','balanced' if b==8192 else 'balanced,zero_gate,empty,skew','--weight-dtype',dtype,'--pointwise-library','/placeholder/libmegamoe_pointwise.so','--output','plan-only']+(['--shared'] if shared else [])
 r=subprocess.run([sys.executable,str(dst/'validate_training_p1.py'),*args],capture_output=True,text=True,check=True,timeout=15);plans.append(json.loads(r.stdout))
out=pathlib.Path('outputs/p1_host_v2_deployment');out.mkdir(exist_ok=True);(out/'cpu-plans.json').write_text(json.dumps(plans,indent=2));(out/'manifest.json').write_text(json.dumps(m,indent=2));print('8 source hashes verified; six CPU plans and py_compile passed')
