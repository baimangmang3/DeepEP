"""One reviewed P3 CPU-only deployment/import preflight; no NPU/ownership call."""
import pathlib,json,hashlib,shlex
from training_native_round import SshBackend,T,parallel,HOSTS
OUT=pathlib.Path('outputs/p3_schema2_cpu_deployment');OUT.mkdir(exist_ok=True)
ROOT=pathlib.Path('work/training_candidates');TRAIN=ROOT/'p3_training';COMP=ROOT/'p3_up_swiglu'
R=T+'/candidate-p3-training-schema2-20261026'
assert hashlib.sha256((TRAIN/'manifest.json').read_bytes()).hexdigest()=='b7ef80a85abcff922c31b37c74f77dec73d3c7efd2eec68a5287119d193b5456'
assert hashlib.sha256(pathlib.Path('work/p3_qualification_cpu_probe.py').read_bytes()).hexdigest()=='77b7a461a649cabab80216e255b8e9db916fe54d376c2e104a24a3b8d0e1288f'
expected={}
for name,root,key in [('p3_training',TRAIN,'files_sha256'),('p3_up_swiglu',COMP,'source_sha256')]:
 m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
 for rel,h in m[key].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
 expected.update({name+'/'+rel:h for rel,h in m[key].items()})
 expected[name+'/manifest.json']=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
prep=json.loads(pathlib.Path('work/p3_qualification_cpu_probe_preparation.json').read_text(encoding='utf-8'))
for path,sha in prep['artifact_byte_contract'].items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==sha,path
x=SshBackend();x.connect()
try:
 def one(h):
  machine='rack05os7' if h==HOSTS[0] else 'rack05os8';OUT.joinpath(machine).mkdir(exist_ok=True)
  # Time-sensitive, exact previously observed foreign-parent IDs only; no process scan.
  seed=3893540 if h==HOSTS[0] else 4029253
  prior=json.loads(pathlib.Path('outputs/training_results_20261026_0310_p1_shared_once/p1-host-v2-main-shared-bf16-balanced/status2-occupancy.json').read_text(encoding='utf-8'))[h]
  wanted=next(v['parent_args'] for v in prior['owners']['foreign_device_processes'] if v['ppid']==seed)
  chain=x.python(h,'cpu_known_external_parent',f'''import pathlib,json,os,time
pid={seed};rows=[]
for depth in range(4):
 if pid<=1:break
 p=pathlib.Path('/proc')/str(pid)
 try:
  stat=(p/'stat').read_text();fields=stat.rsplit(')',1)[1].split();args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
  rows.append({{'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19]),'state':fields[0],'argv':args,'cwd':os.readlink(p/'cwd'),'seed_argv_matches_prior':args=={wanted!r} if depth==0 else None}});pid=int(fields[1])
 except OSError as e:rows.append({{'pid':pid,'missing_or_unreadable':repr(e)}});break
print(json.dumps({{'captured_time_ns':time.time_ns(),'clock_boottime_ns':time.clock_gettime_ns(time.CLOCK_BOOTTIME),'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'clk_tck':os.sysconf('SC_CLK_TCK'),'seed_from_status2':{seed},'chain':rows,'scope':'exact recorded foreign PPID ancestors only; no NPU inquiry or claim about past NPU use'}}))''')
  evidence=pathlib.Path('outputs/training_results_20261026_0310_p1_shared_once/p1-host-v2-main-shared-bf16-balanced')/machine/'known-external-parent-chain.json';evidence.write_text(json.dumps(chain,indent=2),encoding='utf-8')
  x.python(h,'cpu_p3_schema2_setup',f'''import pathlib,json
r=pathlib.Path({R!r});assert not r.exists(),'Refuse overwrite'
for path in ('p3_training/tests','p3_up_swiglu/native','scripts','evidence/original68'):r.joinpath(path).mkdir(parents=True,exist_ok=True)
print(json.dumps({{'created':str(r)}}))''')
  with x.clients[h].open_sftp() as s:
   for rel in expected:s.put(str(ROOT/rel),R+'/'+rel)
   s.put('work/p3_qualification_cpu_probe.py',R+'/scripts/probe_cpu.py')
   old=list(prep['artifact_byte_contract'])
   s.put(old[0],R+'/evidence/original68/up-swiglu-results.json');s.put(old[1],R+'/evidence/original68/unit-audit.json')
  files=x.python(h,'cpu_p3_schema2_hash',f'''import pathlib,json,hashlib,ast
r=pathlib.Path({R!r});items={{}}
for name,sha in {expected!r}.items():
 p=r/name;b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,name
 if p.suffix=='.py':ast.parse(b.decode(),filename=name)
 items[name]={{'sha256':sha,'bytes':len(b)}}
for name,sha in {{'evidence/original68/up-swiglu-results.json':'ef1f347b66f5593a6590d6743546d679b477cae7c7ee55c22618313611e96144','evidence/original68/unit-audit.json':'0b4d4440d9de144f51271a90c901ffd90b5d695fefba3cd7dcf9bf1d99feee06'}}.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha,name
lib=pathlib.Path({T!r}+'/candidate-p3-up-swiglu-20261025/build/libmegamoe_up_swiglu.so')
assert hashlib.sha256(lib.read_bytes()).hexdigest()=='4623d9abf673210f0e3e07473f9f02c541256d699e47ce0a2a2dd7229c8c856b'
print(json.dumps({{'passed':True,'sources':items,'original68_bytes_verified':True,'up_swiglu_SO_sha256':'4623d9abf673210f0e3e07473f9f02c541256d699e47ce0a2a2dd7229c8c856b','Torch_imported':False}}))''')
  args=['--run-cpu']
  for key,value in prep['proposed_explicit_paths'].items():args+=['--'+key.replace('_','-'),value]
  cmd=f'source {T}/activate-qualified-p0.sh\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nnice -n 10 timeout 120 python '+shlex.join([R+'/scripts/probe_cpu.py',*args])+f' > {R}/evidence/probe.stdout 2> {R}/evidence/probe.stderr\n'
  _,o,e=x.clients[h].exec_command('bash -lc '+shlex.quote(cmd),timeout=140);o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
  d=OUT/machine;(d/'source-verification.json').write_text(json.dumps(files,indent=2),encoding='utf-8');(d/'command-result.json').write_text(json.dumps({'rc':rc,'command':cmd,'stderr':err},indent=2),encoding='utf-8')
  with x.clients[h].open_sftp() as s:
   for src,dst in [('evidence/cpu-import-identity.json','cpu-import-identity.json'),('evidence/probe.stdout','probe.stdout'),('evidence/probe.stderr','probe.stderr')]:
    try:s.get(R+'/'+src,str(d/dst))
    except FileNotFoundError:pass
  assert rc==0,(machine,rc,err)
  r=json.loads((d/'cpu-import-identity.json').read_text(encoding='utf-8'));assert r['passed'] and len(r['plans'])==8 and not any(r['before'].values()) and not any(r['after'].values())
  return {'rc':rc,'plans':len(r['plans']),'before':r['before'],'after':r['after'],'identities':[v['content_digest'] for v in r['identities']],'repeat_collector_stable':r['repeat_collector_stable']}
 rows=parallel(one);assert rows[HOSTS[0]]['identities']==rows[HOSTS[1]]['identities'];(OUT/'cpu-summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows))
finally:x.close()
