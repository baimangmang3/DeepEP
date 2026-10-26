"""Single reviewed CPU-only deployment. No occupancy/device/PG/session operations."""
import pathlib,json,hashlib,shlex
from training_native_round import SshBackend,T,parallel,HOSTS
OUT=pathlib.Path('outputs/p1_schema2_cpu_deployment');OUT.mkdir(exist_ok=False)
SRC=pathlib.Path('work/training_candidates/p1_qualification_review')
PROBE=pathlib.Path('work/p1_qualification_cpu_probe.py')
R=T+'/candidate-p1-qualification-schema2-20261026'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(SRC/'manifest.json')=='a5b455f47b40efa6f08faae1c96855c4e335872558b33718af095aa794984129'
assert sha(PROBE)=='8767232898410a80536fade958683e3dc13fc3dd49eec463f15a37d0580d5d5e'
m=json.loads((SRC/'manifest.json').read_text())
expected={k.replace('\\','/'):v for k,v in m['immutable_copied_files'].items()}
expected.update({k:v['review_sha256'] for k,v in m['files'].items()})
expected['manifest.json']=sha(SRC/'manifest.json')
for rel,h in expected.items():assert sha(SRC/rel)==h,rel
prep=json.loads(pathlib.Path('work/p1_qualification_cpu_probe_preparation.json').read_text())
x=SshBackend();x.connect()
try:
 def one(host):
  machine='rack05os7' if host==HOSTS[0] else 'rack05os8';d=OUT/machine;d.mkdir()
  x.python(host,'p1_schema2_cpu_setup',f'''import pathlib,json
r=pathlib.Path({R!r});assert not r.exists(),'Refuse overwrite'
for s in ('p1_pointwise/native','comparison','scripts','evidence'):r.joinpath(s).mkdir(parents=True,exist_ok=True)
print(json.dumps({{'created':str(r)}}))''')
  with x.clients[host].open_sftp() as s:
   for rel in expected:s.put(str(SRC/rel),R+'/'+rel)
   s.put(str(PROBE),R+'/scripts/probe_cpu.py')
  verification=x.python(host,'p1_schema2_cpu_verify',f'''import pathlib,json,hashlib,ast
r=pathlib.Path({R!r});rows={{}}
for rel,h in {expected!r}.items():
 b=(r/rel).read_bytes();assert hashlib.sha256(b).hexdigest()==h,rel
 if rel.endswith('.py'):ast.parse(b.decode(),filename=rel)
 rows[rel]={{'sha256':h,'bytes':len(b)}}
assert hashlib.sha256((r/'scripts/probe_cpu.py').read_bytes()).hexdigest()=={sha(PROBE)!r}
print(json.dumps({{'passed':True,'files':rows,'Torch_imported':False}}))''')
  args=['--run-cpu']
  for k,v in prep['explicit_proposed_paths'].items():args+=['--'+k.replace('_','-'),v]
  cmd=f'source {T}/activate-qualified-p0.sh\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nnice -n 10 timeout 120 python '+shlex.join([R+'/scripts/probe_cpu.py',*args])+f' > {R}/evidence/probe.stdout 2> {R}/evidence/probe.stderr\n'
  _,o,e=x.clients[host].exec_command('bash -lc '+shlex.quote(cmd),timeout=140)
  stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status()
  (d/'source-verification.json').write_text(json.dumps(verification,indent=2))
  (d/'command-result.json').write_text(json.dumps({'command':cmd,'rc':rc,'stdout':stdout,'stderr':stderr},indent=2))
  with x.clients[host].open_sftp() as s:
   for source,target in [('evidence/cpu-import-identity.json','cpu-import-identity.json'),('evidence/probe.stdout','probe.stdout'),('evidence/probe.stderr','probe.stderr')]:
    try:s.get(R+'/'+source,str(d/target))
    except FileNotFoundError:pass
  assert rc==0,(machine,rc)
  report=json.loads((d/'cpu-import-identity.json').read_text())
  assert report['passed'] and len(report['plans'])==8 and not any(report['before'].values()) and not any(report['after'].values())
  ids=[v['initial']['content_digest'] for v in report['identities']]
  assert report['repeat_collector_stable']
  return {'rc':rc,'plans':8,'identities':ids,'before':report['before'],'after':report['after'],'repeat_collector_stable':True}
 results=parallel(one)
 assert results[HOSTS[0]]['identities']==results[HOSTS[1]]['identities']
 (OUT/'cpu-summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
finally:x.close()
