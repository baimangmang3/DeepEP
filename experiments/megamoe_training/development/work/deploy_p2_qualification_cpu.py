"""Approved isolated P2 CPU deployment; no device command or occupancy poll."""
import pathlib,json,hashlib,shlex
from training_native_round import SshBackend,T,parallel,HOSTS
SRC=pathlib.Path('work/training_candidates/p2_backward_overlap');OUT=pathlib.Path('outputs/p2_qualification_deployment');OUT.mkdir(exist_ok=True)
MAN=SRC/'qualification_manifest.json';assert hashlib.sha256(MAN.read_bytes()).hexdigest()=='d43cb9e633e9695ecfeb6e154f1c0c3db662d0d83850afa0ede0841c5eacc2ef'
M=json.loads(MAN.read_text(encoding='utf-8'))
for n,h in M['source_sha256'].items():assert hashlib.sha256((SRC/n).read_bytes()).hexdigest()==h,n
R=T+'/candidate-p2-qualification-20261026';P=R+'/p2_backward_overlap'
x=SshBackend();x.connect()
try:
 def one(h):
  x.python(h,'cpu_p2_deploy',f'''import pathlib,json
p=pathlib.Path({R!r});assert not p.exists(),'Refuse overwrite';(p/'p2_backward_overlap/tests').mkdir(parents=True);(p/'scripts').mkdir();(p/'evidence').mkdir();print(json.dumps({{'created':str(p)}}))''')
  with x.clients[h].open_sftp() as s:
   for source in SRC.rglob('*'):
    if not source.is_file() or '__pycache__' in source.parts or source.suffix=='.pyc':continue
    rel=source.relative_to(SRC);s.put(str(source),P+'/'+str(rel).replace('\\','/'))
   s.put('work/p2_qualification_cpu_probe.py',R+'/scripts/probe_cpu.py')
  check=x.python(h,'cpu_p2_deploy',f'''import pathlib,json,hashlib,ast
p=pathlib.Path({P!r});expected={M['source_sha256']!r}
assert hashlib.sha256((p/'qualification_manifest.json').read_bytes()).hexdigest()=='d43cb9e633e9695ecfeb6e154f1c0c3db662d0d83850afa0ede0841c5eacc2ef'
files={{}}
for f in sorted(p.rglob('*')):
 if f.is_file():
  b=f.read_bytes();rel=str(f.relative_to(p));files[rel]={{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}}
  if f.suffix=='.py':ast.parse(b.decode(),filename=rel)
for n,h in expected.items():assert files[n]['sha256']==h,n
print(json.dumps({{'passed':True,'files':files,'manifest_sha256':'d43cb9e633e9695ecfeb6e154f1c0c3db662d0d83850afa0ede0841c5eacc2ef','Torch_imported':False}}))''')
  command=f'''source {T}/activate-qualified-p0.sh
export P2_SOURCE_ROOT={P} P1_SOURCE_ROOT={T}/candidate-p1-host-v2/p1_pointwise POINTWISE_SO={T}/candidate-p1/build/libmegamoe_pointwise.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 timeout 120 python {R}/scripts/probe_cpu.py > {R}/evidence/cpu-probe.json 2> {R}/evidence/cpu-probe.stderr
'''
  _,o,e=x.clients[h].exec_command('bash -lc '+shlex.quote(command),timeout=140);o.read();err=e.read().decode();rc=o.channel.recv_exit_status();machine='rack05os7' if h==HOSTS[0] else 'rack05os8';d=OUT/machine;d.mkdir(exist_ok=True)
  with x.clients[h].open_sftp() as s:
   for name in ('cpu-probe.json','cpu-probe.stderr'):s.get(R+'/evidence/'+name,str(d/name))
  (d/'source-verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8');(d/'command-result.json').write_text(json.dumps({'rc':rc,'stderr':err,'command':command},indent=2),encoding='utf-8')
  assert rc==0,(machine,rc,err)
  probe=json.loads((d/'cpu-probe.json').read_text(encoding='utf-8'));assert probe['passed']
  return {'rc':rc,'plans':len(probe['plans']),'before':probe['before'],'after':probe['after'],'identities':{k:v['content_digest'] for k,v in probe['identities'].items()},'source_files':len(check['files'])}
 rows=parallel(one);(OUT/'cpu-preflight-summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 assert rows[HOSTS[0]]['identities']==rows[HOSTS[1]]['identities']
 print(json.dumps(rows))
finally:x.close()
