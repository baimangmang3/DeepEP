"""Requested CPU-only owner-cleanup/FD/native-pin revisions; no device action."""
import pathlib,json,hashlib,difflib
from training_native_round import SshBackend,T,parallel,HOSTS
SRC=pathlib.Path('work/training_diagnostics/p2_ep2_serial_control');OUT=pathlib.Path('outputs/p2_ep2_serial_preparation')
R=T+'/validation-p2-ep2-serial-schema2-20261026'
p=SRC/'controller.py';s=p.read_text(encoding='utf-8')
helper='''def cleanup_host27(backend,tag,report):
 # EP2 launches only on host27; cleanup evidence must never mask the first error.
 host=HOSTS[0]
 try:report.setdefault('cleanup',{})[host]=stop_owned(backend,host,tag)
 except Exception as exc:report.setdefault('cleanup_errors',{})[host]=repr(exc)

'''
assert "report['cleanup']=parallel(lambda h:stop_owned(x,h,a.run_tag))" in s
s=s.replace('def main():',helper+'def main():').replace("report['cleanup']=parallel(lambda h:stop_owned(x,h,a.run_tag))","cleanup_host27(x,a.run_tag,report)")
s=s.replace("report['status']='startup_failed_own_cleanup';cleanup_host27", "report['status']='startup_failed_own_cleanup';report.setdefault('first_error',{'startup_states':states});cleanup_host27")
s=s.replace("report['status']='startup_observation_timeout_own_cleanup';cleanup_host27", "report['status']='startup_observation_timeout_own_cleanup';report.setdefault('first_error',{'startup_observer':'did not confirm timely launch'});cleanup_host27")
s=s.replace("report.update(status='controller_error',error=repr(exc))", "report.update(status='controller_error',error=repr(exc));report.setdefault('first_error',repr(exc))")
p.write_text(s,encoding='utf-8',newline='\n')
p=SRC/'capture_live_workers.py';s=p.read_text(encoding='utf-8');anchor="and '/deep_ep/_C' not in parts[5]:continue";assert anchor in s;s=s.replace(anchor,"and '/deep_ep/_C' not in parts[5] and '/torch_npu/_C' not in parts[5]:continue");p.write_text(s,encoding='utf-8',newline='\n')
def cpu_pins(host):
 r=json.loads(pathlib.Path('outputs/p2_qualification_schema2_deployment',host,'cpu-probe.json').read_text(encoding='utf-8'))
 return {v['path']:{'bytes':v['bytes'],'sha256':v['sha256']} for v in r['identities']['serial']['actual_maps']}
pins=cpu_pins('rack05os7');assert len(pins)==34 and pins==cpu_pins('rack05os8')
(OUT/'CPU-proven-native-pins.json').write_text(json.dumps(pins,indent=2),encoding='utf-8')
x=SshBackend();x.connect()
try:
 def one(h):
  machine='rack05os7' if h==HOSTS[0] else 'rack05os8'
  with x.clients[h].open_sftp() as f:f.put(str(SRC/'capture_live_workers.py'),R+'/scripts/capture_live_workers.py')
  r=x.python(h,'cpu_p2_serial_finalize',f'''import pathlib,json,hashlib,subprocess,ast
R=pathlib.Path({R!r});p=R/'diagnostic-deployment-manifest.json';m=json.loads(p.read_text());backup=R/'draft-manifest-before-final-review.json'
if not backup.exists():backup.write_bytes(p.read_bytes())
pins={pins!r}
for n,v in m['external_native'].items():assert pins[n]==v,n
for n,v in pins.items():
 q=pathlib.Path(n)
 with q.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
 assert q.stat().st_size==v['bytes'] and sha==v['sha256'],n
m['external_native']=pins;m['native_pin_basis']='schema2 CPU serial actual34 FD-map objects, exact path/bytes/SHA, cross-host equal; NPU lazy additions not forced equal to CPU digest'
for n in m['files']:
 q=R/n;b=q.read_bytes();m['files'][n]={{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}}
 if q.suffix=='.py':ast.parse(b.decode(),filename=n)
p.write_text(json.dumps(m,indent=2));sha=hashlib.sha256(p.read_bytes()).hexdigest()
v=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(R/'scripts/verify_deployment.py'),str(p),sha],capture_output=True,text=True);assert v.returncode==0,v.stdout+v.stderr
v=subprocess.run(['bash','-n',str(R/'scripts/run_dispatch_host_trace_checked.sh')],capture_output=True,text=True);assert v.returncode==0,v.stderr
print(json.dumps({{'passed':True,'manifest_sha256':sha,'native_pins':len(pins),'source_files':len(m['source_files']),'bash_n_rc':0,'Torch_imported':False,'device_started':False}}))''')
  with x.clients[h].open_sftp() as f:f.get(R+'/diagnostic-deployment-manifest.json',str(OUT/(machine+'-diagnostic-deployment-manifest.json')))
  return r
 rows=parallel(one);assert len({r['manifest_sha256'] for r in rows.values()})==1
 (OUT/'final-CPU-revision.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows))
finally:x.close()
old=pathlib.Path('work/training_diagnostics/p1_main_control');diff=[]
for n in ('run_dispatch_host_trace_checked.sh','controller.py','capture_live_workers.py'):
 diff.extend(difflib.unified_diff((old/n).read_text(encoding='utf-8').splitlines(keepends=True),(SRC/n).read_text(encoding='utf-8').splitlines(keepends=True),fromfile='approved-P1-controls/'+n,tofile='CPU-only-P2-serial/'+n))
(OUT/'minimal-control-diff.patch').write_text(''.join(diff),encoding='utf-8')
hashes={n:hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in ('controller.py','run_dispatch_host_trace_checked.sh','verify_deployment.py','capture_live_workers.py')}
(OUT/'control-file-sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');print(json.dumps(hashes))
