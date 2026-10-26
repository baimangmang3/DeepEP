"""One authorized read-only source/own/foreign-launcher/all16 snapshot, no launch."""
import pathlib,json,hashlib,datetime
from training_native_round import SshBackend,T,HOSTS,parallel
R=T+'/validation-p2-ep2-serial-hostname-20261026';OUT=pathlib.Path('outputs/p2_ep2_serial_hostname_preparation/fresh-gate-once');OUT.mkdir(exist_ok=False)
M=pathlib.Path('outputs/p2_ep2_serial_hostname_preparation/rack05os7-diagnostic-deployment-manifest.json');mh=hashlib.sha256(M.read_bytes()).hexdigest();tag='20261026_0348_p2_prelaunch_gate'
x=SshBackend();x.connect()
try:
 def check(h):
  cpu=x.python(h,tag,f'''import subprocess,json,pathlib,os,time
r=pathlib.Path({R!r})
p=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(r/'scripts/verify_deployment.py'),str(r/'diagnostic-deployment-manifest.json'),{mh!r}],capture_output=True,text=True,timeout=45);assert p.returncode==0,p.stdout+p.stderr
q=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(r/'scripts/external_launcher_gate.py'),'--run-tag',{tag!r}],capture_output=True,text=True,timeout=20);assert q.returncode in (0,12),q.stderr
own=[];listeners=[]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\\0') if v]
  if any(v.startswith({T!r}+'/') and v.endswith(('.py','.sh')) for v in args):
   s=(proc/'stat').read_text().rsplit(')',1)[1].split();own.append({{'pid':int(proc.name),'start':s[19],'ppid':int(s[1]),'argv':args}})
 except OSError:pass
for filename in ('/proc/net/tcp','/proc/net/tcp6'):
 for line in pathlib.Path(filename).read_text().splitlines()[1:]:
  f=line.split()
  if f[3]=='0A' and int(f[1].rsplit(':',1)[1],16) in (29946,27794):listeners.append({{'table':filename,'local':f[1],'inode':f[9]}})
print(json.dumps({{'manifest':json.loads(p.stdout),'external_launcher_gate':json.loads(q.stdout),'own':own,'listeners':listeners,'time_ns':time.time_ns(),'passed':q.returncode==0 and not own and not listeners}}))''')
  # Same connected control; no model turn between CPU evidence and all-card snapshot.
  gate=x.gate(h,tag);return {'cpu':cpu,'gate':gate}
 results=parallel(check)
 free=all(v['cpu']['passed'] and v['gate']['observation']['npu_smi_rc']==0 and v['gate']['observation']['all_eight_free'] and not v['gate']['owners']['processes'] and not v['gate']['observation']['processes'] for v in results.values())
 (OUT/'snapshot.json').write_text(json.dumps({'tag':tag,'all16_free_own0_no_known_foreign_launcher':free,'hosts':results,'device_started':False},indent=2))
 progress=pathlib.Path('outputs/training_device_progress.json');d=json.loads(progress.read_text(encoding='utf-8-sig'))
 d['p2_ep2_serial_hostname_control']={'CPU_deployed':True,'CPU_help_both_passed':True,'CPU_smoke_host27_passed':True,'device_authorized':False,'pending_parent_final_review':True,'manifest_sha256':mh,'fresh_gate':str(OUT/'snapshot.json'),'gate_all16_free':free}
 d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();progress.write_text(json.dumps(d,indent=2));print(json.dumps({'all16_free_own0_no_known_foreign_launcher':free,'path':str(OUT/'snapshot.json'),'device_started':False}))
finally:x.close()
