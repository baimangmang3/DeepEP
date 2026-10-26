"""Read existing CPU build results, ELF/ldd only; no imports/load/device APIs."""
import json,pathlib,hashlib,datetime
from training_native_round import SshBackend,T,HOSTS
D=T+'/candidate-p4-compute-wave-build-20261026';OUT=pathlib.Path('outputs/p4_compute_wave_build')
x=SshBackend();x.connect()
try:
 h=HOSTS[0]
 report=x.python(h,'p4_CPU_build_result',f'''import pathlib,json,hashlib,time
r=pathlib.Path({D!r});logs=r/'logs';start=json.loads((logs/'build-start.json').read_text());p=pathlib.Path('/proc')/str(start['pid'])
try:s=(p/'stat').read_text().rsplit(')',1)[1].split();alive=s[19]==start['start_ticks'];process={{'same_PID_start_alive':alive,'state':s[0],'wchan':(p/'wchan').read_text()}}
except FileNotFoundError:process={{'same_PID_start_alive':False}}
rc={{}}
for name in ('configure','build','overall','readelf','ldd'):
 f=logs/(name+'.rc');rc[name]=int(f.read_text()) if f.exists() else None
metadata={{}}
for f in sorted(logs.glob('*')):
 if f.is_file():
  with f.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
  metadata[f.name]={{'bytes':f.stat().st_size,'sha256':sha}}
source=json.loads((r/'manifest.json').read_text());actual={{}}
for name,pin in source['source_inventory'].items():
 b=(r/name).read_bytes();actual[name]={{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}};assert actual[name]==pin,name
so=r/'build/libmegamoe_wave_compute.so';binary=None
if so.exists():
 with so.open('rb') as f:binary={{'path':str(so),'bytes':so.stat().st_size,'sha256':hashlib.file_digest(f,'sha256').hexdigest()}}
print(json.dumps({{'time_ns':time.time_ns(),'process':process,'rc':rc,'log_files':metadata,'sources':actual,'manifest_sha256':hashlib.sha256((r/'manifest.json').read_bytes()).hexdigest(),'SO':binary,'scope':'Disk bytes, compiler status and saved ELF/ldd only; no native load/Torch/device'}}))''')
 (OUT/'latest-build-status.json').write_text(json.dumps(report,indent=2))
 with x.clients[h].open_sftp() as ftp:
  for name in report['log_files']:
   ftp.get(D+'/logs/'+name,str(OUT/name))
  if report['rc']['overall']==0 and report['SO']:ftp.get(report['SO']['path'],str(OUT/'libmegamoe_wave_compute.so'))
 log=(OUT/'build.log').read_text(errors='replace') if (OUT/'build.log').exists() else ''
 warnings=[{'line':i,'text':line} for i,line in enumerate(log.splitlines(),1) if 'warning:' in line]
 errors=[{'line':i,'text':line} for i,line in enumerate(log.splitlines(),1) if 'error:' in line or 'undefined reference' in line]
 (OUT/'compiler-diagnostics.json').write_text(json.dumps({'warnings':warnings,'first_errors':errors[:16]},indent=2))
 progress=pathlib.Path('outputs/training_device_progress.json');s=json.loads(progress.read_text(encoding='utf-8-sig'));s['cpu_p4_build']={'active':report['rc']['overall'] is None and report['process']['same_PID_start_alive'],'rc':report['rc'],'latest_report':str(OUT/'latest-build-status.json'),'device_authorized':False,'CPU_permission_consumed':True};s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();progress.write_text(json.dumps(s,indent=2))
 print(json.dumps({'process':report['process'],'rc':report['rc'],'warning_count':len(warnings),'first_errors':errors[:3],'SO':report['SO']}))
finally:x.close()
