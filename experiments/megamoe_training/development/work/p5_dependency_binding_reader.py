"""Disk-only exact requested dependency/tool hashes; never execute either tool."""
import pathlib,hashlib,json,shutil,time,os,sys,traceback
ROOT=pathlib.Path(__file__).resolve().parent
EXPECTED='57db77e06a1b82522e79afa633b79c67a6a77edeee92832e7d3b09004d6c47dc'
request_path=ROOT/'binding-request.json';raw=request_path.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
request=json.loads(raw);assert str(ROOT)==request['remote_new_files_root'] and str(ROOT).startswith('/home/zzy/')
paths=request['external_dependency_paths'];assert len(paths)==594 and len(set(paths))==594
report={'passed':False,'host':'177.22.27.1','scope':request['scope'],'request_sha256':EXPECTED,'reader_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'start_wall_ns':time.time_ns(),'start_monotonic_ns':time.monotonic_ns(),'dependencies':{},'errors':[],'tools_executed':False,'source_deployed':False,'NPU_queried':False}
def save():
 p=ROOT/'result.json';tmp=ROOT/'result.json.tmp';tmp.write_text(json.dumps(report,indent=2));os.replace(tmp,p)
def pin(path):
 p=pathlib.Path(path);real=p.resolve(strict=True)
 with real.open('rb') as f:
  before=os.fstat(f.fileno());digest=hashlib.file_digest(f,'sha256').hexdigest();after=os.fstat(f.fileno())
 assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'File mutated while hashing'
 return {'bytes':after.st_size,'sha256':digest}, {'requested_path':str(p),'realpath':str(real),'device':after.st_dev,'inode':after.st_ino,'mtime_ns':after.st_mtime_ns}
try:
 for index,path in enumerate(paths):
  try:value,identity=pin(path);report['dependencies'][path]=dict(value,**identity)
  except Exception as exc:
   report['dependencies'][path]={'bytes':None,'sha256':None,'error':repr(exc)};report['errors'].append({'path':path,'error':repr(exc)})
  if index%25==0:save()
 tools={}
 for name,path in [('compiler',request['compiler']['path']),('readelf',shutil.which('readelf'))]:
  try:
   if not path:raise FileNotFoundError('No existing readelf on PATH')
   value,identity=pin(path)
   # Compiler path must equal compile_objects' existing exact compiler pin;
   # readelf specifically binds the resolved existing executable.
   tools[name]=dict(path=path if name=='compiler' else identity['realpath'],**value)
   report[name]=dict(tools[name],**identity)
   if name=='compiler' and value['sha256']!=request['compiler']['expected_sha256']:raise ValueError('Actual compiler SHA differs from recorded build')
  except Exception as exc:report['errors'].append({'tool':name,'path':path,'error':repr(exc)});report[name]={'path':path,'bytes':None,'sha256':None,'error':repr(exc)}
 bindings={'compiler':tools.get('compiler',report.get('compiler')),'readelf':tools.get('readelf',report.get('readelf')),'external_dependencies':{p:{k:v.get(k) for k in ('bytes','sha256')} for p,v in report['dependencies'].items()},'device_elf_machine':request['device_elf_machine'],'device_symbols':request['device_symbols']}
 assert set(bindings)=={'compiler','readelf','external_dependencies','device_elf_machine','device_symbols'}
 (ROOT/'bindings.json').write_text(json.dumps(bindings,indent=2))
 report.update(passed=not report['errors'],requested_dependencies=len(paths),completed_dependency_entries=len(report['dependencies']),bindings_sha256=hashlib.sha256((ROOT/'bindings.json').read_bytes()).hexdigest(),device_elf_fields_origin='Archived CPU ELF values supplied by approved request; not device observation')
except BaseException as exc:report['errors'].append({'first_error':repr(exc),'traceback':traceback.format_exc()})
finally:
 report.update(end_wall_ns=time.time_ns(),end_monotonic_ns=time.monotonic_ns());save()
 print(json.dumps({'passed':report['passed'],'dependencies':len(report['dependencies']),'errors':report['errors'],'compiler':report.get('compiler'),'readelf':report.get('readelf'),'tools_executed':False}))
raise SystemExit(0 if report['passed'] else 1)
