"""Exact reviewed isolated paired object compile, no link/install/import/device."""
import pathlib,json,hashlib,os,shlex,paramiko,datetime
REVIEW=pathlib.Path('outputs/p5_identity64_cpu_compile_parent_review.json');r=json.loads(REVIEW.read_text());SRC=pathlib.Path(r['package_local']);OUT=pathlib.Path(r['output_local']);OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(SRC/'manifest.json')==r['manifest_sha256'] and sha(SRC/'compile_objects.py')==r['compile_objects_sha256'] and sha(SRC/'cpu_recorder.py')==r['cpu_recorder_sha256'] and sha(r['bindings'])==r['bindings_sha256']
m=json.loads((SRC/'manifest.json').read_text());expected={}
for flavor,files in m['snapshot_files'].items():
 assert len(files)==88
 for rel,pin in files.items():
  name='snapshots/'+flavor+'/'+rel;p=SRC/name;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'];expected[name]=pin
for name,key in [('manifest.json','manifest_sha256'),('compile_objects.py','compile_objects_sha256'),('cpu_recorder.py','cpu_recorder_sha256')]:expected[name]={'bytes':(SRC/name).stat().st_size,'sha256':r[key]}
expected['bindings.json']={'bytes':pathlib.Path(r['bindings']).stat().st_size,'sha256':r['bindings_sha256']}
D=r['package_remote'];RUN=r['output_remote'];TAG=r['run_tag'];host=r['host'];PY='/root/miniconda3/envs/zzy/bin/python';T='/home/zzy/deploy/megamoe-20261024/training';ACT=T+'/activate-qualified-p0.sh'
c=paramiko.SSHClient();c.load_system_host_keys();c.connect(host,username='root',password=os.environ['DEPLOY_PASSWORD_27'],timeout=15,banner_timeout=15,auth_timeout=15)
def call(code,seconds=30):
 cmd=shlex.quote(PY)+" - <<'P5_CONTROL_CPU'\n"+code+'\nP5_CONTROL_CPU\n';_,o,e=c.exec_command(cmd,timeout=seconds);stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status();assert rc==0,(rc,stdout,stderr);return json.loads(stdout)
try:
 pre=call(f'''import pathlib,json,os
root=pathlib.Path({D!r});rows=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
  if any(v.startswith({T!r}+'/') and v.endswith(('.py','.sh')) for v in args):
   s=(p/'stat').read_text().rsplit(')',1)[1].split();rows.append({{'pid':int(p.name),'start':s[19],'ppid':int(s[1]),'argv':args}})
 except OSError:pass
print(json.dumps({{'own_task_processes':rows,'package_exists':root.exists(),'run_exists':pathlib.Path({RUN!r}).exists(),'no_occupancy_query':True}}))''')
 (OUT/'prelaunch-existing-own-results.json').write_text(json.dumps(pre,indent=2));assert not pre['own_task_processes'] and not pre['package_exists'] and not pre['run_exists']
 call(f'''import pathlib,json
d=pathlib.Path({D!r});d.mkdir(parents=True,exist_ok=False)
for rel in {list(expected)!r}:d.joinpath(rel).parent.mkdir(parents=True,exist_ok=True)
(d/'logs').mkdir();print(json.dumps({{'created':str(d)}}))''')
 with c.open_sftp() as ftp:
  for name in expected:ftp.put(str(pathlib.Path(r['bindings']) if name=='bindings.json' else SRC/name),D+'/'+name)
 verified=call(f'''import pathlib,json,hashlib
d=pathlib.Path({D!r});rows={{}}
for name,pin in {expected!r}.items():
 p=d/name
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert h==pin['sha256'] and p.stat().st_size==pin['bytes'],name;rows[name]=pin
activation=pathlib.Path({ACT!r});act_sha=hashlib.sha256(activation.read_bytes()).hexdigest();assert act_sha=='90b1bc782cf83b9d331dd743fb09a865e163a9ed70604291d5b32d2669ad3d3a'
print(json.dumps({{'passed':True,'files':rows,'activation':{{'path':str(activation),'bytes':activation.stat().st_size,'sha256':act_sha}}}}))''',60)
 (OUT/'source-bindings-activation-verification.json').write_text(json.dumps(verified,indent=2))
 driver=[PY,D+'/compile_objects.py','--compile-objects','--bindings',D+'/bindings.json','--output',RUN]
 command='nice -n 10 '+shlex.join(driver)
 script=f'''#!/usr/bin/env bash
set -u
source {ACT}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2 RUN_TAG={TAG}
set +e
{command} > {D}/logs/program.stdout 2> {D}/logs/program.stderr &
driver_pid=$!
{PY} - "$driver_pid" <<'P5_ID'
import pathlib,json,sys,time,os
p=pathlib.Path('/proc')/sys.argv[1];s=(p/'stat').read_text().rsplit(')',1)[1].split()
value=dict(pid=int(sys.argv[1]),start=s[19],uid=p.stat().st_uid,ppid=int(s[1]),session=int(s[3]),argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\\0') if v],time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),environment={{k:os.getenv(k) for k in ('ASCEND_HOME_PATH','LD_LIBRARY_PATH','PATH','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MAX_JOBS','RUN_TAG')}})
pathlib.Path({D!r}+'/logs/driver.identity.json').write_text(json.dumps(value,indent=2))
P5_ID
wait "$driver_pid"
rc=$?
printf '%s\\n' "$rc" > {D}/logs/program.rc
exit "$rc"
'''
 (OUT/'run-pair-recorded.sh').write_text(script,newline='\n')
 with c.open_sftp() as ftp:ftp.put(str(OUT/'run-pair-recorded.sh'),D+'/logs/run-pair-recorded.sh')
 p=pathlib.Path('outputs/training_device_progress.json');s=json.loads(p.read_text(encoding='utf-8-sig'));s['p5_identity64_CPU_object_compile']={'permission_consumed':True,'launch_outcome':'starting_or_unknown','active':True,'run_tag':TAG,'remote':D,'output':RUN,'device_authorized':False};p.write_text(json.dumps(s,indent=2))
 launched=call(f'''import subprocess,pathlib,json,time,os
d=pathlib.Path({D!r})
with (d/'logs/outer.log').open('xb') as log:p=subprocess.Popen(['bash',str(d/'logs/run-pair-recorded.sh')],start_new_session=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
proc=pathlib.Path('/proc')/str(p.pid);f=(proc/'stat').read_text().rsplit(')',1)[1].split()
row={{'pid':p.pid,'start':f[19],'uid':proc.stat().st_uid,'ppid':int(f[1]),'session':int(f[3]),'argv':[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\\0') if v],'time_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns()}}
(d/'logs/recorder.identity.json').write_text(json.dumps(row,indent=2))
end=time.monotonic()+2
while not (d/'logs/driver.identity.json').exists() and time.monotonic()<end:time.sleep(.05)
print(json.dumps({{'recorder':row,'driver':json.loads((d/'logs/driver.identity.json').read_text()) if (d/'logs/driver.identity.json').exists() else None,'run_tag':{TAG!r},'scope':'CPU paired-c objects only; no link/load/NPU/occupancy'}}))''')
 (OUT/'launch.json').write_text(json.dumps(launched,indent=2));s=json.loads(p.read_text());s['p5_identity64_CPU_object_compile'].update(launch_outcome='started',launch=launched,local_evidence=str(OUT));s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(s,indent=2));print(json.dumps(launched))
finally:c.close()
