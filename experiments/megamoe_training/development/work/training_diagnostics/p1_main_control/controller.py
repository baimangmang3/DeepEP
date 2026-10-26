"""One approved diagnostic only; defaults to a local plan, no SSH/device action."""
import argparse,json,pathlib,hashlib,shlex,time
import sys; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]))
from training_native_round import SshBackend,T,HOSTS,parallel
R=T+'/validation-p1-main-routed-20261026'
STAGE='p1-host-v2-main-bf16-balanced'
OUT=pathlib.Path('outputs/p1_main_control_preparation')

def stop_owned(backend,host,tag):
 # Run-local env and PGID membership are revalidated before either signal.
 return backend.python(host,tag,f'''import os,pathlib,json,signal,time
tag={tag!r};run={T!r}+'/runs/'+tag+'/{STAGE}';events=[]
def candidates():
 rows=[]
 for p in pathlib.Path('/proc').glob('[0-9]*'):
  try:
   env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\\0') if b'=' in v)
   if env.get(b'RUN_TAG')!=tag.encode():continue
   a=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
   if not a:continue
   allowed=({R!r}+'/scripts/run_dispatch_host_trace_checked.sh','/home/zzy/deploy/megamoe-20261024/training/candidate-p1-host-v2/p1_pointwise/validate_training_p1.py','torch.distributed.run')
   if not any(v in allowed for v in a):continue
   rows.append({{'pid':int(p.name),'pgid':os.getpgid(int(p.name)),'argv':a,'start':(p/'stat').read_text().rsplit(')',1)[1].split()[19]}})
  except (OSError,ValueError):pass
 return rows
def send(sig):
 rows=candidates();groups={{r['pgid'] for r in rows}}
 for g in groups:
  members=[];safe=True
  for p in pathlib.Path('/proc').glob('[0-9]*'):
   try:
    if os.getpgid(int(p.name))!=g:continue
    e=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\\0') if b'=' in v)
    if e.get(b'RUN_TAG')!=tag.encode():safe=False
    members.append(int(p.name))
   except (OSError,ValueError):pass
  identity_recheck=[]
  for r in rows:
   if r['pgid']!=g:continue
   try:
    p=pathlib.Path('/proc')/str(r['pid']);now=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
    env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\\0') if b'=' in v)
    matches=now==r['start'] and env.get(b'RUN_TAG')==tag.encode() and os.getpgid(r['pid'])==g
    identity_recheck.append({{'pid':r['pid'],'saved_start':r['start'],'current_start':now,'matches':matches}})
    if not matches:safe=False
   except OSError:safe=False
  event={{'pgid':g,'members':members,'safe_own_run_env':safe,'identity_recheck':identity_recheck,'signal':sig}}
  if safe and members:
   try:os.killpg(g,sig);event['sent']=True
   except ProcessLookupError:event['gone']=True
  events.append(event)
send(signal.SIGTERM)
deadline=time.monotonic()+30
while candidates() and time.monotonic()<deadline:time.sleep(.5)
if candidates():send(signal.SIGKILL)
result={{'scope':'only this explicit RUN_TAG groups; no foreign signal','events':events,'remaining':candidates()}}
pathlib.Path(run,'own-startup-cleanup.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))''')

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--run-tag',required=True);a=p.parse_args()
 assert a.run_tag and all(c.isalnum() or c in '_-' for c in a.run_tag)
 manifest=OUT/'rack05os7-diagnostic-deployment-manifest.json'
 mh=hashlib.sha256(manifest.read_bytes()).hexdigest()
 argv=['bash',R+'/scripts/run_dispatch_host_trace_checked.sh','<node0-or-1>']
 report={'run_tag':a.run_tag,'stage':STAGE,'manifest_sha256':mh,'launcher_argv':argv,'environment':{'RUN_TAG':a.run_tag,'TRAIN_MASTER_PORT':'29926','DISPATCH_MANIFEST_SHA256':mh},'timeout_seconds':600,'kill_after_seconds':30,'diagnostic_only':True,'no_successor_queue':True}
 if not a.execute:print(json.dumps(report,indent=2));return
 dest=pathlib.Path('outputs/training_round_'+a.run_tag)/(STAGE+'-controller');dest.mkdir(parents=True,exist_ok=True)
 x=SshBackend();x.connect();launch_attempted=False
 try:
  def verify(h):
   code=f'''import pathlib,json,subprocess,os
R=pathlib.Path({R!r});tag={a.run_tag!r};run=pathlib.Path({T!r})/'runs'/tag/{STAGE!r}
v=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(R/'scripts/verify_deployment.py'),str(R/'diagnostic-deployment-manifest.json'),{mh!r}],capture_output=True,text=True,timeout=45)
assert v.returncode==0,v.stdout+v.stderr
own=[];listeners=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
  if any(v.startswith({T!r}+'/') and v.endswith(('.py','.sh')) for v in args):own.append({{'pid':int(p.name),'argv':args}})
 except OSError:pass
for file in ('/proc/net/tcp','/proc/net/tcp6'):
 for line in pathlib.Path(file).read_text().splitlines()[1:]:
  f=line.split()
  if f[3]=='0A' and int(f[1].rsplit(':',1)[1],16) in (29926,27774):listeners.append({{'table':file,'local':f[1],'inode':f[9]}})
prior=run.exists() and (any(run.glob('.node*-*')) or any(run.glob('rank*.json')) or any(run.glob('host-trace/*.jsonl')))
print(json.dumps({{'passed':not own and not listeners and not prior,'own_processes':own,'listeners':listeners,'prior_run_evidence':bool(prior),'manifest_check':json.loads(v.stdout)}}))'''
   return x.python(h,a.run_tag,code)
  checks=parallel(verify);report['checks']=checks
  if not all(v['passed'] for v in checks.values()):report['status']='blocked_cpu_own_listener_or_prior';return
  gates=parallel(lambda h:x.gate(h,a.run_tag));report['gates']=gates
  if any(g['observation']['npu_smi_rc']!=0 or not g['observation']['all_eight_free'] or g['owners']['processes'] or g['observation']['processes'] for g in gates.values()):report['status']='occupied_or_uncertain';return
  def launch(h):
   node=HOSTS.index(h)
   return x.python(h,a.run_tag,f'''import pathlib,os,json,subprocess,time
run=pathlib.Path({T!r})/'runs'/{a.run_tag!r}/{STAGE!r};run.mkdir(parents=True,exist_ok=True)
marker=run/'.node{node}-controller-once';marker.mkdir()
env=dict(os.environ,RUN_TAG={a.run_tag!r},TRAIN_MASTER_PORT='29926',DISPATCH_MANIFEST_SHA256={mh!r})
cmd=['bash','-c','set +e; timeout --signal=TERM --kill-after=30 690 bash "$1" "$2"; saved_rc=$?; printf "%s\\n" "$saved_rc" > "$3"; exit "$saved_rc"','dispatch-supervisor',{R!r}+'/scripts/run_dispatch_host_trace_checked.sh','{node}',str(run/'supervisor-node{node}.rc')]
with (run/'supervisor-node{node}.log').open('xb') as log:
 p=subprocess.Popen(cmd,env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
proc=pathlib.Path('/proc')/str(p.pid)
row={{'pid':p.pid,'pgid':os.getpgid(p.pid),'start_ticks':(proc/'stat').read_text().rsplit(')',1)[1].split()[19],'argv':cmd,'node':{node},'time_ns':time.time_ns()}}
(marker/'identity.json').write_text(json.dumps(row));print(json.dumps(row))''')
  launch_attempted=True;launches=parallel(launch);report['launches']=launches;report['status']='supervisors_started'
  # Detect the late per-node gate failure while the other side might rendezvous.
  deadline=time.monotonic()+50
  while time.monotonic()<deadline:
   def state(h):
    n=HOSTS.index(h)
    ident=launches[h]
    return x.python(h,a.run_tag,f'''import pathlib,json,os
run=pathlib.Path({T!r})/'runs'/{a.run_tag!r}/{STAGE!r};p=run/'logs';proc=pathlib.Path('/proc')/str({ident['pid']})
r=p/'node{n}.rc';outer=run/'supervisor-node{n}.rc';alive=False
try:
 env=dict(v.split(b'=',1) for v in (proc/'environ').read_bytes().split(b'\\0') if b'=' in v)
 alive=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]=={ident['start_ticks']!r} and env.get(b'RUN_TAG')=={a.run_tag!r}.encode()
except OSError:pass
print(json.dumps({{'rc':int(r.read_text()) if r.exists() else None,'supervisor_rc':int(outer.read_text()) if outer.exists() else None,'supervisor_same_pid_start_run_alive':alive,'torchrun_argv_saved':(p/'launch-node{n}.json').exists()}}))''')
   states=parallel(state);report['startup_states']=states
   if any(r['rc'] not in (None,0) or r['supervisor_rc'] not in (None,0) or (not r['supervisor_same_pid_start_run_alive'] and r['supervisor_rc'] is None) for r in states.values()):
    report['status']='startup_failed_own_cleanup';report['cleanup']=parallel(lambda h:stop_owned(x,h,a.run_tag));break
   if all(r['torchrun_argv_saved'] and (r['supervisor_same_pid_start_run_alive'] or r['supervisor_rc']==0) for r in states.values()):break
   time.sleep(1)
  else:
   report['status']='startup_observation_timeout_own_cleanup';report['cleanup']=parallel(lambda h:stop_owned(x,h,a.run_tag))
  progress=pathlib.Path('outputs/training_device_progress.json');d=json.loads(progress.read_text(encoding='utf-8-sig'))
  d.update(status=report['status'],device_tasks_currently_running=report['status']=='supervisors_started',active={'run_tag':a.run_tag,'stage':STAGE} if report['status']=='supervisors_started' else None,automatic_device_queue=[])
  progress.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
 except Exception as exc:
  report.update(status='controller_error',error=repr(exc))
  if launch_attempted:report['cleanup']=parallel(lambda h:stop_owned(x,h,a.run_tag))
 finally:
  progress=pathlib.Path('outputs/training_device_progress.json');d=json.loads(progress.read_text(encoding='utf-8-sig'))
  d.update(status=report.get('status','controller_error'),device_tasks_currently_running=report.get('status')=='supervisors_started',active={'run_tag':a.run_tag,'stage':STAGE} if report.get('status')=='supervisors_started' else None,automatic_device_queue=[])
  d['p1_main_control']['latest_controller_report']=str(dest/'summary.json')
  progress.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
  x.close();(dest/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
