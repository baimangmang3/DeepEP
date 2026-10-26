"""One approved diagnostic only; defaults to a local plan, no SSH/device action."""
import argparse,json,pathlib,hashlib,shlex,time
import sys; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]))
from training_native_round import SshBackend,T,HOSTS,parallel
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from dual_host_lifecycle import launch_hosts_once,cleanup_started,cleanup_is_uncertain
R=T+'/validation-p2-ep16-overlap-hostname-20261026'
STAGE='p2-ep16-bf16-overlap'
OUT=pathlib.Path('outputs/p2_ep16_overlap_hostname_preparation')

def stop_owned(backend,host,tag):
 assert host in HOSTS;node=HOSTS.index(host)
 cmd='RUN_TAG='+shlex.quote(tag)+' timeout 180 /root/miniconda3/envs/zzy/bin/python '+shlex.quote(R+'/scripts/cleanup_own.py')+' --run-tag '+shlex.quote(tag)+' --node-rank '+str(node)
 _,out,err=backend.clients[host].exec_command(cmd,timeout=190)
 stdout,stderr=out.read().decode(),err.read().decode();rc=out.channel.recv_exit_status()
 assert rc==0,repr({'cleanup_rc':rc,'stdout':stdout,'stderr':stderr})
 return json.loads(stdout)

def cleanup_launched(backend,tag,report):
 cleanup_started(lambda host:stop_owned(backend,host,tag),report)


def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--run-tag',required=True);a=p.parse_args()
 assert a.run_tag and all(c.isalnum() or c in '_-' for c in a.run_tag)
 manifest=OUT/'rack05os7-diagnostic-deployment-manifest.json'
 mh=hashlib.sha256(manifest.read_bytes()).hexdigest()
 argv=['bash',R+'/scripts/run_dispatch_host_trace_checked.sh','<node0-or-1>']
 report={'run_tag':a.run_tag,'stage':STAGE,'manifest_sha256':mh,'launcher_argv':argv,'environment':{'RUN_TAG':a.run_tag,'TRAIN_MASTER_PORT':'29926','DISPATCH_MANIFEST_SHA256':mh},'timeout_seconds':600,'kill_after_seconds':30,'diagnostic_only':True,'no_successor_queue':True}
 if not a.execute:print(json.dumps(report,indent=2));return
 state=json.loads(pathlib.Path('outputs/training_device_progress.json').read_text(encoding='utf-8-sig'));assert state.get('p2_ep16_overlap_hostname_control',{}).get('device_authorized') is True,'P2 EP16 overlap device execution is not authorized'
 dest=pathlib.Path('outputs/training_round_'+a.run_tag)/(STAGE+'-controller');dest.mkdir(parents=True,exist_ok=True)
 x=SshBackend();x.connect();launch_attempted=False
 try:
  def verify(h):
   code=f'''import pathlib,json,subprocess,os
R=pathlib.Path({R!r});tag={a.run_tag!r};run=pathlib.Path({T!r})/'runs'/tag/{STAGE!r}
v=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(R/'scripts/verify_deployment.py'),str(R/'diagnostic-deployment-manifest.json'),{mh!r}],capture_output=True,text=True,timeout=45)
assert v.returncode==0,v.stdout+v.stderr
external=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(R/'scripts/external_launcher_gate.py'),'--run-tag',tag],capture_output=True,text=True,timeout=20)
external_report=json.loads(external.stdout);assert external.returncode in (0,12),external.stderr
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
prior=run.exists()
print(json.dumps({{'passed':not own and not listeners and not prior and external.returncode==0,'external_launcher_gate':external_report,'own_processes':own,'listeners':listeners,'prior_run_evidence':bool(prior),'manifest_check':json.loads(v.stdout)}}))'''
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
row.update(run_tag={a.run_tag!r},contract_sha256='3d85227667b4602de1b7b0f7bfe307451d672df1033f85ddcf21c4ebc24eaa47',deployment_manifest_sha256={mh!r},ppid=int((proc/'stat').read_text().rsplit(')',1)[1].split()[1]),cwd=os.readlink(proc/'cwd'),uid=(proc/'status').read_text().split('Uid:',1)[1].splitlines()[0].strip())
(marker/'identity.json').write_text(json.dumps(row));print(json.dumps(row))''')
  def recover(h):
   node=HOSTS.index(h)
   return x.python(h,a.run_tag,f'''import pathlib,json
p=pathlib.Path({T!r})/'runs'/{a.run_tag!r}/{STAGE!r}/'.node{node}-controller-once/identity.json'
if not p.exists():print('null')
else:
 row=json.loads(p.read_text());assert row['run_tag']=={a.run_tag!r} and row['node']=={node} and row['deployment_manifest_sha256']=={mh!r} and int(row['pid'])>0 and row['start_ticks']
 print(json.dumps(row))''')
  def before_next(started):
   for h in started:
    node=HOSTS.index(h)
    row=x.python(h,a.run_tag,f'''import pathlib,json
run=pathlib.Path({T!r})/'runs'/{a.run_tag!r}/{STAGE!r}
paths=[run/'logs/node{node}.rc',run/'supervisor-node{node}.rc']
print(json.dumps({{'known_rc':[int(p.read_text()) for p in paths if p.exists()]}}))''')
    if any(rc!=0 for rc in row['known_rc']):raise RuntimeError('Started peer failed before next host submission: '+h)
  launch_attempted=True;launches=launch_hosts_once(HOSTS,launch,recover,before_next,report);report['status']='supervisors_started' 
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
   states={h:state(h) for h in launches};report['startup_states']=states
   if any(r['rc'] not in (None,0) or r['supervisor_rc'] not in (None,0) or (not r['supervisor_same_pid_start_run_alive'] and r['supervisor_rc'] is None) for r in states.values()):
    report['status']='startup_failed_own_cleanup';report.setdefault('first_error',{'startup_states':states});cleanup_launched(x,a.run_tag,report);break
   if all(r['torchrun_argv_saved'] and (r['supervisor_same_pid_start_run_alive'] or r['supervisor_rc']==0) for r in states.values()):break
   time.sleep(1)
  else:
   report['status']='startup_observation_timeout_own_cleanup';report.setdefault('first_error',{'startup_observer':'did not confirm timely launch'});cleanup_launched(x,a.run_tag,report)
  progress=pathlib.Path('outputs/training_device_progress.json');d=json.loads(progress.read_text(encoding='utf-8-sig'))
  d.update(status=report['status'],device_tasks_currently_running=(report['status']=='supervisors_started' or cleanup_is_uncertain(report)),active={'run_tag':a.run_tag,'stage':STAGE} if (report['status']=='supervisors_started' or cleanup_is_uncertain(report)) else None,automatic_device_queue=[])
  progress.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
 except Exception as exc:
  report.update(status='controller_error',error=repr(exc));report.setdefault('first_error',repr(exc))
  if launch_attempted:cleanup_launched(x,a.run_tag,report)
 finally:
  progress=pathlib.Path('outputs/training_device_progress.json');d=json.loads(progress.read_text(encoding='utf-8-sig'))
  d.update(status=report.get('status','controller_error'),device_tasks_currently_running=(report.get('status')=='supervisors_started' or cleanup_is_uncertain(report)),active={'run_tag':a.run_tag,'stage':STAGE} if (report.get('status')=='supervisors_started' or cleanup_is_uncertain(report)) else None,automatic_device_queue=[])
  d['p2_ep16_overlap_hostname_control']['latest_controller_report']=str(dest/'summary.json')
  progress.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
  x.close();(dest/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
