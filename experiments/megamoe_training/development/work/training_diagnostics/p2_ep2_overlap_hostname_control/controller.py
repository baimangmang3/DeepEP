"""One approved diagnostic only; defaults to a local plan, no SSH/device action."""
import argparse,json,pathlib,hashlib,shlex,time
import sys; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]))
from training_native_round import SshBackend,T,HOSTS,parallel
R=T+'/validation-p2-ep2-overlap-hostname-20261026'
STAGE='p2-ep2-bf16-overlap'
OUT=pathlib.Path('outputs/p2_ep2_overlap_hostname_preparation')

def stop_owned(backend,host,tag):
 assert host==HOSTS[0], 'Only host27 was launched by this EP2 contract'
 cmd='RUN_TAG='+shlex.quote(tag)+' timeout 180 /root/miniconda3/envs/zzy/bin/python '+shlex.quote(R+'/scripts/cleanup_own.py')+' --run-tag '+shlex.quote(tag)
 _,out,err=backend.clients[host].exec_command(cmd,timeout=190)
 stdout,stderr=out.read().decode(),err.read().decode();rc=out.channel.recv_exit_status()
 assert rc==0,repr({'cleanup_rc':rc,'stdout':stdout,'stderr':stderr})
 return json.loads(stdout)

def cleanup_host27(backend,tag,report):
 # EP2 launches only on host27; cleanup evidence must never mask the first error.
 host=HOSTS[0]
 try:report.setdefault('cleanup',{})[host]=stop_owned(backend,host,tag)
 except Exception as exc:report.setdefault('cleanup_errors',{})[host]=repr(exc)

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--run-tag',required=True);a=p.parse_args()
 assert a.run_tag and all(c.isalnum() or c in '_-' for c in a.run_tag)
 manifest=OUT/'rack05os7-diagnostic-deployment-manifest.json'
 mh=hashlib.sha256(manifest.read_bytes()).hexdigest()
 argv=['bash',R+'/scripts/run_dispatch_host_trace_checked.sh','0']
 report={'run_tag':a.run_tag,'stage':STAGE,'manifest_sha256':mh,'launcher_argv':argv,'environment':{'RUN_TAG':a.run_tag,'TRAIN_MASTER_PORT':'29946','DISPATCH_MANIFEST_SHA256':mh},'timeout_seconds':600,'kill_after_seconds':30,'diagnostic_only':True,'no_successor_queue':True}
 if not a.execute:print(json.dumps(report,indent=2));return
 state=json.loads(pathlib.Path('outputs/training_device_progress.json').read_text(encoding='utf-8-sig'));assert state.get('p2_ep2_overlap_hostname_control',{}).get('device_authorized') is True,'P2 overlap device execution is not authorized'
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
  if f[3]=='0A' and int(f[1].rsplit(':',1)[1],16) in (29946,27794):listeners.append({{'table':file,'local':f[1],'inode':f[9]}})
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
env=dict(os.environ,RUN_TAG={a.run_tag!r},TRAIN_MASTER_PORT='29946',DISPATCH_MANIFEST_SHA256={mh!r})
cmd=['bash','-c','set +e; timeout --signal=TERM --kill-after=30 690 bash "$1" "$2"; saved_rc=$?; printf "%s\\n" "$saved_rc" > "$3"; exit "$saved_rc"','dispatch-supervisor',{R!r}+'/scripts/run_dispatch_host_trace_checked.sh','{node}',str(run/'supervisor-node{node}.rc')]
with (run/'supervisor-node{node}.log').open('xb') as log:
 p=subprocess.Popen(cmd,env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
proc=pathlib.Path('/proc')/str(p.pid)
row={{'pid':p.pid,'pgid':os.getpgid(p.pid),'start_ticks':(proc/'stat').read_text().rsplit(')',1)[1].split()[19],'argv':cmd,'node':{node},'time_ns':time.time_ns()}}
row.update(run_tag={a.run_tag!r},contract_sha256='07cfad0e860ffdfd49b515774e7f8196f661ee58ecc57e39fd42ee7839ae9496',deployment_manifest_sha256={mh!r},ppid=int((proc/'stat').read_text().rsplit(')',1)[1].split()[1]),cwd=os.readlink(proc/'cwd'),uid=(proc/'status').read_text().split('Uid:',1)[1].splitlines()[0].strip())
(marker/'identity.json').write_text(json.dumps(row));print(json.dumps(row))''')
  launch_attempted=True;launches={HOSTS[0]:launch(HOSTS[0])};report['launches']=launches;report['status']='supervisors_started'
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
   states={HOSTS[0]:state(HOSTS[0])};report['startup_states']=states
   if any(r['rc'] not in (None,0) or r['supervisor_rc'] not in (None,0) or (not r['supervisor_same_pid_start_run_alive'] and r['supervisor_rc'] is None) for r in states.values()):
    report['status']='startup_failed_own_cleanup';report.setdefault('first_error',{'startup_states':states});cleanup_host27(x,a.run_tag,report);break
   if all(r['torchrun_argv_saved'] and (r['supervisor_same_pid_start_run_alive'] or r['supervisor_rc']==0) for r in states.values()):break
   time.sleep(1)
  else:
   report['status']='startup_observation_timeout_own_cleanup';report.setdefault('first_error',{'startup_observer':'did not confirm timely launch'});cleanup_host27(x,a.run_tag,report)
  progress=pathlib.Path('outputs/training_device_progress.json');d=json.loads(progress.read_text(encoding='utf-8-sig'))
  d.update(status=report['status'],device_tasks_currently_running=report['status']=='supervisors_started',active={'run_tag':a.run_tag,'stage':STAGE} if report['status']=='supervisors_started' else None,automatic_device_queue=[])
  progress.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
 except Exception as exc:
  report.update(status='controller_error',error=repr(exc));report.setdefault('first_error',repr(exc))
  if launch_attempted:cleanup_host27(x,a.run_tag,report)
 finally:
  progress=pathlib.Path('outputs/training_device_progress.json');d=json.loads(progress.read_text(encoding='utf-8-sig'))
  d.update(status=report.get('status','controller_error'),device_tasks_currently_running=report.get('status')=='supervisors_started',active={'run_tag':a.run_tag,'stage':STAGE} if report.get('status')=='supervisors_started' else None,automatic_device_queue=[])
  d['p2_ep2_overlap_hostname_control']['latest_controller_report']=str(dest/'summary.json')
  progress.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
  x.close();(dest/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
