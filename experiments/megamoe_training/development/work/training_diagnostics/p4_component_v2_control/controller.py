"""One P4 component RUN. Default CPU plan; --execute needs exact separate permit."""
import argparse,hashlib,json,pathlib,sys,time,traceback,shlex
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1])]
from training_native_round import SshBackend,HOSTS,parallel
from process_control import T,R,STAGE,SO,run_path
from verify_report import audit
from component_proof import compare
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def local_backend_identity(manifest):
    path=pathlib.Path(sys.modules[SshBackend.__module__].__file__).resolve()
    expected_path=(HERE.parents[1]/'training_native_round.py').resolve()
    value=dict(bytes=path.stat().st_size,sha256=sha(path))
    if path!=expected_path or value!=manifest['local_backend']:raise ValueError('Local SSH backend source changed/escaped')
    return dict(path=str(path),**value)
def cleanup_call(backend,host,tag,code):
    # The inherited backend has a hard125s remote limit; do not use it for a
    # worker40 + harvest40 + two parent40 =160s cleanup. Only this call gets200s.
    cmd='RUN_TAG='+shlex.quote(tag)+" timeout --signal=TERM --kill-after=10 200 /root/miniconda3/envs/zzy/bin/python - <<'P4_CLEANUP_PY'\n"+code+'\nP4_CLEANUP_PY\n'
    _,out,err=backend.clients[host].exec_command(cmd,timeout=215)
    stdout,stderr=out.read().decode(),err.read().decode();rc=out.channel.recv_exit_status()
    if rc:raise RuntimeError('Bounded cleanup rc='+str(rc)+': '+stderr+' '+stdout)
    return json.loads(stdout)
def authorize(a,manifest,mh):
    permit=json.loads(a.authorization_file.read_bytes())
    required=dict(device_authorized=True,run_tag=a.run_tag,manifest_sha256=mh,controller_sha256=sha(__file__),
                  component_contract_sha256=manifest['component_contract_sha256'],SO=SO)
    if any(permit.get(k)!=v for k,v in required.items()):raise ValueError('Separate exact RUN/source/SO permission required')
    return permit
def launch_recorded(launch,recover,report):
    try:report['launch']=launch();return report['launch']
    except Exception as exc:
        report.setdefault('first_error',repr(exc))
        try:report['launch_recovered']=recover()
        except Exception as recovery:report['launch_recovery_error']=repr(recovery)
        raise
def runtime_decision(samples,state):
    if any(s.get('conflicts') or s.get('launchers',{}).get('controllers') for s in samples.values()):return 'confirmed_unapproved_activity'
    if any(s.get('error') or s.get('npu_smi_rc')!=0 or s.get('all_cards_accounted') is not True for s in samples.values()):return 'resource_sample_error'
    unknown=[v for s in samples.values() for v in s.get('unknown_rows',[])+s.get('launchers',{}).get('errors',[])+s.get('launchers',{}).get('unknown_candidates',[])]
    if unknown:
        worker=state.get('worker_identity');rc=state.get('worker_status',{}).get('worker_rc')
        if worker and type(rc) is int and all(v.get('pid')==worker['pid'] and v.get('proc_missing') for v in unknown):return 'own_exit_stale_observation'
        return 'unresolved_observation' # not a proof of foreign execution
    return 'clear_sample'
def real_rcs_pass(state):
    ws=state.get('worker_status',{})
    return all(type(v) is int and v==0 for v in (state.get('recorder_rc'),state.get('supervisor_rc'),ws.get('worker_rc'))) and ws.get('timeout') is False and ws.get('passed') is True and ws.get('cleanup_errors')==[]

def release_decision(samples,state):
    if state.get('own_remaining') or state.get('own_identity_unknown'):return 'own_release_unconfirmed'
    for s in samples.values():
        launch=s.get('launchers',{})
        if s.get('npu_smi_rc')!=0 or s.get('error') or not s.get('all_cards_accounted') or s.get('unknown_rows') or s.get('allowed_owned_rows') or launch.get('errors') or launch.get('unknown_candidates'):return 'own_release_unconfirmed'
    return 'released_free' if all(s.get('all_eight_free') is True for s in samples.values()) else 'released_post_busy'
def resolve_unknowns(history,state):
    worker=state.get('worker_identity');rc=state.get('worker_status',{}).get('worker_rc');unresolved=[]
    for entry in history:
        for sample in entry['observations'].values():
            for row in sample.get('unknown_rows',[])+sample.get('launchers',{}).get('errors',[])+sample.get('launchers',{}).get('unknown_candidates',[]):
                saved=row.get('saved_worker_identity')
                candidate=row.get('candidate_identity') or row.get('current_proc')
                born_matches=worker and candidate and all(candidate.get(k)==worker.get(k) for k in ('pid','start','uid','argv','run_tag','role','visible'))
                exit_matches=worker and saved and saved['pid']==worker['pid'] and saved['start']==worker['start'] and row.get('proc_missing') is True and type(rc) is int
                if not (born_matches or exit_matches):unresolved.append(row)
    return unresolved
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--run-tag',required=True);p.add_argument('--authorization-file',type=pathlib.Path);a=p.parse_args()
    run_path(a.run_tag);mf=HERE/'manifest.json';m=json.loads(mf.read_bytes());mh=sha(mf)
    backend_identity=local_backend_identity(m)
    report=dict(run_tag=a.run_tag,stage=STAGE,mode='execute' if a.execute else 'CPU_plan',manifest_sha256=mh,local_backend=backend_identity,
        worker_argv=m['worker_argv_template'],worker_limit_seconds=180,TERM_wait_seconds=30,recorder_harvest_seconds=40,
        roles=['worker','worker_recorder','supervisor'],no_torchrun_or_PG=True,device_authorized=False,performance_authorized=False,no_successor_queue=True)
    if not a.execute:print(json.dumps(report,indent=2));return
    if a.authorization_file is None:p.error('--execute requires separate authorization-file')
    permit=authorize(a,m,mh);report['device_authorized']=True
    dest=pathlib.Path('outputs/training_round_'+a.run_tag)/(STAGE+'-controller');dest.mkdir(parents=True,exist_ok=False)
    x=SshBackend();started=False
    def call(h,code):return x.python(h,a.run_tag,code)
    def state():return call(HOSTS[0],f'''import sys,pathlib,json
sys.path.insert(0,{R!r}+'/scripts');from process_control import *
run=run_path({a.run_tag!r});result={{}}
for field,file in (('worker_identity','worker.identity.json'),('worker_status','worker-status.json'),('recorder_rc','recorder.rc'),('supervisor_rc','supervisor.rc'),('supervisor_identity','supervisor.identity.json')):
 if (run/file).exists():result[field]=read(run/file)
result['own_remaining']=[];result['own_identity_unknown']=[]
for role in ('worker','recorder','supervisor','bootstrap'):
 marker=run/(role+'.identity.json')
 if marker.exists():
  saved=read(marker)
  try:
   if alive(saved):result['own_remaining'].append(saved)
  except Exception as exc:result['own_identity_unknown'].append({{'role':role,'error':repr(exc)}})
print(json.dumps(result))''')
    def sample(h,worker=None):
        before=dict(time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns())
        result=call(h,f'''import sys,json
sys.path.insert(0,{R!r}+'/scripts');from resource_gate import sample
print(json.dumps(sample({a.run_tag!r},{worker!r})))''')
        result['controller_receive_window']=dict(before=before,after=dict(time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns()),cross_host_clock_calibrated=False)
        return result
    def stop():return cleanup_call(x,HOSTS[0],a.run_tag,f'''import sys,json
sys.path.insert(0,{R!r}+'/scripts');from process_control import cleanup_worker
print(json.dumps(cleanup_worker({a.run_tag!r},parents=True)))''')
    try:
        x.connect()
        def verify(h):return call(h,f'''import subprocess,json,pathlib
cmd=['/root/miniconda3/envs/zzy/bin/python',{R!r}+'/scripts/verify_deployment.py',{R!r}+'/manifest.json',{mh!r}]+({['--worker-source']!r} if {h==HOSTS[0]!r} else [])
p=subprocess.run(cmd,capture_output=True,text=True,timeout=45);assert p.returncode==0,p.stdout+p.stderr
print(json.dumps({{'passed':True,'prior_RUN':(pathlib.Path({T!r})/'runs'/{a.run_tag!r}/{STAGE!r}).exists()}}))''')
        checks=parallel(verify);report['deployment_checks']=checks
        if any(v['prior_RUN'] for v in checks.values()):raise RuntimeError('Repeated RUN refused')
        gates=parallel(lambda h:sample(h));report['fresh_gate']=gates
        if not all(v.get('all_eight_free') is True for v in gates.values()):report['status']='blocked_before_launch';return
        def launch():return call(HOSTS[0],f'''import sys,subprocess,os,pathlib,json
sys.path.insert(0,{R!r}+'/scripts');from process_control import *
run=run_path({a.run_tag!r});run.mkdir(parents=True,exist_ok=False);save(run/'authorization.json',{permit!r})
env=dict(os.environ,RUN_TAG={a.run_tag!r},P4_ROLE='outer_recorder',ASCEND_RT_VISIBLE_DEVICES='0')
with (run/'outer.log').open('xb') as log:
 p=subprocess.Popen(['bash',{R!r}+'/scripts/start_supervisor.sh',{a.run_tag!r}],env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 identity=proc(p.pid);save(run/'bootstrap.identity.json',identity)
print(json.dumps(identity))''')
        started=True
        launch_recorded(launch,lambda:state().get('supervisor_identity'),report)
        deadline=time.monotonic()+360;report['samples']=[]
        while time.monotonic()<deadline:
            current=state();report['latest_state']=current
            observations=parallel(lambda h:sample(h,current.get('worker_identity') if h==HOSTS[0] else None))
            if current.get('supervisor_rc') is not None:
                if not real_rcs_pass(current):raise RuntimeError('Real child/recorder/supervisor failure or unknown rc')
                # This sample is after observed real completion; known foreign
                # activity here is post-run evidence, never an inflight conflict.
                report['samples'].append(dict(state=current,observations=observations,decision='post_completion'))
                report['unresolved_resource_observations']=resolve_unknowns(report['samples'],current)
                if report['unresolved_resource_observations']:raise RuntimeError('Unresolved runtime observation; no foreign-cause inference')
                body=call(HOSTS[0],f'''import json,pathlib
print((pathlib.Path({run_path(a.run_tag).as_posix()!r})/'component-results.json').read_text())''')
                (dest/'component-results.json').write_bytes((json.dumps(body,indent=2)+'\n').encode())
                if body.get('worker',{}).get('pid')!=current['worker_identity']['pid'] or body['worker'].get('start_ticks')!=current['worker_identity']['start']:raise RuntimeError('Report not bound to exact launched worker birth')
                report['component_audit']=audit(body,m,compare)
                release_deadline=time.monotonic()+30
                final=observations;released=current
                while True:
                    report['release_gate']=final;report['release_state']=released
                    release=release_decision(final,released);report['release_decision']=release
                    if release in ('released_free','released_post_busy'):
                        if release=='released_post_busy':report.update(post_busy=True,next_start_requires_fresh_gate=True,next_busy_recheck_seconds=1800)
                        break
                    if time.monotonic()>=release_deadline:raise RuntimeError('Own/device release not confirmed')
                    time.sleep(2)
                    released=state();final=parallel(lambda h:sample(h))
                report.update(status='component_passed',passed=True);break
            decision=runtime_decision(observations,current);report['samples'].append(dict(state=current,observations=observations,decision=decision))
            if decision in ('confirmed_unapproved_activity','resource_sample_error'):raise RuntimeError(decision)
            time.sleep(5)
        else:raise TimeoutError('Controller bounded deadline')
    except BaseException as exc:
        report.update(passed=False,status='failed',error=repr(exc),traceback=traceback.format_exc());report.setdefault('first_error',repr(exc))
        if started:
            try:report['cleanup']=stop()
            except Exception as cleanup:report['cleanup_error']=repr(cleanup)
    finally:
        try:report['local_backend_final']=local_backend_identity(m)
        except Exception as exc:report.update(passed=False,local_backend_final_error=repr(exc));report.setdefault('first_error',repr(exc))
        x.close();(dest/'summary.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps(report))
    if not report.get('passed'):raise SystemExit(1)
if __name__=='__main__':main()
