"""Single worker/recorder supervision primitives. Stdlib, no Torch/device API."""
import hashlib,json,os,pathlib,signal,subprocess,time
T='/home/zzy/deploy/megamoe-20261024/training'
R=T+'/validation-p4-component-v2-control-20261026'
PACKAGE=T+'/candidate-p4-component-validation-v2-20261026/p4_wave_compute'
LIB=T+'/candidate-p4-compute-wave-v2-build-20261026/build/libmegamoe_wave_compute.so'
SO='eea6c53d1a9ffd7d06aed510b9d2f80e59a9112476008a3796f25ab57fc797ac'
PYTHON=T+'/venv/bin/python'
STAGE='p4-v2-component-small'
def run_path(tag):
    if not tag or not all(c.isalnum() or c in '_-' for c in tag):raise ValueError('Invalid RUN')
    return pathlib.Path(T)/'runs'/tag/STAGE
def save(path,value):
    path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf8'));os.replace(tmp,path)
def proc(pid):
    p=pathlib.Path('/proc')/str(pid);st=(p/'stat').read_text().rsplit(')',1)[1].split()
    env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
    return dict(pid=pid,start=st[19],state=st[0],ppid=int(st[1]),pgid=int(st[2]),uid=p.stat().st_uid,
        argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v],
        run_tag=env.get(b'RUN_TAG',b'').decode(),role=env.get(b'P4_ROLE',b'').decode(),visible=env.get(b'ASCEND_RT_VISIBLE_DEVICES',b'').decode())
def exact(a,b):return all(a.get(k)==b.get(k) for k in ('pid','start','uid','argv','run_tag','role','visible')) and a.get('visible')=='0'
def alive(saved):
    try:current=proc(saved['pid'])
    except FileNotFoundError:return False
    if current['start']!=saved['start']:return False # original gone; never signal reused PID
    if not exact(saved,current):raise RuntimeError('Live own identity changed; refuse signal')
    return current['state'] not in ('Z','X')
def signal_exact(saved,sig):
    if alive(saved):os.kill(saved['pid'],sig);return True
    return False
def stop_exact(saved):
    events=[]
    for sig,seconds in ((signal.SIGTERM,30),(signal.SIGKILL,10)):
        events.append(dict(signal=int(sig),sent=signal_exact(saved,sig),time_ns=time.time_ns()))
        end=time.monotonic()+seconds
        while alive(saved) and time.monotonic()<end:time.sleep(.2)
        if not alive(saved):return dict(gone=True,events=events)
    return dict(gone=False,events=events)
def wait_worker(child,stop,timeout=180):
    """Actual waitpid result only; timeout/cleanup never invents an exit code."""
    report=dict(worker_rc=None,timeout=False,cleanup_errors=[])
    try:report['worker_rc']=child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        report.update(timeout=True,first_error='worker_timeout')
        try:report['stop']=stop()
        except Exception as exc:report['cleanup_errors'].append(repr(exc))
        try:report['worker_rc']=child.wait(timeout=10)
        except Exception as exc:report['cleanup_errors'].append(repr(exc))
    except Exception as exc:
        report['first_error']=repr(exc)
        try:report['stop']=stop()
        except Exception as cleanup:report['cleanup_errors'].append(repr(cleanup))
    if report['worker_rc'] not in (None,0):report.setdefault('first_error','worker_nonzero_rc')
    report['passed']=report['worker_rc']==0 and not report['timeout'] and not report['cleanup_errors'] and not report.get('first_error')
    return report
def spawn_owned(argv,env,log,run,tag,popen=subprocess.Popen,read_identity=proc,write=save):
    """If marker capture fails, only the unreaped Popen child is signal authority."""
    if env.get('RUN_TAG')!=tag or env.get('visible',env.get('ASCEND_RT_VISIBLE_DEVICES'))!='0' or env.get('P4_ROLE')!='worker':raise ValueError('Worker env differs')
    if (run/'abort.json').exists():raise RuntimeError('RUN aborted before worker spawn')
    write(run/'worker.launch-intent.json',dict(run_tag=tag,argv=argv,role='worker'))
    child=None;failure={}
    try:
        child=popen(argv,env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        failure=dict(run_tag=tag,owned_child_pid=child.pid,argv=argv,authority='direct unreaped Popen child only',owned_child_reaped=False,worker_rc=None)
        saved=read_identity(child.pid)
        if saved['argv']!=argv or saved['run_tag']!=tag or saved['visible']!='0' or saved['role']!='worker':raise RuntimeError('Worker identity differs after exec')
        write(run/'worker.identity.json',saved);return child,saved
    except BaseException as original:
        failure.update(run_tag=tag,error=repr(original),owned_child_created=child is not None)
        if child is not None:
            # No os.kill with an unvalidated PID; Popen retains direct-child waitpid ownership.
            for operation,seconds in ((child.terminate,30),(child.kill,10)):
                try:
                    if child.poll() is None:operation()
                    failure['worker_rc']=child.wait(timeout=seconds);failure['owned_child_reaped']=True;break
                except BaseException as cleanup:failure.setdefault('cleanup_errors',[]).append(repr(cleanup))
        try:write(run/'startup-failure.json',failure)
        except BaseException:pass # preserve original error; absent proof remains unknown
        raise
def worker_argv(tag):return [PYTHON,PACKAGE+'/validate_component.py','--run-npu','--library',LIB,'--expected-so-sha256',SO,
    '--device','0','--seed','4317','--cap-bytes',str(12<<30),'--output',(run_path(tag)/'component-results.json').as_posix()]
def read(path):return json.loads(pathlib.Path(path).read_bytes())
def claim_check(tag):
    p=run_path(tag);a=read(p/'authorization.json')
    manifest_path=pathlib.Path(R)/'manifest.json';m=read(manifest_path)
    if (a.get('run_tag')!=tag or a.get('device_authorized') is not True or
        a.get('manifest_sha256')!=hashlib.sha256(manifest_path.read_bytes()).hexdigest() or
        a.get('controller_sha256')!=m['files']['scripts/controller.py']['sha256'] or
        a.get('component_contract_sha256')!=m['component_contract_sha256'] or a.get('SO')!=SO):raise RuntimeError('No exact RUN/source/SO authorization')
    return p
def cleanup_worker(tag,parents=False):
    run=run_path(tag);report=dict(run_tag=tag,events=[],errors=[])
    save(run/'abort.json',dict(run_tag=tag,reason='explicit own cleanup; revoke any unstarted worker'))
    worker=run/'worker.identity.json'
    if worker.exists():
        try:report['worker_stop']=stop_exact(read(worker))
        except Exception as exc:report['errors'].append(repr(exc))
    else:report['errors'].append('No exact worker marker: no guessed signal')
    confirmed=report.get('worker_stop',{}).get('gone') is True
    for file in ('startup-failure.json','worker-not-started.json'):
        if not confirmed and not worker.exists() and (run/file).exists():
            proof=read(run/file)
            confirmed=proof.get('run_tag')==tag and (proof.get('owned_child_created') is False or
                proof.get('owned_child_reaped') is True and type(proof.get('worker_rc')) is int or proof.get('worker_never_started') is True)
    report['worker_exit_or_nonstart_confirmed']=confirmed
    if parents:
        if not confirmed:
            report['parents_preserved']='Worker exit/nonstart unknown; do not terminate recording parents'
            save(run/'cleanup.json',report);return report
        # Worker first; recorders get40s to waitpid and save true child statuses.
        end=time.monotonic()+40
        while not (run/'supervisor.rc').exists() and time.monotonic()<end:time.sleep(.2)
        for role in ('recorder','supervisor'):
            marker=run/(role+'.identity.json')
            if marker.exists():
                try:report['events'].append(dict(role=role,stop=stop_exact(read(marker))))
                except Exception as exc:report['errors'].append(repr(exc))
    save(run/'cleanup.json',report);return report
