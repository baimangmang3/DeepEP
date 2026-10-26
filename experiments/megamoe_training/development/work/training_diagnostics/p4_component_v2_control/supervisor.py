"""Parent of worker recorder; retains real recorder waitpid rc. CPU plan default."""
import argparse,os,pathlib,subprocess,sys,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from process_control import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--run-tag');a=p.parse_args()
    if not a.execute:print(json.dumps(dict(role='supervisor',action='plan only')));return 0
    run=claim_check(a.run_tag);save(run/'supervisor.identity.json',proc(os.getpid()))
    report=dict(passed=False,recorder_rc=None);child=None
    try:
        smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
        (run/'npu-before.txt').write_text(smi.stdout+smi.stderr)
        if smi.returncode or smi.stdout.count('No running processes')!=8:raise RuntimeError('Final local card gate busy/unknown')
        if (run/'abort.json').exists():raise RuntimeError('Aborted before recorder spawn')
        env=dict(os.environ,P4_ROLE='recorder')
        with (run/'recorder.log').open('xb') as log:
            child=subprocess.Popen([PYTHON,R+'/scripts/worker_recorder.py','--execute','--run-tag',a.run_tag],env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            report['recorder_rc']=child.wait(timeout=260)
        save(run/'recorder.rc',report['recorder_rc'])
        report['passed']=report['recorder_rc']==0
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc());report.setdefault('first_error',repr(exc))
        if child is None:save(run/'worker-not-started.json',dict(run_tag=a.run_tag,worker_never_started=True,reason=repr(exc)))
        try:report['cleanup']=cleanup_worker(a.run_tag) # worker before recorder, never foreign PID guessing
        except Exception as cleanup:report['cleanup_error']=repr(cleanup)
        if child is not None:
            try:report['recorder_rc']=child.wait(timeout=40);save(run/'recorder.rc',report['recorder_rc'])
            except Exception as wait_error:report['wait_error']=repr(wait_error)
    finally:save(run/'supervisor-status.json',report)
    return 0 if report['passed'] and not report.get('error') else 1
if __name__=='__main__':sys.exit(main())
