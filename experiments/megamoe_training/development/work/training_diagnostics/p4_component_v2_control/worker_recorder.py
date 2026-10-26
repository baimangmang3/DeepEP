"""Direct worker recorder; no torchrun or PG. No-execute prints CPU plan."""
import argparse,os,pathlib,subprocess,sys,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from process_control import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--run-tag');a=p.parse_args()
    if not a.execute:print(json.dumps(dict(role='worker_recorder',worker_timeout=180,action='plan only')));return 0
    run=claim_check(a.run_tag);save(run/'recorder.identity.json',proc(os.getpid()))
    report=dict(passed=False,worker_rc=None,argv=worker_argv(a.run_tag));child=None
    try:
        env=dict(os.environ,P4_ROLE='worker',ASCEND_RT_VISIBLE_DEVICES='0')
        with (run/'worker.log').open('xb') as log:
            child,saved=spawn_owned(report['argv'],env,log,run,a.run_tag)
            report.update(wait_worker(child,lambda:stop_exact(saved)))
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc());report.setdefault('first_error',repr(exc))
        if (run/'startup-failure.json').exists():
            report['startup_failure']=read(run/'startup-failure.json')
            if type(report['startup_failure'].get('worker_rc')) is int:report['worker_rc']=report['startup_failure']['worker_rc']
        # Only a saved exact marker is signal authority. Unknown startup stays failed.
        if (run/'worker.identity.json').exists():
            try:report['startup_cleanup']=cleanup_worker(a.run_tag)
            except Exception as cleanup:report['cleanup_error']=repr(cleanup)
        if child is not None:
            try:report['worker_rc']=child.wait(timeout=10)
            except Exception as wait_error:report['wait_error']=repr(wait_error)
    finally:save(run/'worker-status.json',report)
    return 0 if report.get('passed') and not report.get('error') else 1
if __name__=='__main__':sys.exit(main())
