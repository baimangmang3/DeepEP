"""Standalone watchdog: recorder stays outside the ACL child's process group."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--run-tag', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--probe', default='/home/zzy/deploy/megamoe-20261024/training/scripts/acl_initialization_probe.py')
    p.add_argument('--report-name', choices=('up-swiglu-results.json',), default='up-swiglu-results.json')
    p.add_argument('--up-library',required=True)
    p.add_argument('--p0-source-root',required=True)
    p.add_argument('--catlass-library',required=True)
    p.add_argument('--timeout-seconds',type=int,choices=(120,600),default=120)
    a = p.parse_args()
    if not a.execute:
        p.error('--execute and controller dual-host idle gate required')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Atomic once-only stage claim; never start a duplicate for this output path.
    (out / 'supervisor-claim').mkdir()
    result = dict(run_tag=a.run_tag, supervisor_pid=os.getpid(), child_pid=None,
                  timeout=False, returncode=None, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

    def save():
        temporary = out / 'supervisor-result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(out / 'supervisor-result.json')

    save()
    try:
        gate = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
    except Exception as error:
        result.update(status='local_gate_error', error=repr(error))
        save()
        return 10
    (out / 'local-gate.txt').write_text(gate.stdout + gate.stderr)
    if gate.returncode or gate.stdout.count('No running processes') != 8:
        result.update(status='local_gate_blocked', gate_rc=gate.returncode)
        save()
        return 10
    env = dict(os.environ, RUN_TAG=a.run_tag)
    with (out / 'child.log').open('w') as log:
        argv=[sys.executable,a.probe,'--execute','--output',str(out/a.report_name)]
        argv=[sys.executable,a.probe,'--run-npu','--library',a.up_library,'--p0-source-root',a.p0_source_root,'--catlass-library',a.catlass_library,'--output',str(out/a.report_name),'--device','0']
        result['child_argv']=argv;result['timeout_seconds']=a.timeout_seconds;save()
        child = subprocess.Popen(argv,
                                 env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result.update(child_pid=child.pid, child_pgid=child.pid, status='running')
        save()
        try:
            child.wait(timeout=a.timeout_seconds)
        except subprocess.TimeoutExpired:
            result.update(timeout=True, status='timeout_term')
            save()
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                result['status'] = 'timeout_kill'
                save()
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    result['status'] = 'unreaped_after_kill'
        result['returncode'] = child.returncode
        if child.returncode is not None:
            result['status'] = 'reaped'
        result['own_child_proc_exists'] = Path(f'/proc/{child.pid}').exists()
        if result['own_child_proc_exists']:
            for name in ('stat', 'wchan', 'cmdline'):
                try:
                    result[name] = Path(f'/proc/{child.pid}/{name}').read_text().replace('\0', ' ')
                except OSError as e:
                    result[name] = str(e)
        save()
    return 0 if child.returncode == 0 and not result['timeout'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
