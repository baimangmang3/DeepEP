"""One-shot CPU hash -> parallel two-host gate -> immediate native supervisor.

No polling, reservation or device initialization in this controller. Only
--execute contacts hosts. Passwords are supplied through process environment.
The native helper repeats its local gate and uses bounded own process groups.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

HOSTS = ('177.22.27.1', '177.22.28.1')
T = '/home/zzy/deploy/megamoe-20261024/training'
PYTHON = '/root/miniconda3/envs/zzy/bin/python'
EXPECTED_IDENTITY = '04c7371f9d9c78623e54ad823c224dddffbb6f9e63b35c665f3f379e10a0fba6'


def parallel(function):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {host:pool.submit(function, host) for host in HOSTS}
        return {host:future.result() for host,future in futures.items()}


class SshBackend:
    def __init__(self):
        self.clients = {}
        self.helper_hashes = {
            T+'/scripts/'+name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('quick_training_identity.py', 'query_training_cores.py', 'run_native_training_checked.sh')}

    def connect(self):
        import paramiko
        def open_host(host):
            key = 'DEPLOY_PASSWORD_27' if host == HOSTS[0] else 'DEPLOY_PASSWORD_28'
            password = os.environ[key]
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.connect(host, username='root', password=password, timeout=15, banner_timeout=15, auth_timeout=15)
            self.clients[host] = client
        parallel(open_host)

    def close(self):
        for client in self.clients.values():
            client.close()

    def python(self, host, tag, payload, timeout=140):
        command = ('RUN_TAG='+shlex.quote(tag)+' timeout 125 '+PYTHON+" - <<'TRAINING_CONTROL_PY'\n"
                   +payload+'\nTRAINING_CONTROL_PY\n')
        _,out,err = self.clients[host].exec_command(command, timeout=timeout)
        stdout,stderr = out.read().decode(errors='replace'),err.read().decode(errors='replace')
        rc = out.channel.recv_exit_status()
        if rc:
            raise RuntimeError(f'{host} control rc={rc}: {stderr[-4000:]} {stdout[-2000:]}')
        return json.loads(stdout)

    def verify(self, host, tag):
        expected = {path:sha for path,sha in self.helper_hashes.items()
                    if host == HOSTS[0] or not path.endswith('/run_native_training_checked.sh')}
        return self.python(host, tag, f'''
import pathlib,hashlib,runpy,contextlib,io,json
expected={expected!r}
for path,sha in expected.items():
 assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==sha,path
capture=io.StringIO()
with contextlib.redirect_stdout(capture):runpy.run_path({T!r}+'/scripts/quick_training_identity.py')
report=json.loads(capture.getvalue())
assert report['passed'] is True
report['verified_helpers']=expected
print(json.dumps(report))
''')

    def gate(self, host, tag):
        return self.python(host, tag, f'''
import pathlib,runpy,contextlib,io,json,os
t=pathlib.Path({T!r});capture=io.StringIO()
with contextlib.redirect_stdout(capture):runpy.run_path(str(t/'scripts/training_occupancy_gate.py'))
prefix=pathlib.Path(capture.getvalue().split('EVIDENCE_PREFIX=')[-1].strip())
capture=io.StringIO()
with contextlib.redirect_stdout(capture):runpy.run_path(str(t/'scripts/training_npu_process_owners.py'))
owners_path=pathlib.Path(capture.getvalue().split('OWNERS_EVIDENCE=')[-1].strip())
observation=json.loads(prefix.with_suffix('.json').read_text())
owners=json.loads(owners_path.read_text())
stage=t/'runs'/os.environ['RUN_TAG']/'native-gemm'
prior={{'marker':(stage/'launch-marker').exists(),'rc':None,'passed':False,'results_exists':(stage/'results.json').exists()}}
if (stage/'rc').is_file():prior['rc']=(stage/'rc').read_text().strip()
if (stage/'results.json').is_file():prior['passed']=json.loads((stage/'results.json').read_text()).get('passed') is True
print(json.dumps({{'observation':observation,'owners':owners,'npu_smi':prefix.with_suffix('.txt').read_text(),'prior_native':prior}}))
''')

    def launch(self, host, tag):
        # Reuse the same connected client; no model/SSH handshake between gate and launch.
        stage = T+'/runs/'+tag+'/native-gemm'
        command = f'''set -e
mkdir -p {shlex.quote(stage)}
if ! mkdir {shlex.quote(stage+'/launch-marker')} 2>/dev/null; then
 printf '%s\\n' '{{"status":"already_claimed"}}'; exit 0
fi
RUN_TAG={shlex.quote(tag)} nohup timeout --signal=TERM --kill-after=30 360 bash \
 {shlex.quote(T+'/scripts/run_native_training_checked.sh')} \
 > {shlex.quote(stage+'/supervisor.log')} 2>&1 < /dev/null &
native_pid=$!
printf '%s\\n' "$native_pid" > {shlex.quote(stage+'/supervisor.pid')}
printf '{{"status":"native_supervisor_started","pid":%s}}\\n' "$native_pid"
'''
        _,out,err = self.clients[host].exec_command(command, timeout=15)
        stdout,stderr = out.read().decode(),err.read().decode()
        rc = out.channel.recv_exit_status()
        if rc:
            raise RuntimeError(f'native supervisor launch rc={rc}: {stderr}')
        result = json.loads(stdout)
        result.update(host=host, run_tag=tag, stage_path=stage,
                      timeout_seconds=360, kill_after_seconds=30,
                      local_gate_repeated=True, device_execution_confirmed=False)
        return result


def run_round(backend, tag, directory):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        raise ValueError('Invalid RUN_TAG')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = {'run_tag':tag, 'npu_initialized_by_controller':False, 'native_started':False}
    def save(name, value):
        (directory/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    try:
        backend.connect()
        # All CPU checks complete before any device occupancy observation.
        checks = parallel(lambda host:backend.verify(host, tag))
        if not all(row.get('passed') is True for row in checks.values()):
            raise ValueError('CPU identity failed')
        if any(row.get('cached_content_digest') != EXPECTED_IDENTITY for row in checks.values()):
            raise ValueError('Cached identity differs from the frozen review qualification')
        save('quick-identities.json', checks)
        gates = parallel(lambda host:backend.gate(host, tag))
        gate_finished = time.monotonic()
        save('gates.json', gates)
        occupied = any(not gate['observation'].get('all_eight_free') or
                       gate['observation'].get('npu_smi_rc') != 0 or
                       gate['owners'].get('processes') or
                       any(row.get('own') for row in gate['observation'].get('processes', []))
                       for gate in gates.values())
        if occupied:
            report['status'] = 'occupied_or_uncertain_deferred'
        elif any(gate['prior_native'].get('marker') or gate['prior_native'].get('rc') is not None or
                 gate['prior_native'].get('passed') or gate['prior_native'].get('results_exists') for gate in gates.values()):
            report['status'] = 'prior_native_found_no_duplicate_launch'
        else:
            # Intentionally no file transfer, user interaction or CPU validation here.
            report['gate_to_launch_call_ms'] = 1000*(time.monotonic()-gate_finished)
            report['launch'] = backend.launch(HOSTS[0], tag)
            report['native_started'] = report['launch']['status'] == 'native_supervisor_started'
            report['status'] = report['launch']['status']
        report['gate_summary'] = {host:{'time_utc':gate['observation']['time_utc'],
            'free_cards':gate['observation']['free_cards'],
            'own_device_processes':gate['owners']['own_device_processes'],
            'foreign_device_pids':[row['pid'] for row in gate['owners']['foreign_device_processes']]}
            for host,gate in gates.items()}
    except Exception as error:
        report.update(status='control_error_no_further_launch', error=repr(error))
    finally:
        backend.close()
        save('summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Connect, CPU-hash, gate and conditionally launch once')
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        parser.error('No action taken: --execute is required; use test_training_native_round.py for CPU simulation')
    result = run_round(SshBackend(), args.run_tag, args.output or Path('outputs')/('training_round_'+args.run_tag))
    print(json.dumps(result), flush=True)
    return 2 if result['status']=='control_error_no_further_launch' else 0


if __name__ == '__main__':
    raise SystemExit(main())
