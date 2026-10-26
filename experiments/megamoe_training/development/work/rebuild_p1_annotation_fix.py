import hashlib,json,shlex
from pathlib import Path
from training_native_round import SshBackend,parallel,T,HOSTS

D=T+'/candidate-p1';old='d56f8b8be42ca133f76d56faedb7478d43bbaa956b49263745d18847e02947bd';new='020745fc4de0d84cd05bb304b250a2c8d6b91569d407219732e556439df4c7b1'
source=Path('work/training_candidates/p1_pointwise/native/pointwise.cpp');assert hashlib.sha256(source.read_bytes()).hexdigest()==new
backend=SshBackend()
try:
    backend.connect()
    def upload(host):
        backend.python(host,'p1_annotation_fix',f'import pathlib,hashlib,json,shutil\np=pathlib.Path({D!r});q=p/"native/pointwise.cpp";s=hashlib.sha256(q.read_bytes()).hexdigest();assert s in ({old!r},{new!r})\nif s=={old!r}:shutil.copy2(q,p/"logs/pointwise.cpp.attempt1")\nprint(json.dumps({{"before_sha":s}}))')
        with backend.clients[host].open_sftp() as s:
            dest=D+'/native/pointwise.cpp';s.put(str(source),dest+'.tmp');s.posix_rename(dest+'.tmp',dest)
        return backend.python(host,'p1_annotation_fix',f'''
import pathlib,hashlib,json
p=pathlib.Path({D!r});m=json.loads((p/'source-manifest.json').read_text());m['files']['native/pointwise.cpp']={new!r}
m['compile_fix']='Pure AIV entry syntax only: core_ratio(0,1) __global__ __aicore__; no math/synchronization/tile/block changes'
for name,sha in m['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
(p/'source-manifest.json').write_text(json.dumps(m,indent=2))
print(json.dumps(m))
''')
    deployed=parallel(upload)
    script=f'''#!/usr/bin/env bash
source {T}/activate-qualified-p0.sh
set +e
timeout --signal=TERM --kill-after=30 600 cmake --build {D}/build --parallel 2 > {D}/logs/build-attempt2.log 2>&1
RC=$?
printf '%s\\n' "$RC" > {D}/logs/build-attempt2.rc
exit "$RC"
'''
    with backend.clients[HOSTS[0]].open_sftp() as s:
        with s.open(D+'/logs/build-attempt2.sh','w') as f:f.write(script)
    command='if ! mkdir '+shlex.quote(D+'/logs/build-attempt2-claim')+' 2>/dev/null; then printf \'{"status":"already_claimed"}\\n\'; exit 0; fi\nnohup bash '+shlex.quote(D+'/logs/build-attempt2.sh')+' > '+shlex.quote(D+'/logs/build-attempt2-supervisor.log')+' 2>&1 < /dev/null &\nprintf \'{"status":"cpu_build_started","pid":%s}\\n\' "$!"\n'
    _,stdout,stderr=backend.clients[HOSTS[0]].exec_command(command,timeout=20);data=stdout.read().decode();error=stderr.read().decode();assert stdout.channel.recv_exit_status()==0,error
    report=dict(deployment=deployed,build=json.loads(data),no_NPU_test_started=True)
    Path('outputs/p1_pointwise_build/attempt2-deployment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
finally:backend.close()
