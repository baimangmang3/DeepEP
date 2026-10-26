"""CPU-only deployment/build of exactly four approved standalone candidate files."""
import hashlib
import json
from pathlib import Path
import shlex
from training_native_round import SshBackend,parallel,T,HOSTS

FILES={'native/pointwise.cpp':'d56f8b8be42ca133f76d56faedb7478d43bbaa956b49263745d18847e02947bd',
       'pointwise.py':'d01a65729f61f947daf0b73a0639b12390eca230a8ca43a90c2669f21f0980e1',
       'CMakeLists.txt':'cd992ddacc86d011efcafa10307a7f0d4272685699193fec509ad08ec0c4a318',
       'test_pointwise_npu.py':'2b5e49ad02222946ad20865e371dbcce0a9d4738c5d4834a9a79e6290c293acb'}
SRC=Path('work/training_candidates/p1_pointwise');D=T+'/candidate-p1'
for name,sha in FILES.items():assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,name
manifest=dict(files=FILES,scope='Standalone pointwise candidate only; no adapter/integrated training authorization',P0_unchanged=True)
out=Path('outputs/p1_pointwise_build');out.mkdir(exist_ok=True)
build_script=f'''#!/usr/bin/env bash
set -e
source {T}/activate-qualified-p0.sh
D={D}
set +e
timeout 120 cmake -S "$D" -B "$D/build" > "$D/logs/build.log" 2>&1
RC=$?
if [ "$RC" -eq 0 ]; then
 timeout --signal=TERM --kill-after=30 600 cmake --build "$D/build" --parallel 2 >> "$D/logs/build.log" 2>&1
 RC=$?
fi
printf '%s\\n' "$RC" > "$D/logs/build.rc"
exit "$RC"
'''
backend=SshBackend()
try:
    backend.connect()
    def deploy(host):
        backend.python(host,'p1_cpu_deployment',f'''
import pathlib,hashlib,json
p=pathlib.Path({D!r});expected={FILES!r}
if p.exists() and any(p.iterdir()) and not (p/'source-manifest.json').exists():raise RuntimeError('Unowned existing candidate directory; do not overwrite')
for name,sha in expected.items():
 q=p/name
 if q.exists():assert hashlib.sha256(q.read_bytes()).hexdigest()==sha,'Existing candidate modified: '+name
(p/'native').mkdir(parents=True,exist_ok=True);(p/'logs').mkdir(exist_ok=True)
print(json.dumps({{'candidate':str(p),'cpu_only':True}}))
''')
        with backend.clients[host].open_sftp() as s:
            for name in FILES:
                dest=D+'/'+name;s.put(str(SRC/name),dest+'.tmp');s.posix_rename(dest+'.tmp',dest)
            with s.open(D+'/source-manifest.json','w') as f:f.write(json.dumps(manifest,indent=2))
            with s.open(D+'/logs/build-script.sh','w') as f:f.write(build_script)
        return backend.python(host,'p1_cpu_deployment',f'''
import pathlib,hashlib,json,subprocess
p=pathlib.Path({D!r});expected={FILES!r}
for name,sha in expected.items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
r=subprocess.run(['bash','-n',str(p/'logs/build-script.sh')],capture_output=True,text=True);assert r.returncode==0,r.stderr
print(json.dumps({{'four_files_verified':True,'candidate':str(p),'build_script_bash_n_rc':0}}))
''')
    deployed=parallel(deploy)
    command='if ! mkdir '+shlex.quote(D+'/logs/build-claim')+' 2>/dev/null; then printf \'{"status":"already_claimed"}\\n\'; exit 0; fi\nnohup bash '+shlex.quote(D+'/logs/build-script.sh')+' > '+shlex.quote(D+'/logs/build-supervisor.log')+' 2>&1 < /dev/null &\npid=$!\nprintf \'%s\\n\' "$pid" > '+shlex.quote(D+'/logs/build.pid')+'\nprintf \'{"status":"cpu_build_started","pid":%s}\\n\' "$pid"\n'
    _,stdout,stderr=backend.clients[HOSTS[0]].exec_command(command,timeout=20);data=stdout.read().decode();error=stderr.read().decode();rc=stdout.channel.recv_exit_status();assert rc==0,error
    result=dict(deployment=deployed,build=json.loads(data),build_host=HOSTS[0],source_manifest=manifest,cpu_only=True)
    (out/'deployment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:
    backend.close()
