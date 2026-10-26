"""Approved standalone P3 CPU compile only: no CDLL, import Torch or device API."""
import hashlib,json,pathlib,shlex
from training_native_round import SshBackend,T,HOSTS
SRC=pathlib.Path('work/training_candidates/p3_up_swiglu');D=T+'/candidate-p3-up-swiglu-20261025'
OUT=pathlib.Path('outputs/p3_up_swiglu_build');OUT.mkdir(exist_ok=True)
manifest_sha='1e2f1362663bf3d1726999d0c3df85d8b23135f2e0e951199cb9af36c4cbf612'
assert hashlib.sha256((SRC/'manifest.json').read_bytes()).hexdigest()==manifest_sha
manifest=json.loads((SRC/'manifest.json').read_text());files=manifest['source_sha256']
for n,h in files.items():assert hashlib.sha256((SRC/n).read_bytes()).hexdigest()==h,n
script=f'''#!/usr/bin/env bash
set -eo pipefail
source {T}/activate-qualified-p0.sh
D={D}
CATLASS=/home/zzy/deploy/megamoe-20261024/DeepEP/third-party/catlass
test "$(git -C "$CATLASS" rev-parse HEAD)" = 2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1
test -z "$(git -C "$CATLASS" status --porcelain)"
export MAX_JOBS=2
set +e
timeout 120 cmake -S "$D" -B "$D/build" -DCATLASS_ROOT="$CATLASS" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" > "$D/logs/build.log" 2>&1
RC=$?
echo "$RC" > "$D/logs/configure.rc"
if [ "$RC" -eq 0 ]; then
 timeout --signal=TERM --kill-after=30 600 cmake --build "$D/build" --parallel 2 --verbose >> "$D/logs/build.log" 2>&1
 RC=$?
fi
echo "$RC" > "$D/logs/build.rc"
if [ "$RC" -eq 0 ]; then
 readelf -h -d -Ws "$D/build/libmegamoe_up_swiglu.so" > "$D/logs/readelf.txt" 2>&1
 echo "$?" > "$D/logs/readelf.rc"
 ldd "$D/build/libmegamoe_up_swiglu.so" > "$D/logs/ldd.txt" 2>&1
 echo "$?" > "$D/logs/ldd.rc"
fi
exit "$RC"
'''
(OUT/'build-script.sh').write_text(script,newline='\n')
x=SshBackend()
try:
 x.connect();host=HOSTS[0]
 x.python(host,'p3_cpu_prepare',f'''import pathlib,json
p=pathlib.Path({D!r});assert not p.exists(),'Independent candidate path already exists'
(p/'native').mkdir(parents=True);(p/'logs').mkdir()
print(json.dumps({{'directory':str(p),'device_calls':False}}))''')
 with x.clients[host].open_sftp() as s:
  for n in [*files,'manifest.json']:s.put(str(SRC/n),D+'/'+n)
  s.put(str(OUT/'build-script.sh'),D+'/logs/build-script.sh')
 pre=x.python(host,'p3_cpu_prepare',f'''import pathlib,hashlib,subprocess,json
p=pathlib.Path({D!r});expected={files!r};rows={{}}
for n,h in expected.items():
 actual=hashlib.sha256((p/n).read_bytes()).hexdigest();assert actual==h;rows[n]=actual
assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()=={manifest_sha!r}
root=pathlib.Path('/home/zzy/deploy/megamoe-20261024/cann-env/Ascend/cann-9.2.0')
compiler=(root/'tools/bisheng_compiler/bin/bisheng').resolve();version=subprocess.run([str(compiler),'--version'],capture_output=True,text=True,timeout=15)
check=subprocess.run(['bash','-n',str(p/'logs/build-script.sh')],capture_output=True,text=True);assert check.returncode==0,check.stderr
print(json.dumps({{'source_sha256':rows,'manifest_sha256':{manifest_sha!r},'compiler':{{'path':str(compiler),'sha256':hashlib.sha256(compiler.read_bytes()).hexdigest(),'version_rc':version.returncode,'version':version.stdout+version.stderr}},'catlass_root':'/home/zzy/deploy/megamoe-20261024/DeepEP/third-party/catlass','catlass_head':'2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1','catlass_clean':True,'cann_root':str(root),'bash_n_rc':0,'scope':'CPU compiler/ELF only; no CDLL/Torch/device/test'}}))''')
 (OUT/'preflight.json').write_text(json.dumps(pre,indent=2));(OUT/'manifest.json').write_bytes((SRC/'manifest.json').read_bytes())
 cmd='nohup nice -n 10 bash '+shlex.quote(D+'/logs/build-script.sh')+' > '+shlex.quote(D+'/logs/supervisor.log')+' 2>&1 < /dev/null &\npid=$!\nprintf \'%s\\n\' "$pid" > '+shlex.quote(D+'/logs/build.pid')+'\nprintf \'{"status":"cpu_build_started","pid":%s}\\n\' "$pid"\n'
 _,o,e=x.clients[host].exec_command(cmd,timeout=15);r=json.loads(o.read());assert o.channel.recv_exit_status()==0,e.read().decode();r.update(host=host,directory=D,source_manifest_sha256=manifest_sha,build_script_sha256=hashlib.sha256((OUT/'build-script.sh').read_bytes()).hexdigest());(OUT/'deployment.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
finally:x.close()
