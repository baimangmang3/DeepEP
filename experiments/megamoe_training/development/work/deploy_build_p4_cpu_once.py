"""Reviewed isolated P4 CPU compiler only; no Torch/CDLL/device calls."""
import hashlib,json,pathlib,shlex
from training_native_round import SshBackend,T,HOSTS
SRC=pathlib.Path('work/training_candidates/p4_wave_compute');D=T+'/candidate-p4-compute-wave-build-20261026';OUT=pathlib.Path('outputs/p4_compute_wave_build')
manifest_sha='deec8aea2e29c9a365fbbc6ddbc51fe3a64328db4173f5b9b72f97ead3e3f9d9'
assert hashlib.sha256((SRC/'manifest.json').read_bytes()).hexdigest()==manifest_sha
manifest=json.loads((SRC/'manifest.json').read_text());files=manifest['source_inventory']
for name,v in files.items():
 raw=(SRC/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==v['sha256'] and len(raw)==v['bytes'],name
OUT.mkdir(exist_ok=False)
script=f'''#!/usr/bin/env bash
set -eo pipefail
source {T}/activate-qualified-p0.sh
D={D}
CATLASS=/home/zzy/deploy/megamoe-20261024/DeepEP/third-party/catlass
test "$(git -C "$CATLASS" rev-parse HEAD)" = 2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1
test -z "$(git -C "$CATLASS" status --porcelain)"
export MAX_JOBS=2
set +e
nice -n 10 timeout 120 cmake -S "$D" -B "$D/build" -DCATLASS_ROOT="$CATLASS" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" > "$D/logs/configure.log" 2>&1
RC=$?
printf '%s\\n' "$RC" > "$D/logs/configure.rc"
if [ "$RC" -eq 0 ]; then
 nice -n 10 timeout --signal=TERM --kill-after=30 600 cmake --build "$D/build" --parallel 2 --verbose > "$D/logs/build.log" 2>&1
 RC=$?
 printf '%s\\n' "$RC" > "$D/logs/build.rc"
fi
printf '%s\\n' "$RC" > "$D/logs/overall.rc"
if [ "$RC" -eq 0 ]; then
 readelf -h -d -Ws "$D/build/libmegamoe_wave_compute.so" > "$D/logs/readelf.txt" 2>&1
 printf '%s\\n' "$?" > "$D/logs/readelf.rc"
 ldd "$D/build/libmegamoe_wave_compute.so" > "$D/logs/ldd.txt" 2>&1
 printf '%s\\n' "$?" > "$D/logs/ldd.rc"
 sha256sum "$D/build/libmegamoe_wave_compute.so" > "$D/logs/native-SHA256.txt"
fi
exit "$RC"
'''
(OUT/'build-script.sh').write_text(script,encoding='utf-8',newline='\n')
x=SshBackend();x.connect()
try:
 host=HOSTS[0]
 x.python(host,'p4_cpu_source_setup',f'''import pathlib,json
r=pathlib.Path({D!r});assert not r.exists(),'Refuse overwrite'
for name in ('native','tests','logs'):r.joinpath(name).mkdir(parents=True,exist_ok=True)
print(json.dumps({{'created':str(r)}}))''')
 with x.clients[host].open_sftp() as ftp:
  for name in [*files,'manifest.json']:ftp.put(str(SRC/name),D+'/'+name)
  ftp.put(str(OUT/'build-script.sh'),D+'/logs/build-script.sh')
 pre=x.python(host,'p4_cpu_preflight',f'''import pathlib,hashlib,subprocess,json
r=pathlib.Path({D!r});checked={{}}
for name,pin in {files!r}.items():
 b=(r/name).read_bytes();assert len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['sha256'],name;checked[name]=pin
assert hashlib.sha256((r/'manifest.json').read_bytes()).hexdigest()=={manifest_sha!r}
root=pathlib.Path('/home/zzy/deploy/megamoe-20261024/cann-env/Ascend/cann-9.2.0');compiler=(root/'tools/bisheng_compiler/bin/bisheng').resolve(strict=True)
version=subprocess.run([str(compiler),'--version'],capture_output=True,text=True,timeout=15);assert version.returncode==0
catlass=pathlib.Path('/home/zzy/deploy/megamoe-20261024/DeepEP/third-party/catlass')
head=subprocess.run(['git','-C',str(catlass),'rev-parse','HEAD'],capture_output=True,text=True,timeout=15);clean=subprocess.run(['git','-C',str(catlass),'status','--porcelain'],capture_output=True,text=True,timeout=30)
assert head.returncode==clean.returncode==0 and head.stdout.strip()=='2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1' and not clean.stdout
bash=subprocess.run(['bash','-n',str(r/'logs/build-script.sh')],capture_output=True,text=True,timeout=10);assert bash.returncode==0,bash.stderr
print(json.dumps({{'sources':checked,'manifest_sha256':{manifest_sha!r},'compiler':{{'path':str(compiler),'sha256':hashlib.sha256(compiler.read_bytes()).hexdigest(),'version_rc':version.returncode,'version':version.stdout+version.stderr}},'CATLASS_ROOT':str(catlass),'catlass_head':head.stdout.strip(),'catlass_clean':True,'cann_root':str(root),'bash_n_rc':0,'scope':'Source/compile-only. No Torch import, native loading or NPU query/init/test.'}}))''')
 (OUT/'preflight.json').write_text(json.dumps(pre,indent=2));(OUT/'manifest.json').write_bytes((SRC/'manifest.json').read_bytes())
 launched=x.python(host,'p4_cpu_build_start',f'''import subprocess,pathlib,os,json,time
r=pathlib.Path({D!r})
with (r/'logs/supervisor.log').open('xb') as f:
 p=subprocess.Popen(['bash',str(r/'logs/build-script.sh')],start_new_session=True,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
proc=pathlib.Path('/proc')/str(p.pid);s=(proc/'stat').read_text().rsplit(')',1)[1].split()
row={{'pid':p.pid,'pgid':os.getpgid(p.pid),'start_ticks':s[19],'ppid':int(s[1]),'argv':['bash',str(r/'logs/build-script.sh')],'time_ns':time.time_ns(),'scope':'CPU compiler only'}}
(r/'logs/build-start.json').write_text(json.dumps(row));print(json.dumps(row))''')
 (OUT/'deployment.json').write_text(json.dumps(dict(launched,host=host,directory=D,source_manifest_sha256=manifest_sha,build_script_sha256=hashlib.sha256((OUT/'build-script.sh').read_bytes()).hexdigest()),indent=2));print(json.dumps(launched))
finally:x.close()
