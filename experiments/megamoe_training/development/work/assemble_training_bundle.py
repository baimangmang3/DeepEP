import pathlib,tarfile,zipfile,hashlib,json
base=pathlib.Path('work/training_bundle_final');base.mkdir(exist_ok=True)
snapshot=pathlib.Path('outputs/training_source_snapshot.zip')
assert hashlib.sha256(snapshot.read_bytes()).hexdigest()=='d7df9794d10b8aa92a016b396d6db2ff1f85752f2ebce546a095191e3db801ba'
with zipfile.ZipFile(snapshot) as z:z.extractall(base/'DeepEP')
manifest=json.loads(pathlib.Path('outputs/training_source_manifest.json').read_text())
for row in manifest['files']:
 assert hashlib.sha256((base/'DeepEP'/row['path']).read_bytes()).hexdigest()==row['sha256']
for host in (7,8):
 d=base/'hosts'/f'rack05os{host}';d.mkdir(parents=True,exist_ok=True)
 with tarfile.open(f'work/training_native/rack05os{host}_logs.tar.gz') as a:
  a.extractall(d,members=[m for m in a.getmembers() if not m.name.startswith('logs/cann-selector')],filter='data')
 logs=d/'logs';validation=json.loads((logs/'cpu-deployment-validation.json').read_text())
 assert validation['passed'] and not validation['npu_tests_executed'] and not validation['npu_initialized_after']
 assert validation['source_files_verified']==13 and validation['python_files_compiled']==8
 assert json.loads((logs/'own-device-processes.json').read_text())==[]
 assert (logs/'core-wheel-install.rc').read_text().strip()=='0'
 assert (logs/'cpu-deployment-validation.rc').read_text().strip()=='0'
 assert (logs/'cpu-final-syntax.rc').read_text().strip()=='0'
 assert 'not found' not in (logs/'native-final-ldd.txt').read_text()
 assert (logs/'inference-source-status.txt').stat().st_size==0
assert (base/'hosts/rack05os7/logs/native-build.rc').read_text().strip()=='0'
import shutil
shutil.copyfile('outputs/training_cann_selector_evidence.tar.gz',base/'CANN_SELECTOR_EVIDENCE.tar.gz')
lib=pathlib.Path('work/training_native/libmegamoe_training.so')
assert hashlib.sha256(lib.read_bytes()).hexdigest()==manifest['native_library_sha256']
dest=base/'DeepEP/experiments/megamoe_training/build/libmegamoe_training.so';dest.parent.mkdir(exist_ok=True);shutil.copyfile(lib,dest)
for source,name in [('outputs/training_source_manifest.json','SOURCE_MANIFEST.json'),('outputs/training_environment_status.md','STATUS.md'),('outputs/training_runbook.md','RUNBOOK.md'),('outputs/training_occupancy_observation.json','OCCUPANCY_OBSERVATION.json')]:shutil.copyfile(source,base/name)
(base/'deployment_scripts').mkdir(exist_ok=True)
for name in ('run_training_checked.sh','probe_cann_aiv.py','validate_training_cpu.py','activate_training.sh'):shutil.copyfile(pathlib.Path('work')/name,base/'deployment_scripts'/name)
identity={str(p.relative_to(base)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob('*') if p.is_file() and p.name!='ARCHIVE_MANIFEST.json'}
(base/'ARCHIVE_MANIFEST.json').write_text(json.dumps(identity,indent=2))
out=pathlib.Path('outputs/training_environment_bundle.tar.gz')
with tarfile.open(out,'w:gz') as tar:
 for p in sorted(base.iterdir()):tar.add(p,arcname=p.name)
print(json.dumps(dict(bundle=str(out),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),files=len(identity),cpu_checks='both hosts passed; no NPU initialized or test process',native_build_rc=0),ensure_ascii=False))
