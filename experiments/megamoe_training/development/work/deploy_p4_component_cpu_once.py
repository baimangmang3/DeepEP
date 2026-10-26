"""Reviewed eight-file v2 assembly/default plan only; no Torch/native/device."""
import pathlib,json,hashlib,shlex,datetime
from training_native_round import SshBackend,T,HOSTS
SRC=pathlib.Path('work/training_candidates/p4_wave_compute');M=SRC/'component_validation_manifest.json'
assert hashlib.sha256(M.read_bytes()).hexdigest()=='ace5458f11ffda50a9ea1c4321923d76eae8bb9593d424e2a1b8cec3bf10a1dd'
m=json.loads(M.read_text());inventory=m['assembled_V2_package_source_inventory'];assert len(inventory)==8
for name,v in inventory.items():
 raw=pathlib.Path(v['local_source']).read_bytes();assert hashlib.sha256(raw).hexdigest()==v['sha256'] and len(raw)==v['bytes'],name
assert hashlib.sha256((SRC/'first_device_contract.json').read_bytes()).hexdigest()==m['first_device_contract_sha256']
R=T+'/candidate-p4-component-validation-v2-20261026';PACKAGE=R+'/p4_wave_compute';SO=T+'/candidate-p4-compute-wave-v2-build-20261026/build/libmegamoe_wave_compute.so'
OUT=pathlib.Path('outputs/p4_component_validation_v2_deployment');OUT.mkdir(exist_ok=False)
x=SshBackend();x.connect()
try:
 h=HOSTS[0]
 x.python(h,'p4_component_CPU_setup',f'''import pathlib,json
r=pathlib.Path({R!r});assert not r.exists(),'Refuse overwrite'
for name in ('p4_wave_compute/native','evidence'):r.joinpath(name).mkdir(parents=True,exist_ok=True)
print(json.dumps({{'created':str(r)}}))''')
 with x.clients[h].open_sftp() as ftp:
  for name,v in inventory.items():ftp.put(v['local_source'],PACKAGE+'/'+name)
  ftp.put(str(M),R+'/component_validation_manifest.json');ftp.put(str(SRC/'first_device_contract.json'),R+'/first_device_contract.json')
 verify=x.python(h,'p4_component_CPU_verify',f'''import pathlib,json,hashlib
r=pathlib.Path({R!r});package=r/'p4_wave_compute';checked={{}}
for name,v in {inventory!r}.items():
 raw=(package/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==v['sha256'] and len(raw)==v['bytes'],name
 checked[name]={{'sha256':v['sha256'],'bytes':len(raw)}}
assert hashlib.sha256((r/'component_validation_manifest.json').read_bytes()).hexdigest()=='ace5458f11ffda50a9ea1c4321923d76eae8bb9593d424e2a1b8cec3bf10a1dd'
assert hashlib.sha256((r/'first_device_contract.json').read_bytes()).hexdigest()=={m['first_device_contract_sha256']!r}
so=pathlib.Path({SO!r})
with so.open('rb') as f:so_sha=hashlib.file_digest(f,'sha256').hexdigest()
assert so_sha=={m['actual_compiled_SO_sha256']!r}
print(json.dumps({{'passed':True,'package':str(package),'source_inventory':checked,'SO':str(so),'SO_sha256':so_sha,'Torch_imported':False,'native_loaded':False}}))''')
 (OUT/'source-and-SO-disk-verification.json').write_text(json.dumps(verify,indent=2))
 code=f'''import runpy,sys,pathlib,json,time
entry={PACKAGE!r}+'/validate_component.py';sys.argv=[entry,'--library',{SO!r},'--expected-so-sha256',{m['actual_compiled_SO_sha256']!r},'--output',{R!r}+'/evidence/not-executed-device-report.json']
before={{'Torch_imported':'torch' in sys.modules,'PTA_imported':'torch_npu' in sys.modules,'P4_SO_mapped':{SO!r} in pathlib.Path('/proc/self/maps').read_text()}}
assert not any(before.values())
runpy.run_path(entry,run_name='__main__')
after={{'Torch_imported':'torch' in sys.modules,'PTA_imported':'torch_npu' in sys.modules,'P4_SO_mapped':{SO!r} in pathlib.Path('/proc/self/maps').read_text()}}
assert not any(after.values())
state={{'passed':True,'before':before,'after':after,'argv':sys.argv,'time_ns':time.time_ns(),'scope':'Default plan only; no run-npu/library load/device query/initialization'}}
pathlib.Path({R!r}+'/evidence/default-plan-state.json').write_text(json.dumps(state,indent=2));print(json.dumps(state))
'''
 cmd='/root/miniconda3/envs/zzy/bin/python - <<\'P4_CPU_DEFAULT_PLAN\'\n'+code+'\nP4_CPU_DEFAULT_PLAN\n'
 _,o,e=x.clients[h].exec_command('timeout 30 bash -c '+shlex.quote(cmd),timeout=45);stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status()
 (OUT/'default-plan.stdout').write_text(stdout);(OUT/'default-plan.stderr').write_text(stderr);(OUT/'default-plan-command.json').write_text(json.dumps({'command':cmd,'rc':rc},indent=2))
 assert rc==0,(rc,stderr)
 lines=[json.loads(line) for line in stdout.splitlines() if line.startswith('{')];assert len(lines)==2 and lines[0]['action']=='plan only, no Torch import' and len(lines[0]['configuration']['cases'])==8 and lines[1]['passed']
 (OUT/'default-plan.json').write_text(json.dumps(lines[0],indent=2));(OUT/'default-plan-state.json').write_text(json.dumps(lines[1],indent=2))
 p=pathlib.Path('outputs/training_device_progress.json');s=json.loads(p.read_text(encoding='utf-8-sig'));s['p4_component_CPU_deployment']={'passed':True,'CPU_permission_consumed':True,'package':PACKAGE,'manifest_sha256':'ace5458f11ffda50a9ea1c4321923d76eae8bb9593d424e2a1b8cec3bf10a1dd','plan_rc':0,'plan_cases':8,'Torch_imported':False,'native_loaded':False,'device_authorized':False,'evidence':str(OUT)};s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(s,indent=2))
 print(json.dumps({'CPU_assembly_passed':True,'plan_rc':0,'cases':8,'Torch_imported':False,'native_loaded':False,'device_started':False,'root':PACKAGE}))
finally:x.close()
