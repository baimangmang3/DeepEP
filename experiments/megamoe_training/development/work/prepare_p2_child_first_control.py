"""Offline-only control revision; never SSH/deploy/device."""
import pathlib,json,hashlib,difflib,shutil,ast,subprocess
OLD=pathlib.Path('work/training_diagnostics/p2_ep2_serial_control')
NEW=pathlib.Path('work/training_diagnostics/p2_ep2_serial_child_first_control');NEW.mkdir(exist_ok=False)
OUT=pathlib.Path('outputs/p2_ep2_serial_child_first_preparation');OUT.mkdir(exist_ok=False)
for p in OLD.glob('*'):
 if p.is_file():shutil.copyfile(p,NEW/p.name)
shutil.copyfile('work/offline_control_revision/cleanup_policy.py',NEW/'cleanup_policy.py')
controller=(NEW/'controller.py').read_text()
start=controller.index('def stop_owned(');end=controller.index('\ndef cleanup_host27(',start)
replacement='''def stop_owned(backend,host,tag):
 assert host==HOSTS[0], 'Only host27 was launched by this EP2 contract'
 return backend.python(host,tag,f\"\"\"import subprocess,json
p=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',{R!r}+'/scripts/cleanup_own.py','--run-tag',{tag!r}],capture_output=True,text=True,timeout=180)
assert p.returncode==0,p.stdout+p.stderr
print(p.stdout)\"\"\",timeout=190)
'''
controller=controller[:start]+replacement+controller[end:]
controller=controller.replace('validation-p2-ep2-serial-schema2-20261026','validation-p2-ep2-serial-child-first-20261026').replace("outputs/p2_ep2_serial_preparation","outputs/p2_ep2_serial_child_first_preparation")
needle="assert v.returncode==0,v.stdout+v.stderr\n"
controller=controller.replace(needle,needle+"external=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(R/'scripts/external_launcher_gate.py'),'--run-tag',tag],capture_output=True,text=True,timeout=20)\nexternal_report=json.loads(external.stdout);assert external.returncode in (0,12),external.stderr\n")
controller=controller.replace("'passed':not own and not listeners and not prior", "'passed':not own and not listeners and not prior and external.returncode==0,'external_launcher_gate':external_report")
(NEW/'controller.py').write_text(controller,encoding='utf-8',newline='\n')
launcher=(NEW/'run_dispatch_host_trace_checked.sh').read_text().replace('validation-p2-ep2-serial-schema2-20261026','validation-p2-ep2-serial-child-first-20261026')
needle='# Last local gate follows the controller\'s parallel all16 / own-process gate.\n'
launcher=launcher.replace(needle,needle+'timeout 20 python "$DISPATCH_DIAGNOSTIC_ROOT/scripts/external_launcher_gate.py" --run-tag "$RUN_TAG" > "$RUN/logs/foreign-launchers-before-node$NODE_RANK.json"\n')
(NEW/'run_dispatch_host_trace_checked.sh').write_text(launcher,encoding='utf-8',newline='\n')
print(json.dumps({'prepared_local_only':str(NEW),'device_authorized':False}))
