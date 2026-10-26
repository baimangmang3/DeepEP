import pathlib,json,hashlib,shutil,ast,difflib,subprocess
OLD=pathlib.Path('work/training_diagnostics/p2_ep2_serial_control');NEW=OLD.with_name('p2_ep2_serial_child_first_control');OUT=pathlib.Path('outputs/p2_ep2_serial_child_first_preparation')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for a,b in [('work/p2_cleanup_own_template.py','cleanup_own.py'),('work/p2_external_launcher_gate_template.py','external_launcher_gate.py'),('work/offline_control_revision/cleanup_policy.py','cleanup_policy.py')]:shutil.copyfile(a,NEW/b)
p=NEW/'controller.py';s=p.read_text();start=s.index('def stop_owned(');end=s.index('\ndef cleanup_host27(',start)
s=s[:start]+'''def stop_owned(backend,host,tag):
 assert host==HOSTS[0], 'Only host27 was launched by this EP2 contract'
 cmd='RUN_TAG='+shlex.quote(tag)+' timeout 180 /root/miniconda3/envs/zzy/bin/python '+shlex.quote(R+'/scripts/cleanup_own.py')+' --run-tag '+shlex.quote(tag)
 _,out,err=backend.clients[host].exec_command(cmd,timeout=190)
 stdout,stderr=out.read().decode(),err.read().decode();rc=out.channel.recv_exit_status()
 assert rc==0,repr({'cleanup_rc':rc,'stdout':stdout,'stderr':stderr})
 return json.loads(stdout)
'''+s[end:]
s=s.replace("state.get('p2_ep2_serial_control',{})","state.get('p2_ep2_serial_child_first_control',{})").replace("d['p2_ep2_serial_control']","d['p2_ep2_serial_child_first_control']")
p.write_text(s,encoding='utf-8',newline='\n')
test=pathlib.Path('work/test_p2_serial_cleanup_cpu.py').read_text().replace('p2_ep2_serial_control','p2_ep2_serial_child_first_control').replace('p2_ep2_serial_preparation','p2_ep2_serial_child_first_preparation')
pathlib.Path('work/test_p2_child_first_cleanup_cpu.py').write_text(test,encoding='utf-8')
for p in NEW.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
result=subprocess.run(['C:/Program Files/Git/bin/bash.exe','-n',str(NEW/'run_dispatch_host_trace_checked.sh')],capture_output=True,text=True)
assert result.returncode==0,result.stderr
save=lambda p,v:pathlib.Path(p).write_text(json.dumps(v,indent=2),encoding='utf-8')
for host in ('rack05os7','rack05os8'):
 old=json.loads(pathlib.Path(f'outputs/p2_ep2_serial_preparation/{host}-diagnostic-deployment-manifest.json').read_text())
 new=json.loads(json.dumps(old));new['schema']='p2-ep2-serial-child-first-controls-CPU-only-2';new['device_authorized']=False
 new['control_revision']={'scope':'Offline only, not deployed','inner_first_cleanup':True,'recorder_harvest_seconds':40,'cleanup_parent_harvest_order':['node_recorder','outer_recorder'],'external_launcher_gate':'recognizable foreign torchrun even when no NPU PIDs','DNS_wrapper':'not integrated; original torch.distributed.run command unchanged','source_or_math_changed':False}
 for filename in [v.name for v in NEW.iterdir() if v.is_file() and v.name!='controller.py']:
  new['files']['scripts/'+filename]={'sha256':sha(NEW/filename),'bytes':(NEW/filename).stat().st_size}
 assert new['source_files']==old['source_files'] and new['external_native']==old['external_native']
 save(OUT/f'{host}-diagnostic-deployment-manifest.json',new)
patch=''
for name in ('controller.py','run_dispatch_host_trace_checked.sh'):
 patch+=''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True),(NEW/name).read_text().splitlines(True),fromfile='old/'+name,tofile='child-first/'+name))
(OUT/'minimal-control-diff.patch').write_text(patch,encoding='utf-8')
save(OUT/'CPU-preparation.json',{'python_AST_passed':True,'bash_n_rc':result.returncode,'source_files_unchanged':True,'all34_native_pins_unchanged':True,'device_authorized':False,'deployed':False,'scope':'Exact P2 schema2/original P1 and original DeepEP; EP2 serial56, no overlap successor. All16 preflight remains two hosts; launch/cleanup only host27. New helper gates do not query/initialize NPU.'})
save(OUT/'control-file-sha256.json',{p.name:sha(p) for p in NEW.iterdir() if p.is_file()})
print(json.dumps({'CPU_prepared':True,'bash_n_rc':0,'deployed':False,'controller_sha256':sha(NEW/'controller.py'),'launcher_sha256':sha(NEW/'run_dispatch_host_trace_checked.sh'),'manifest_sha256':sha(OUT/'rack05os7-diagnostic-deployment-manifest.json')}))
