"""One bounded host27-only CPU disk read, approved request; no compiler/readelf execution."""
import pathlib,json,hashlib,os,shlex,paramiko,time
REQUEST=pathlib.Path('outputs/p5_identity64_dependency_binding_request.json');READER=pathlib.Path('work/p5_dependency_binding_reader.py')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(REQUEST)=='57db77e06a1b82522e79afa633b79c67a6a77edeee92832e7d3b09004d6c47dc'
r=json.loads(REQUEST.read_text());assert r['host']=='177.22.27.1' and len(r['external_dependency_paths'])==594
PLAN=pathlib.Path('work/training_candidates/p5_identity64_compile_plan/manifest.json');m=json.loads(PLAN.read_text())
assert set(m['external_dependency_paths'])==set(r['external_dependency_paths']) and m['compiler']==r['compiler']['path']
assert m['device_elf_machine']==r['device_elf_machine']==4137 and r['device_symbols']==['dispatch_kernel','dispatch_profile_kernel']
OUT=pathlib.Path('outputs/p5_identity64_tool_dependency_binding');OUT.mkdir(exist_ok=False)
R=r['remote_new_files_root'];assert R.startswith('/home/zzy/')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect(r['host'],username='root',password=os.environ['DEPLOY_PASSWORD_27'],timeout=15,banner_timeout=15,auth_timeout=15)
try:
 # Only allowed binding-root new files; no P5 candidate source or backend is copied.
 cmd='/root/miniconda3/envs/zzy/bin/python -c '+shlex.quote('import pathlib; pathlib.Path('+repr(R)+').mkdir(parents=True,exist_ok=False)')
 _,o,e=c.exec_command(cmd,timeout=20);stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status()
 (OUT/'setup-command.json').write_text(json.dumps({'command':cmd,'rc':rc,'stdout':stdout,'stderr':stderr},indent=2));assert rc==0,(rc,stderr)
 with c.open_sftp() as ftp:
  ftp.put(str(REQUEST),R+'/binding-request.json');ftp.put(str(READER),R+'/binding_reader.py')
 cmd='timeout --signal=TERM --kill-after=1 120 /root/miniconda3/envs/zzy/bin/python '+shlex.quote(R+'/binding_reader.py')
 start_wall=time.time_ns();start_mono=time.monotonic_ns()
 _,o,e=c.exec_command(cmd,timeout=135);stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status()
 result={'command':cmd,'rc':rc,'stdout':stdout,'stderr':stderr,'local_start_wall_ns':start_wall,'local_start_monotonic_ns':start_mono,'local_end_wall_ns':time.time_ns(),'local_end_monotonic_ns':time.monotonic_ns(),'remote_timeout_seconds':120}
 (OUT/'read-command-result.json').write_text(json.dumps(result,indent=2));(OUT/'stdout.log').write_text(stdout);(OUT/'stderr.log').write_text(stderr)
 with c.open_sftp() as ftp:
  for source,target in [('result.json','full-result.json'),('bindings.json','bindings.json')]:
   try:ftp.get(R+'/'+source,str(OUT/target))
   except FileNotFoundError:pass
 inputs={'approved_request_sha256':sha(REQUEST),'reader_sha256':sha(READER),'compile_plan_manifest_sha256':sha(PLAN),'compile_objects_source_sha256':sha('work/training_candidates/p5_identity64_compile_plan/compile_objects.py'),'remote_root':R,'host':r['host'],'compiler_or_readelf_executed':False,'NPU_or_occupancy_queried':False}
 (OUT/'input-hashes.json').write_text(json.dumps(inputs,indent=2));(OUT/'binding_reader.py').write_bytes(READER.read_bytes());(OUT/'approved-binding-request.json').write_bytes(REQUEST.read_bytes())
 report=json.loads((OUT/'full-result.json').read_text()) if (OUT/'full-result.json').exists() else None
 print(json.dumps({'actual_rc':rc,'passed':bool(report and report.get('passed')),'dependency_entries':len(report['dependencies']) if report else None,'missing_or_read_errors':report.get('errors') if report else 'No final report; see raw command','compiler':report.get('compiler') if report else None,'readelf':report.get('readelf') if report else None,'no_compiler_readelf_or_NPU_execution':True}))
finally:c.close()
