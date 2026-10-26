"""Read only this existing paired CPU RUN and exact recorded PID births."""
import pathlib,json,os,shlex,paramiko,datetime
OUT=pathlib.Path('outputs/p5_identity64_cpu_build');r=json.loads(pathlib.Path('outputs/p5_identity64_cpu_compile_parent_review.json').read_text());D=r['package_remote'];RUN=r['output_remote'];TAG=r['run_tag']
c=paramiko.SSHClient();c.load_system_host_keys();c.connect(r['host'],username='root',password=os.environ['DEPLOY_PASSWORD_27'],timeout=15,banner_timeout=15,auth_timeout=15)
try:
 code=f'''import pathlib,json,os,time
d=pathlib.Path({D!r});run=pathlib.Path({RUN!r});result={{'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'program_rc':int((d/'logs/program.rc').read_text()) if (d/'logs/program.rc').exists() else None,'known_PID_checks':[]}}
report=json.loads((run/'report.json').read_text()) if (run/'report.json').exists() else None
saved=[]
for name in ('recorder.identity.json','driver.identity.json'):
 p=d/'logs'/name
 if p.exists():saved.append(json.loads(p.read_text()))
if report:
 for step in report.get('runs',[]):
  saved+=step.get('process',{{}}).get('owned',[])
  for item in step.get('embedded_ELFs',[]):saved+=item.get('readelf_process',{{}}).get('owned',[])
seen=set()
for item in saved:
 key=(item['pid'],str(item['start']),item['uid'])
 if key in seen:continue
 seen.add(key);p=pathlib.Path('/proc')/str(item['pid'])
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split();same=fields[19]==str(item['start']) and p.stat().st_uid==item['uid'];now={{'start':fields[19],'uid':p.stat().st_uid,'state':fields[0],'wchan':(p/'wchan').read_text(),'same_PID_start_uid':same}}
 except FileNotFoundError:now={{'same_PID_start_uid':False,'proc_absent':True}}
 result['known_PID_checks'].append({{'saved':item,'current':now}})
result['report']=report
result['files']=[str(p.relative_to(d)) for parent in (d/'logs',run) if parent.exists() for p in parent.rglob('*') if p.is_file()]
result['all_observed_PID_births_gone']=all(not v['current']['same_PID_start_uid'] for v in result['known_PID_checks'])
print(json.dumps(result))'''
 cmd='/root/miniconda3/envs/zzy/bin/python - <<\'P5_CPU_STATUS\'\n'+code+'\nP5_CPU_STATUS\n';_,o,e=c.exec_command(cmd,timeout=30);stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status()
 (OUT/'status-command.json').write_text(json.dumps({'rc':rc,'stderr':stderr},indent=2));assert rc==0,(rc,stderr)
 result=json.loads(stdout);(OUT/'latest-status.json').write_text(json.dumps(result,indent=2))
 with c.open_sftp() as ftp:
  for rel in result['files']:
   local=OUT/'remote-results'/rel;local.parent.mkdir(parents=True,exist_ok=True);ftp.get(D+'/'+rel,str(local))
 report=result['report'];summary={'program_rc':result['program_rc'],'all_observed_PID_births_gone':result['all_observed_PID_births_gone'],'passed':report.get('passed') if report else None,'first_error':report.get('first_error') if report else None,'variants':[{k:v for k,v in item.items() if k in ('flavor','returncode','device_symbol_sources','object')} for item in (report or {}).get('runs',[])]}
 p=pathlib.Path('outputs/training_device_progress.json');s=json.loads(p.read_text(encoding='utf-8-sig'));key=s['p5_identity64_CPU_object_compile'];key.update(active=result['program_rc'] is None or not result['all_observed_PID_births_gone'],program_rc=result['program_rc'],passed=summary['passed'],latest_status=str(OUT/'latest-status.json'),first_error=summary['first_error']);s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(s,indent=2));print(json.dumps(summary))
finally:c.close()
