"""One approved host27 standard-library worker; no NPU/PG worker operations."""
import pathlib,json,shlex
from training_native_round import SshBackend,T,HOSTS
R=T+'/validation-p2-ep2-serial-hostname-20261026';OUT=pathlib.Path('outputs/p2_ep2_serial_hostname_preparation/rack05os7')
code=r'''import pathlib,json,os,time,subprocess,signal,sys
r=pathlib.Path(__R__);out=r/'evidence';result_path=out/'cpu-worker-smoke.json';assert not result_path.exists(),'One shot already consumed'
for table in ('/proc/net/tcp','/proc/net/tcp6'):
 for line in pathlib.Path(table).read_text().splitlines()[1:]:
  f=line.split();assert not (f[3]=='0A' and int(f[1].rsplit(':',1)[1],16)==29956),'CPU port29956 occupied; no launch'
worker=out/'stdlib_worker.py'
worker.write_text("import os,json,pathlib,sys,time\np=pathlib.Path('/proc')/str(os.getpid());v={'pid':os.getpid(),'start':p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],'rank':os.getenv('RANK'),'world_size':os.getenv('WORLD_SIZE'),'Torch_imported':'torch' in sys.modules,'torch_npu_imported':'torch_npu' in sys.modules,'wall_ns':time.time_ns()};pathlib.Path(sys.argv[1]).write_text(json.dumps(v));time.sleep(1);print(json.dumps(v),flush=True)\n")
tag='CPU_p2_hostname_smoke_once';wrapper=r/'scripts/launcher.py'
cmd=[sys.executable,str(wrapper),'--nnodes=1','--nproc-per-node=1','--max-restarts=0','--rdzv-backend=static','--master-addr=127.0.0.1','--master-port=29956',str(worker),str(out/'stdlib-worker-report.json')]
report=dict(command=cmd,run_tag=tag,wall_start_ns=time.time_ns(),monotonic_start_ns=time.monotonic_ns(),scope='CPU torchrun launcher +1 stdlib worker only; no NPU/tensor/worker PG',timeout_seconds=120)
env=dict(os.environ,RUN_TAG=tag)
with (out/'smoke.stdout').open('xb') as stdout,(out/'smoke.stderr').open('xb') as stderr:
 p=subprocess.Popen(cmd,env=env,start_new_session=True,stdout=stdout,stderr=stderr)
 proc=pathlib.Path('/proc')/str(p.pid);fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split();start=fields[19]
 argv=[v.decode(errors='replace') for v in proc.joinpath('cmdline').read_bytes().split(b'\0') if v]
 sys.path.insert(0,str(r/'scripts'));import cleanup_own,cleanup_policy
 assert cleanup_own.WRAPPER==str(wrapper)
 report['inner_preworker_identity']=dict(pid=p.pid,start=start,pgid=os.getpgid(p.pid),argv=argv,matched_reviewed_inner_selector=str(wrapper) in argv,wall_ns=time.time_ns(),monotonic_ns=time.monotonic_ns())
 assert str(wrapper) in argv
 foreign=cleanup_policy.external_launcher_evidence([dict(pid=p.pid,start=start,ppid=os.getpid(),argv=argv,run_tag=tag)],'other',known_wrappers=(str(wrapper),))
 own=cleanup_policy.external_launcher_evidence([dict(pid=p.pid,start=start,ppid=os.getpid(),argv=argv,run_tag=tag)],tag,known_wrappers=(str(wrapper),))
 assert foreign['busy_or_uncertain'] and not own['busy_or_uncertain'];report['known_wrapper_foreign_vs_own_classification_passed']=True
 try:p.wait(timeout=120);report['timeout']=False
 except subprocess.TimeoutExpired as exc:
  report.update(timeout=True,error=repr(exc),signals=[])
  for sig,wait in ((signal.SIGTERM,5),(signal.SIGKILL,5)):
   try:
    now=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
    assert now==start and os.getpgid(p.pid)==p.pid,'PID/start/group mismatch'
    event=dict(signal=int(sig),time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),pid=p.pid,start=start);os.killpg(p.pid,sig);report['signals'].append(event)
    p.wait(timeout=wait);break
   except FileNotFoundError:break
   except subprocess.TimeoutExpired:pass
 report.update(returncode=p.returncode,reaped=p.returncode is not None,wall_end_ns=time.time_ns(),monotonic_end_ns=time.monotonic_ns())
if (out/'stdlib-worker-report.json').exists():
 worker_report=json.loads((out/'stdlib-worker-report.json').read_text());report['worker']=worker_report
 wp=pathlib.Path('/proc')/str(worker_report['pid']);same=False
 try:same=wp.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==worker_report['start']
 except FileNotFoundError:pass
 report['worker_pid_start_gone']=not same
report['passed']=report['returncode']==0 and not report['timeout'] and report.get('worker_pid_start_gone') is True and report.get('worker',{}).get('Torch_imported') is False and report.get('worker',{}).get('torch_npu_imported') is False
result_path.write_text(json.dumps(report,indent=2));print(json.dumps(report))
raise SystemExit(0 if report['passed'] else 1)
'''
x=SshBackend();x.connect()
try:
 host=HOSTS[0];payload=code.replace('__R__',repr(R))
 # Run under activated venv, bounded independently from the child120s supervisor.
 cmd=f'source {T}/activate-qualified-p0.sh\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\npython - <<\'CPU_SMOKE\'\n'+payload+'\nCPU_SMOKE\n'
 _,o,e=x.clients[host].exec_command('bash -lc '+shlex.quote(cmd),timeout=145);stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status()
 (OUT/'smoke-controller-command.json').write_text(json.dumps({'command':cmd,'rc':rc,'stdout':stdout,'stderr':stderr},indent=2))
 with x.clients[host].open_sftp() as ftp:
  for name in ('cpu-worker-smoke.json','smoke.stdout','smoke.stderr','stdlib-worker-report.json','stdlib_worker.py'):
   try:ftp.get(R+'/evidence/'+name,str(OUT/name))
   except FileNotFoundError:pass
 assert rc==0,(rc,stderr)
 report=json.loads((OUT/'cpu-worker-smoke.json').read_text());assert report['passed'];print(json.dumps(report))
finally:x.close()
