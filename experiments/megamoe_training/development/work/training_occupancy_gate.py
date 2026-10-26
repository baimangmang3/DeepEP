"""Read-only ownership preflight; no torch/ACL import or device initialization."""
import datetime,json,pathlib,socket,subprocess,os
t=pathlib.Path('/home/zzy/deploy/megamoe-20261024/training')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run=t/'runs/20261025_0300/preflight';run.mkdir(parents=True,exist_ok=True)
smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
processes=[]
tests={'test_dispatch.py','test_native_gemm.py','validate_training_npu.py','benchmark_compare.py','probe_cann_aiv.py','test_training_gemm.py','hccl_minimal_smoke.py','hccl_paired_smoke.py','query_training_cores.py','run_training_review_checked.sh','run_native_training_checked.sh','acl_initialization_probe.py','acl_probe_supervisor.py','pta_initialization_probe.py','test_pointwise_npu.py','validate_training_p1.py','lifecycle_probe.py','run_training_lifecycle_checked.sh','run_training_p1_host_v2_checked.sh','pure_hccl_probe.py','run_hccl_control_checked.sh','hccl_control_supervisor.py','acl_probe_supervisor.py'}
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=[x.decode(errors='replace') for x in p.read_bytes().split(b'\0') if x]
  if args and (any(pathlib.PurePosixPath(x).name in tests for x in args) or str(t/'scripts/training_perf_probe/probe.py') in args):
   st=(p.parent/'status').read_text();ppid=next(x.split(':',1)[1].strip() for x in st.splitlines() if x.startswith('PPid:'))
   processes.append(dict(pid=int(p.parent.name),ppid=int(ppid),own=any(str(t) in x for x in args),args=args))
 except (OSError,ValueError,StopIteration):pass
result_paths=[dict(path=str(p),bytes=p.stat().st_size) for p in (t/'runs').glob('*/**/rank*.json')]
r=dict(time_utc=stamp,host=socket.gethostname(),npu_smi_rc=smi.returncode,free_cards=smi.stdout.count('No running processes'),all_eight_free=smi.returncode==0 and smi.stdout.count('No running processes')==8,processes=processes,existing_rank_results=result_paths)
prefix=run/(socket.gethostname()+'-'+stamp)
prefix.with_suffix('.json').write_text(json.dumps(r,indent=2));prefix.with_suffix('.txt').write_text(smi.stdout+smi.stderr)
print(smi.stdout+smi.stderr);print(json.dumps(r,indent=2));print('EVIDENCE_PREFIX='+str(prefix))
