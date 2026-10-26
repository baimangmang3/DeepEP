import pathlib,hashlib,json,os,paramiko
T='/home/zzy/deploy/megamoe-20261024/training';D=T+'/cpu-review-p2';W=D+'/work/training_candidates'
files={
'p2_backward_overlap/training_adapter.py':pathlib.Path('work/training_candidates/p2_backward_overlap/training_adapter.py'),
'p2_backward_overlap/__init__.py':pathlib.Path('work/training_candidates/p2_backward_overlap/__init__.py'),
'p2_backward_overlap/tests/test_math_cpu.py':pathlib.Path('work/training_candidates/p2_backward_overlap/tests/test_math_cpu.py'),
'p1_pointwise/training_adapter.py':pathlib.Path('work/training_candidates/p1_pointwise/training_adapter.py'),
'ep_gmm_training/tests/test_adapter_cpu.py':pathlib.Path('work/training_candidates/ep_gmm_training/tests/test_adapter_cpu.py')}
assert hashlib.sha256(files['p2_backward_overlap/training_adapter.py'].read_bytes()).hexdigest()=='10b3c34d27dd97273c4c19c64c2416dfe76ccb953243c45e806cd8c3f750abce'
assert hashlib.sha256(files['p2_backward_overlap/__init__.py'].read_bytes()).hexdigest()=='67cc087f7091e31315742ed34d021599bb caff7af20b190ef45273289183af02'.replace(' ','')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('177.22.28.1',username='root',password=os.environ['DEPLOY_PASSWORD_28'],timeout=15)
def call(cmd):
 _,o,e=c.exec_command(cmd,timeout=210);a=o.read().decode(errors='replace');b=e.read().decode(errors='replace');return o.channel.recv_exit_status(),a,b
layout=f'''import pathlib,json
root=pathlib.Path({D!r});files={list(files)!r}
for n in files:(root/'work/training_candidates'/n).parent.mkdir(parents=True,exist_ok=True)
link=root/'work/DeepEP';target=pathlib.Path({T!r})/'qualified-p0-20261025'
if link.exists():assert link.resolve()==target.resolve()
else:link.symlink_to(target,target_is_directory=True)
print(json.dumps({{'reference_root':str(link.resolve())}}))'''
rc,text,error=call("timeout 20 /root/miniconda3/envs/zzy/bin/python - <<'PY'\n"+layout+"\nPY");assert rc==0,error
with c.open_sftp() as s:
 for n,p in files.items():s.put(str(p),W+'/'+n)
 runner=f'''import pathlib,runpy,sys,json,traceback,hashlib,os,torch,torch_npu,torch.distributed as dist
root=pathlib.Path({D!r});target=root/'work/training_candidates/p2_backward_overlap/tests/test_math_cpu.py'
def state():
 return {{'pid':os.getpid(),'npu_initialized':torch.npu.is_initialized(),'process_group_initialized':dist.is_initialized(),'udma_modules':[k for k in sys.modules if k.startswith(('deep_ep_experimental.ep_gmm_fused','deep_ep._C'))]}}
report={{'scope':'P2 CPU stub/autograd only; no NPU/transport/overlap qualification','before':state(),'torch':torch.__version__,'torch_npu':torch_npu.__version__}}
assert not report['before']['npu_initialized'] and not report['before']['process_group_initialized'] and not report['before']['udma_modules']
files=[root/'work/training_candidates'/n for n in {list(files)!r}]
files += [root/'work/DeepEP/experiments/megamoe_training'/n for n in ['tests/test_training_math_cpu.py','tests/validate_training_npu.py','megamoe_training/training.py']]
report['source_sha256']={{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
code=0
try:
 sys.argv=[str(target),'--output',str(root/'cpu-results.json')];runpy.run_path(str(target),run_name='__main__')
except Exception:
 code=1;report['error']=traceback.format_exc()
finally:
 report['after']=state();report['isolated']=not report['after']['npu_initialized'] and not report['after']['process_group_initialized'] and not report['after']['udma_modules']
 (root/'isolation.json').write_text(json.dumps(report,indent=2))
 if not report['isolated']:code=1
raise SystemExit(code)
'''
 with s.open(D+'/cpu-runner.py','w') as f:f.write(runner)
rc,text,error=call(f"bash -c 'source {T}/activate-qualified-p0.sh; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10 180 nice -n 10 python {D}/cpu-runner.py'")
out=pathlib.Path('outputs/p2_training_math_cpu');out.mkdir(exist_ok=True);(out/'execution.json').write_text(json.dumps({'returncode':rc,'stdout':text,'stderr':error,'remote_root':D,'source_sha256':{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}},indent=2))
with c.open_sftp() as s:
 for n in ('cpu-results.json','isolation.json'):
  try:s.get(D+'/'+n,str(out/n))
  except OSError:pass
c.close();print(json.dumps({'returncode':rc,'stdout':text,'stderr':error[-2000:],'output':str(out)}))
