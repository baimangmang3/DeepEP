"""Deploy unchanged CPU-only AST/stub tests; mirror layout links frozen P0 read-only."""
import hashlib,json,os
from pathlib import Path
import paramiko

T='/home/zzy/deploy/megamoe-20261024/training';D=T+'/cpu-review-ep-gmm';P=D+'/work/training_candidates/ep_gmm_training'
sources={'training_adapter.py':Path('work/training_candidates/ep_gmm_training/training_adapter.py'),
         'tests/test_adapter_cpu.py':Path('work/training_candidates/ep_gmm_training/tests/test_adapter_cpu.py')}
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('177.22.28.1',username='root',password=os.environ['DEPLOY_PASSWORD_28'],timeout=15)
def run(command):
    _,stdout,stderr=c.exec_command(command,timeout=150);text=stdout.read().decode(errors='replace');error=stderr.read().decode(errors='replace');return stdout.channel.recv_exit_status(),text,error
try:
    command=f"/root/miniconda3/envs/zzy/bin/python - <<'CPU_LAYOUT'\nimport pathlib,json\np=pathlib.Path({P!r});(p/'tests').mkdir(parents=True,exist_ok=True)\nlink=pathlib.Path({D!r})/'work/DeepEP';target=pathlib.Path({T!r})/'qualified-p0-20261025'\nif link.exists():assert link.resolve()==target.resolve()\nelse:link.symlink_to(target,target_is_directory=True)\nprint(json.dumps({{'frozen_cpu_reference':str((link/'experiments/megamoe_training/tests/test_training_math_cpu.py').resolve())}}))\nCPU_LAYOUT\n"
    rc,text,error=run(command);assert rc==0,error
    s=c.open_sftp()
    for name,source in sources.items():s.put(str(source),P+'/'+name)
    runner=f'''
import pathlib,runpy,sys,json,traceback,torch,torch.distributed as dist
p=pathlib.Path({D!r});report={{'scope':'CPU AST/stubs only; no native UD transport or NPU qualification'}}
def state():
 n=getattr(torch,'npu',None)
 return {{'npu_initialized':bool(n is not None and n.is_initialized()),'process_group_initialized':dist.is_initialized(),'udma_modules':[k for k in sys.modules if k.startswith('deep_ep_experimental.ep_gmm_fused')]}}
report['before']=state();assert not report['before']['npu_initialized'] and not report['before']['process_group_initialized'] and not report['before']['udma_modules']
code=0
try:
 sys.argv=[{P!r}+'/tests/test_adapter_cpu.py','--output',str(p/'cpu-results.json')]
 runpy.run_path(sys.argv[0],run_name='__main__')
except Exception:
 code=1;report['error']=traceback.format_exc()
finally:
 report['after']=state();report['isolated']=not report['after']['npu_initialized'] and not report['after']['process_group_initialized'] and not report['after']['udma_modules'];(p/'isolation.json').write_text(json.dumps(report,indent=2))
 if not report['isolated']:code=1
raise SystemExit(code)
'''
    with s.open(D+'/cpu-runner.py','w') as f:f.write(runner)
    s.close()
    rc,text,error=run(f"bash -c 'source {T}/activate-qualified-p0.sh; OMP_NUM_THREADS=1 timeout --signal=TERM --kill-after=10 120 python {D}/cpu-runner.py'")
    out=Path('outputs/ep_gmm_training_cpu_review');out.mkdir(exist_ok=True)
    (out/'execution.json').write_text(json.dumps(dict(host='177.22.28.1',returncode=rc,stdout=text,stderr=error,source_sha256={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sources.items()},frozen_layout=text),indent=2))
    with c.open_sftp() as s:
        for name in ('cpu-results.json','isolation.json'):
            try:s.get(D+'/'+name,str(out/name))
            except OSError:pass
    print(json.dumps(dict(returncode=rc,stdout=text,stderr=error[-2000:],output=str(out))))
finally:c.close()
