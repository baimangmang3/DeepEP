"""Actual wrapper help import in-process, CPU-only state assertions."""
import pathlib,runpy,sys,json,os,time,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--wrapper',required=True);p.add_argument('--output',required=True);a=p.parse_args()
report={'before_import_wall_ns':time.time_ns(),'wrapper_sha256':hashlib.sha256(pathlib.Path(a.wrapper).read_bytes()).hexdigest()}
sys.argv=[a.wrapper,'--help'];rc=None
try:runpy.run_path(a.wrapper,run_name='__main__')
except SystemExit as e:rc=e.code
import torch,torch.distributed as dist
report.update(rc=rc,npu_initialized=bool(hasattr(torch,'npu') and torch.npu.is_initialized()),PG_initialized=dist.is_initialized(),deep_ep_imported='deep_ep' in sys.modules,
              torch_npu_imported='torch_npu' in sys.modules,api_path=sys.modules['torch.distributed.elastic.agent.server.api'].__file__,pid=os.getpid(),wall_end_ns=time.time_ns())
assert rc in (0,None) and not report['npu_initialized'] and not report['PG_initialized'] and not report['deep_ep_imported']
report['passed']=True;pathlib.Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report))
