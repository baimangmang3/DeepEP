"""Read-only CPU deployment validation. Never initialize or select an NPU."""
import pathlib,json,hashlib,py_compile,datetime,os
t=pathlib.Path('/home/zzy/deploy/megamoe-20261024/training')
m=json.loads((t/'logs/training_source_manifest.json').read_text())
for row in m['files']:
 p=t/'DeepEP'/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p)
compiled=[]
for p in (t/'DeepEP/experiments/megamoe_training').rglob('*.py'):
 py_compile.compile(str(p),doraise=True);compiled.append(str(p))
lib=t/'DeepEP/experiments/megamoe_training/build/libmegamoe_training.so'
assert hashlib.sha256(lib.read_bytes()).hexdigest()==m['native_library_sha256']
import torch,torch_npu,deep_ep
before=torch.npu.is_initialized()
from megamoe_training import TrainingMoE
from megamoe_training.gemm import CatlassGemm
native=CatlassGemm(cores=32)
after=torch.npu.is_initialized()
assert not before and not after
assert str(t/'venv') in deep_ep.__file__,deep_ep.__file__
report=dict(time=datetime.datetime.now().astimezone().isoformat(),source_files_verified=len(m['files']),python_files_compiled=len(compiled),library_sha256=m['native_library_sha256'],backend=native.backend,cores=native.cores,deep_ep=deep_ep.__file__,training_module=__import__('megamoe_training').__file__,torch=torch.__version__,torch_npu=torch_npu.__version__,npu_initialized_before=before,npu_initialized_after=after,npu_tests_executed=False,passed=True)
(t/'logs/cpu-deployment-validation.json').write_text(json.dumps(report,indent=2))
own=[]
names={'test_native_gemm.py','test_training_gemm.py','validate_training_npu.py','benchmark_compare.py','probe_cann_aiv.py'}
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=[x.decode(errors='replace') for x in p.read_bytes().split(b'\0') if x]
  if any(pathlib.PurePosixPath(x).name in names and str(t) in x for x in args):own.append(dict(pid=int(p.parent.name),args=args))
 except (OSError,ValueError):pass
(t/'logs/own-device-processes.json').write_text(json.dumps(own,indent=2))
assert not own,own
print(json.dumps(report,indent=2))
