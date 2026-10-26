"""Explicit no-device imports and host identity collection, never an engine/session."""
import pathlib,sys,os,json,importlib.util,types
R=pathlib.Path(os.environ['P2_SOURCE_ROOT']).resolve();F=pathlib.Path(os.environ['TRAINING_SOURCE_ROOT']).resolve();P1=pathlib.Path(os.environ['P1_SOURCE_ROOT']).resolve()
sys.path[:0]=[str(R),str(F/'tests'),str(F),str(R.parent),str(P1.parent)]
import torch,torch_npu,torch.distributed as dist
before={'npu_initialized':torch.npu.is_initialized(),'PG_initialized':dist.is_initialized()}
assert not any(before.values())
spec=importlib.util.spec_from_file_location('p2_cpu_validator',R/'validate_training_p2.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
plans=[];identities={}
for world,B,H,I,E,K in ((2,32,128,128,8,4),(16,32,128,128,64,8),(16,8192,6144,2048,64,8)):
 for schedule in ('serial','overlap'):
  a=types.SimpleNamespace(training_source_root=F,candidate_source_root=R,p1_source_root=P1,pointwise_library=pathlib.Path(os.environ['POINTWISE_SO']),world_size=world,B=B,H=H,I=I,E=E,K=K,cases='balanced' if B==8192 else 'balanced,zero_gate,empty,skew',shared=False,weight_dtype='bf16',seed=4317,oracle_chunk=1024,max_reference_gib=12.,backward_schedule=schedule)
  base,ref,candidate,root,reservation,execution=v.prepare(a,world,torch,torch_npu)
  plans.append({'world':world,'shape':execution['shape'],'schedule':schedule,'reservation':reservation,'passed':True})
  if world==2:
   identity,pointwise=v.collect_identity(a,root,execution,base)
   identities[schedule]=identity
import deep_ep._runtime as runtime
after={'npu_initialized':torch.npu.is_initialized(),'PG_initialized':dist.is_initialized(),'ElasticBuffer_owner_exists':getattr(runtime,'_OWNER',None) is not None}
assert not any(after.values())
assert identities['serial']['content_digest']!=identities['overlap']['content_digest']
assert identities['serial']['content']['mapped_libraries']==identities['overlap']['content']['mapped_libraries']
trace=[k for k in sys.modules if k.startswith('dispatch_host_integration')];assert not trace
print(json.dumps({'passed':True,'scope':'CPU import/host CDLL/maps identity only; no NPU tensor/PG/SHMEM Session/engine','before':before,'after':after,'plans':plans,'identities':identities,'host_trace_modules':trace,'NPU_qualification':False,'performance_qualification':False},indent=2))
