"""Forensic CPU map observation only; does not call/bypass the rejected P2 gate."""
import pathlib,os,sys,json,types,importlib.util,collections
R=pathlib.Path(os.environ['P2_SOURCE_ROOT']).resolve();F=pathlib.Path(os.environ['TRAINING_SOURCE_ROOT']).resolve();P1=pathlib.Path(os.environ['P1_SOURCE_ROOT']).resolve()
sys.path[:0]=[str(R),str(F/'tests'),str(F),str(R.parent),str(P1.parent)]
import torch,torch_npu,torch.distributed as dist
before={'npu_initialized':torch.npu.is_initialized(),'PG_initialized':dist.is_initialized()};assert not any(before.values())
spec=importlib.util.spec_from_file_location('p2_cpu_validator',R/'validate_training_p2.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
a=types.SimpleNamespace(training_source_root=F,candidate_source_root=R,p1_source_root=P1,pointwise_library=pathlib.Path(os.environ['POINTWISE_SO']),world_size=2,B=32,H=128,I=128,E=8,K=4,cases='balanced,zero_gate,empty,skew',shared=False,weight_dtype='bf16',seed=4317,oracle_chunk=1024,max_reference_gib=12.,backward_schedule='serial')
base,ref,candidate,root,reservation,execution=v.prepare(a,2,torch,torch_npu)
from p1_pointwise.pointwise import TrainingPointwise
pointwise=TrainingPointwise(a.pointwise_library,vector_cores=64)
identity=base.collect_runtime_identity()
from p2_backward_overlap.mapped_identity import mapped_libraries
maps=mapped_libraries();groups=collections.defaultdict(list)
for item in maps['libraries']:groups[pathlib.PurePosixPath(item['path']).name].append(item)
collisions={k:items for k,items in groups.items() if len({v['sha256'] for v in items})>1}
import deep_ep._runtime as runtime
after={'npu_initialized':torch.npu.is_initialized(),'PG_initialized':dist.is_initialized(),'ElasticBuffer_owner_exists':getattr(runtime,'_OWNER',None) is not None};assert not any(after.values())
print(json.dumps({'scope':'CPU map forensics; rejected P2 checked_maps/collect_identity not called; not qualification','before':before,'after':after,'collision_groups':collisions,'raw_mapping_result':maps,'base_identity':identity,'qualification':False},indent=2))
