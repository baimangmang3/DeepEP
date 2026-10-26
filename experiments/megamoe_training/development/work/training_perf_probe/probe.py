"""Qualified full-training totals/trace only; no NPU use unless explicitly run.

Requires TRAINING_SOURCE_ROOT pointing at the frozen qualified experiment root.
Default performs qualification gate only. --benchmark and --profile are separate
runs. Example: --backends shmem_catlass,cann_ai_cpu --qualification
shmem_catlass=/ABS/P0 --qualification cann_ai_cpu=/ABS/CANN --benchmark.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from gate import qualify


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key,value in [('B',8192),('H',6144),('I',2048),('E',64),('K',8)]:ap.add_argument('--'+key,type=int,default=value)
    ap.add_argument('--seed',type=int,default=4317)
    ap.add_argument('--backends',default='shmem_catlass,cann_ai_cpu')
    ap.add_argument('--qualification',action='append',default=[],metavar='BACKEND=DIRECTORY')
    ap.add_argument('--shared',action='store_true')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--benchmark',action='store_true')
    mode.add_argument('--profile',action='store_true')
    mode.add_argument('--debug-steps',type=int,default=0)
    ap.add_argument('--warmup',type=int,default=5);ap.add_argument('--iters',type=int,default=30)
    ap.add_argument('--profile-steps',type=int,choices=[2,3],default=3)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();names=args.backends.split(',')
    directories=dict(pair.split('=',1) for pair in args.qualification)
    if (not names or len(set(names))!=len(names) or set(names)-{'shmem_catlass','cann_ai_cpu'} or
            set(directories)!=set(names) or len(directories)!=len(args.qualification) or
            (args.shared and names!=['shmem_catlass'])):
        raise ValueError('Supply exactly one qualification directory per backend; shared is P0-only')
    if min(args.B,args.H,args.I,args.E,args.K,args.iters)<=0 or args.warmup<0 or args.debug_steps<0:
        raise ValueError('Invalid dimensions or step counts')
    import torch,torch_npu,torch.distributed as dist
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
    dist.init_process_group('hccl')
    rank,world=dist.get_rank(),dist.get_world_size()
    device=f'npu:{int(os.environ["LOCAL_RANK"])}'
    args.output.mkdir(parents=True,exist_ok=True)
    report={'rank':rank,'world':world,'passed':False,'mode':'gate_only','argv':[sys.executable,*sys.argv],
        'harness_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('gate.py'))},
        'environment':{key:value for key,value in os.environ.items() if key.startswith(('ASCEND_','HCCL_','DEEPEP_','SHMEM_','TORCH_','NPU_','ACL_')) or key in ('TRAINING_SOURCE_ROOT','LD_LIBRARY_PATH','PYTHONPATH','PATH','OMP_NUM_THREADS')},
        'environment_scope':'Exact recorded execution-variable values; unrelated account/secrets are not dumped',
        'backend_scope':{'shmem_catlass':'P0 separate SHMEM routing and CATLASS GEMMs',
                         'cann_ai_cpu':'CANN fused communication/GMM forward+dgrad; separate activation/pack/unpack; same CATLASS FP32 wgrad'},
        'scope':'full forward+backward totals, gradient reset included; no comm-only/overlap/MFU/HFU claim'}
    def agree(error,stage):
        flag=torch.tensor([int(error is None)],device=device,dtype=torch.int32)
        dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        if not flag.item():raise ValueError(error or f'Another rank rejected {stage}; no rank may enter workloads')
    engines={}
    try:
        error=None
        try:
            root=Path(os.environ['TRAINING_SOURCE_ROOT']).resolve(strict=True)
            sys.path[:0]=[str(root/'tests'),str(root)]
            prov=importlib.import_module('provenance')
            validation=importlib.import_module('validate_training_npu')
            comparison=importlib.import_module('benchmark_compare')
            package=importlib.import_module('megamoe_training')
            cann=importlib.import_module('cann_training') if 'cann_ai_cpu' in names else None
            for module in (prov,validation,comparison,package,*(() if cann is None else (cann,))):
                if not Path(module.__file__).resolve().is_relative_to(root):raise ValueError('Import escaped frozen TRAINING_SOURCE_ROOT: '+module.__name__)
            if world!=16 or args.E%world or args.B>16384 or args.K>min(args.E,16):raise ValueError('Requires legal EP16 HT shape')
        except Exception as exc:error=repr(exc)
        agree(error,'frozen imports/shape')
        identity=prov.collective_identity(torch,dist);report['identity']=identity
        shape={key:getattr(args,key) for key in ('B','H','I','E','K')}
        report['execution']={'shape':shape,'seed':args.seed,'weight_dtype':'torch.bfloat16','shared':args.shared,
                             'route':'balanced','backends':names,'schedule':'frozen default; no overlap/recompute options exposed'}
        common=dict(report['execution'],identity=identity['content_digest'],harness=report['harness_sha256'],
                    benchmark=args.benchmark,profile=args.profile,debug_steps=args.debug_steps,
                    warmup=args.warmup,iters=args.iters,profile_steps=args.profile_steps)
        digest=hashlib.sha256(json.dumps(common,sort_keys=True).encode()).digest()
        token=torch.tensor(list(digest),device=device,dtype=torch.uint8)
        peers=[torch.empty_like(token) for _ in range(world)]
        dist.all_gather(peers,token)
        agree(None if all(bool(torch.equal(peer,token)) for peer in peers) else 'Cross-rank execution/harness identity mismatch','execution agreement')
        error=None
        try:
            report['qualification']={name:qualify(directories[name],name,world,shape,args.seed,args.shared,
                                                  identity,validation.LIMITS,prov.validate_identity) for name in names}
        except Exception as exc:error=repr(exc)
        agree(error,'independent backend full-gradient qualifications')
        report['qualified']=True
        if args.benchmark or args.profile or args.debug_steps:
            args.weight_dtype=torch.bfloat16
            data=validation.inputs(args,rank,world,'balanced',device)
            x,w1,w2,ids,p,dy=data
            shared=()
            if args.shared:
                # Exact frozen validator generator/scale/dtype/order, no new math.
                gen=torch.Generator(device=x.device).manual_seed(args.seed+9000+rank)
                sw1=torch.randn((2*args.I,args.H),device=x.device,generator=gen).mul_(1/math.sqrt(args.H)).to(args.weight_dtype).requires_grad_()
                sw2=torch.randn((args.H,args.I),device=x.device,generator=gen).mul_(1/math.sqrt(args.I)).to(args.weight_dtype).requires_grad_()
                shared=(sw1,sw2)
            for name in names:
                kwargs=dict(group=dist.group.WORLD,max_tokens=args.B,hidden=args.H,topk=args.K,num_experts=args.E)
                engines[name]=package.TrainingMoE(**kwargs) if name=='shmem_catlass' else cann.CannTrainingMoE(**kwargs,comm_mode='ai_cpu')
            def step(name):
                if not torch.is_grad_enabled():
                    raise RuntimeError('Full-training probe requires grad enabled; no_grad is forbidden')
                validation.clear_grads((x,w1,w2,p)+shared)
                kw={} if not shared else dict(zip(('shared_w1','shared_w2'),shared))
                y=engines[name](x,w1,w2,ids,p,**kw)
                y.backward(dy)
                return y
            functions={name:(lambda name=name:step(name)) for name in names}
            flops=6*args.B*args.K*args.H*args.I
            shared_flops=6*args.B*args.H*args.I if shared else 0
            report['effective_GEMM_FLOPs_per_rank']={'forward':flops+shared_flops,'full_forward_backward':3*(flops+shared_flops),
                'convention':'2 FLOPs per multiply-add; excludes pointwise/router/communication/optimizer; no peak utilization claim'}
            if args.benchmark:
                report['mode']='benchmark'
                # Intentionally grad-enabled; reuse exact frozen rotation/event/wall scope.
                report['timing']=comparison.timing_round_robin(functions,dist.group.WORLD,args.warmup,args.iters,x.device)
            elif args.profile:
                report['mode']='profile';report['traces']={}
                report['profile_alignment_scope']='Each profiled call: synchronize/barrier/synchronize outside record_function; step plus final synchronize inside label. Alignment barrier is external calibration, not training operator cost.'
                from torch_npu.profiler import profile,ProfilerActivity
                from torch.profiler import record_function
                for name in names:
                    for _ in range(2):step(name)
                    torch.npu.synchronize();dist.barrier()
                    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],record_shapes=True) as prof:
                        for _ in range(args.profile_steps):
                            torch.npu.synchronize();dist.barrier();torch.npu.synchronize()
                            with record_function('training_perf_probe/'+name+'/forward_backward'):
                                step(name)
                                torch.npu.synchronize()
                        torch.npu.synchronize()
                    path=args.output/f'{name}_rank{rank}_trace.json'
                    prof.export_chrome_trace(str(path));report['traces'][name]=str(path.resolve())
            else:
                report['mode']='debug';report['debug_steps']=args.debug_steps
                for _ in range(args.debug_steps):
                    for name in names:step(name)
                torch.npu.synchronize()
        report['passed']=True
    except Exception as exc:
        report['error']=repr(exc)
        raise
    finally:
        for engine in engines.values():
            try:engine.close()
            except Exception as exc:report.setdefault('cleanup_errors',[]).append(repr(exc))
        try:dist.destroy_process_group()
        except Exception as exc:report.setdefault('cleanup_errors',[]).append(repr(exc))
        if report.get('cleanup_errors'):report['passed']=False
        (args.output/f'perf_rank{rank}.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
