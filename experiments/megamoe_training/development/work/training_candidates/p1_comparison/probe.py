"""One backend per process for qualified P0/P1/CANN comparisons.

Run separate P0,P1,P1,P0 trials (and reversed blocks) with identical inputs.
Only --benchmark or --profile occupies NPU; otherwise prints a CPU-only plan.
P1 full-gradient qualification must match its current source, actual libraries
and fixed accuracy configuration. Runtime creation is outside measured steps.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', required=True,
        choices=['shmem_catlass', 'shmem_catlass_pointwise_p1', 'cann_ai_cpu'])
    ap.add_argument('--qualification', type=Path, required=True)
    ap.add_argument('--candidate-source-root', type=Path)
    ap.add_argument('--pointwise-library', type=Path)
    ap.add_argument('--trial', required=True)
    for key, value in [('B',8192), ('H',6144), ('I',2048), ('E',64), ('K',8)]:
        ap.add_argument('--'+key, type=int, default=value)
    ap.add_argument('--seed', type=int, default=4317)
    ap.add_argument('--shared', action='store_true')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--profile-steps', type=int, default=2, choices=[2,3])
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--benchmark', action='store_true')
    modes.add_argument('--profile', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    is_p1 = args.backend == 'shmem_catlass_pointwise_p1'
    if (min(args.B,args.H,args.I,args.E,args.K,args.iters) <= 0 or args.warmup < 0 or
            (args.shared and args.backend == 'cann_ai_cpu') or
            (is_p1 and (args.candidate_source_root is None or args.pointwise_library is None))):
        raise ValueError('Invalid dimensions/steps/shared/backend/candidate options')
    if not args.benchmark and not args.profile:
        print(json.dumps({'status':'CPU plan only; no qualification or NPU claim',
                          'options':vars(args)}, default=str, indent=2))
        return
    import torch, torch_npu, torch.distributed as dist
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = False
    dist.init_process_group('hccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    device = f'npu:{int(os.environ["LOCAL_RANK"])}'
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'rank':rank, 'world':world, 'passed':False, 'backend':args.backend,
        'trial':args.trial, 'argv':[sys.executable,*sys.argv],
        'harness_sha256':{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('gate.py'))},
        'scope':'gradient clear + dynamic complete forward + backward; no optimizer/recompute; one SHMEM owner per process',
        'mode':'profile' if args.profile else 'benchmark',
        'environment':{k:v for k,v in os.environ.items()
            if k.startswith(('ASCEND_','HCCL_','DEEPEP_','SHMEM_','TORCH_','NPU_','ACL_')) or
            k in ('TRAINING_SOURCE_ROOT','LD_LIBRARY_PATH','PYTHONPATH','PATH','OMP_NUM_THREADS')}}
    engine = None
    def agree(error, stage):
        flag = torch.tensor([int(error is None)], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not flag.item():
            raise RuntimeError(error or f'Another rank rejected {stage}')
    try:
        error = None
        try:
            root = Path(os.environ['TRAINING_SOURCE_ROOT']).resolve(strict=True)
            sys.path[:0] = [str(root/'tests'),str(root)]
            prov = importlib.import_module('provenance')
            ref = importlib.import_module('validate_training_npu')
            timing = importlib.import_module('benchmark_compare')
            package = importlib.import_module('megamoe_training')
            for module in (prov,ref,timing,package):
                if not Path(module.__file__).resolve().is_relative_to(root):
                    raise ValueError('Import escaped frozen root: '+module.__name__)
            if world != 16 or args.B > 16384 or args.E % world or args.K > min(args.E,16):
                raise ValueError('Require legal EP16 dimensions')
            internal = torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
            if isinstance(internal,bytes): internal = internal.decode()
            if torch.npu.matmul.allow_hf32 is not False or internal != 'disable':
                raise ValueError('Strict precision settings not effective')
            if is_p1:
                candidate_root = args.candidate_source_root.resolve(strict=True)
                if (candidate_root.name != 'p1_pointwise' or candidate_root.is_relative_to(root) or
                        root.is_relative_to(candidate_root)):
                    raise ValueError('Require isolated p1_pointwise package outside frozen root')
                sys.path[:0] = [str(candidate_root),str(candidate_root.parent)]
                cp = importlib.import_module('candidate_provenance')
                candidate = importlib.import_module('p1_pointwise')
                if (Path(cp.__file__).resolve() != candidate_root/'candidate_provenance.py' or
                        Path(candidate.__file__).resolve() != candidate_root/'__init__.py'):
                    raise ValueError('Candidate import escaped selected root')
            elif args.backend == 'cann_ai_cpu':
                cann = importlib.import_module('cann_training')
                if not Path(cann.__file__).resolve().is_relative_to(root):
                    raise ValueError('CANN import escaped frozen root')
        except Exception as exc:
            error = repr(exc)
        agree(error,'imports and configuration')
        shape = {k:getattr(args,k) for k in ('B','H','I','E','K')}
        if is_p1:
            execution = dict(cp.FIXED,world=world,shape=shape,seed=args.seed,
                weight_dtype='torch.bfloat16',shared=args.shared,cases=['balanced'],
                oracle_chunk=1024,max_reference_gib=12.0,limits=ref.LIMITS)
            identity, _ = cp.collective_checked(torch,dist,
                lambda: cp.collect_identity(candidate_root,args.pointwise_library,execution,prov),
                'P1 actual source/library identity')
            validate = lambda value: cp.validate_identity(value,prov)
        else:
            identity = prov.collective_identity(torch,dist)
            validate = prov.validate_identity
        report['identity'] = identity
        report['execution'] = {'shape':shape,'seed':args.seed,'weight_dtype':'torch.bfloat16',
            'shared':args.shared,'route':'balanced','backend':args.backend,
            'warmup':args.warmup,'iters':args.iters,'profile_steps':args.profile_steps}
        common = dict(report['execution'],identity=validate(identity),trial=args.trial,
                      mode=report['mode'],harness=report['harness_sha256'])
        digest = hashlib.sha256(json.dumps(common,sort_keys=True).encode()).digest()
        token = torch.tensor(list(digest),device=device,dtype=torch.uint8)
        peers = [torch.empty_like(token) for _ in range(world)]
        dist.all_gather(peers,token)
        agree(None if all(bool(torch.equal(peer,token)) for peer in peers)
              else 'Execution/source/harness differs across ranks','execution agreement')
        error = None
        try:
            report['qualification'] = qualify(args.qualification,args.backend,world,shape,args.seed,
                                              args.shared,identity,ref.LIMITS,validate)
        except Exception as exc:
            error = repr(exc)
        agree(error,'full-gradient qualification')
        report['qualified'] = True
        args.weight_dtype = torch.bfloat16
        x,w1,w2,ids,p,dy = ref.inputs(args,rank,world,'balanced',device)
        shared = ()
        if args.shared:
            gen = torch.Generator(device=x.device).manual_seed(args.seed+9000+rank)
            sw1 = torch.randn((2*args.I,args.H),device=x.device,generator=gen).mul_(1/math.sqrt(args.H)).bfloat16().requires_grad_()
            sw2 = torch.randn((args.H,args.I),device=x.device,generator=gen).mul_(1/math.sqrt(args.I)).bfloat16().requires_grad_()
            shared = (sw1,sw2)
        kwargs = dict(group=dist.group.WORLD,max_tokens=args.B,hidden=args.H,topk=args.K,num_experts=args.E)
        if is_p1:
            engine = candidate.PointwiseTrainingMoE(pointwise_library=args.pointwise_library,**kwargs)
        elif args.backend == 'shmem_catlass':
            engine = package.TrainingMoE(**kwargs)
        else:
            engine = cann.CannTrainingMoE(comm_mode='ai_cpu',**kwargs)
        def step():
            if not torch.is_grad_enabled(): raise RuntimeError('Full training requires autograd')
            ref.clear_grads((x,w1,w2,p)+shared)
            kw = {} if not shared else dict(zip(('shared_w1','shared_w2'),shared))
            y = engine(x,w1,w2,ids,p,**kw)
            y.backward(dy)
            return y
        forward = 6*args.B*args.H*args.I*(args.K+int(args.shared))
        report['effective_GEMM_FLOPs_per_rank'] = {'forward':forward,'full_forward_backward':3*forward}
        if args.benchmark:
            report['timing'] = timing.timing_round_robin({args.backend:step},dist.group.WORLD,args.warmup,args.iters,x.device)
        else:
            from torch_npu.profiler import profile, ProfilerActivity
            from torch.profiler import record_function
            for _ in range(args.warmup): step()
            torch.npu.synchronize(); dist.barrier(); torch.npu.synchronize()
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],record_shapes=True) as prof:
                for _ in range(args.profile_steps):
                    torch.npu.synchronize(); dist.barrier(); torch.npu.synchronize()
                    with record_function('p1_comparison/'+args.backend+'/forward_backward'):
                        step(); torch.npu.synchronize()
            path = args.output/f'{args.backend}_rank{rank}_trace.json'
            prof.export_chrome_trace(str(path))
            report['trace'] = {'path':str(path.resolve()),'sha256':sha(path)}
            report['profile_scope'] = 'alignment outside record_function; step and final synchronization inside; no profiler durations used as formal timing'
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        if engine is not None:
            try: engine.close()
            except Exception as exc: report.setdefault('cleanup_errors',[]).append(repr(exc))
        try: dist.destroy_process_group()
        except Exception as exc: report.setdefault('cleanup_errors',[]).append(repr(exc))
        if report.get('cleanup_errors'): report['passed'] = False
        (args.output/f'perf_rank{rank}.json').write_text(json.dumps(report,indent=2))
        if report.get('cleanup_errors'): raise RuntimeError('Comparison cleanup failed')


if __name__ == '__main__':
    main()
