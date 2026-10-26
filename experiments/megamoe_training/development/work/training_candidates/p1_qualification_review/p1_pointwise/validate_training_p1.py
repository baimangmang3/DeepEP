"""Independent P1 full-gradient accuracy entry; no timing/profiler/perf path.

Explicit roots must identify frozen P0 and this p1_pointwise package. Run only
after separate authorization and standalone pointwise qualification. No automatic
device execution: omit --run-npu to print the plan before importing Torch.
"""
import argparse
import gc
import importlib
import json
import math
import os
from pathlib import Path
import sys
from candidate_provenance import FIXED,collect_identity,collective_checked,finish_identity_host


def roots_overlap(frozen,root):
    return frozen==root or frozen.is_relative_to(root) or root.is_relative_to(frozen)


def check_dtype_names(actual,weight_dtype,shared):
    expected={'Y':'torch.bfloat16','dx':'torch.bfloat16','dp':'torch.float32',
              'dW1':'torch.float32','dW2':'torch.float32',
              'leaf_dW1':weight_dtype,'leaf_dW2':weight_dtype}
    if shared:
        expected.update(dSW1='torch.float32',dSW2='torch.float32',
                        leaf_dSW1=weight_dtype,leaf_dSW2=weight_dtype)
    return actual==expected


def finish_report(report,destroy,save):
    """Destruction is part of success. Always persist its failure after cleanup."""
    cleanup_error=None
    try:destroy()
    except Exception as exc:
        cleanup_error=exc;report['cleanup_error']=repr(exc);report['all_passed']=False
    save()
    if cleanup_error is not None:raise RuntimeError('Process-group cleanup failed') from cleanup_error


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-npu',action='store_true')
    ap.add_argument('--training-source-root',type=Path,default=os.getenv('TRAINING_SOURCE_ROOT'))
    ap.add_argument('--candidate-source-root',type=Path,default=os.getenv('CANDIDATE_SOURCE_ROOT'))
    ap.add_argument('--pointwise-library',type=Path,required=True)
    ap.add_argument('--world-size',type=int,required=True)
    for key,default in [('B',32),('H',128),('I',128),('K',4)]:ap.add_argument('--'+key,type=int,default=default)
    ap.add_argument('--E',type=int,required=True)
    ap.add_argument('--cases',default='balanced,zero_gate,empty,skew')
    ap.add_argument('--shared',action='store_true')
    ap.add_argument('--weight-dtype',choices=['bf16','fp32'],default='bf16')
    ap.add_argument('--seed',type=int,default=4317)
    ap.add_argument('--oracle-chunk',type=int,default=1024)
    ap.add_argument('--max-reference-gib',type=float,default=12.)
    ap.add_argument('--output',type=Path,required=True)
    return ap.parse_args()


def prepare(args,world,torch,torch_npu):
    if args.training_source_root is None or args.candidate_source_root is None:
        raise ValueError('Explicit TRAINING_SOURCE_ROOT and CANDIDATE_SOURCE_ROOT required')
    frozen=args.training_source_root.resolve(strict=True);root=args.candidate_source_root.resolve(strict=True)
    if root!=Path(__file__).resolve().parent or root.name!='p1_pointwise':
        raise ValueError('CANDIDATE_SOURCE_ROOT must be this isolated p1_pointwise namespace')
    for file in ('tests/provenance.py','tests/validate_training_npu.py','megamoe_training/training.py'):
        if not (frozen/file).is_file():raise ValueError('Incomplete frozen P0 root: '+file)
    if roots_overlap(frozen,root):raise ValueError('Frozen P0 root overlaps candidate in either direction')
    sys.path[:0]=[str(frozen/'tests'),str(frozen),str(root.parent)]
    base=importlib.import_module('provenance');ref=importlib.import_module('validate_training_npu')
    candidate=importlib.import_module('p1_pointwise')
    for module,path in ((base,frozen/'tests/provenance.py'),(ref,frozen/'tests/validate_training_npu.py'),
                        (candidate,root/'__init__.py'),(importlib.import_module('megamoe_training'),frozen/'megamoe_training/__init__.py')):
        if Path(module.__file__).resolve()!=path:raise ValueError('Imported namespace escaped explicit source root')
    if args.world_size!=world or world not in (2,16):raise ValueError('Explicit world-size must match EP2/EP16 process group')
    if not (0<args.B<=16384 and args.E>0 and args.E%world==0 and 0<args.K<=min(16,args.E)):
        raise ValueError('Invalid HT B/E/K/world ABI')
    if any(n<=0 or n%128 or n>2**20 for n in (args.H,args.I)):
        raise ValueError('P1 requires native H/I multiples of128, <=2**20')
    cases=args.cases.split(',')
    if not cases or len(set(cases))!=len(cases) or set(cases)-{'balanced','zero_gate','empty','skew'}:
        raise ValueError('Unknown/duplicate cases')
    if 'empty' in cases and args.K>=args.E:raise ValueError('Empty case requires K<E')
    if args.oracle_chunk<=0 or not math.isfinite(args.max_reference_gib) or args.max_reference_gib<=0:
        raise ValueError('Invalid oracle reservation')
    if args.shared and not ref.shared_precision_shape_allowed(args,world):
        raise ValueError('Shared shapes restricted exactly as frozen P0')
    reservation=ref.reference_memory_reservation(args,world)
    if reservation['total_GiB']>args.max_reference_gib:raise MemoryError('Frozen reference memory estimate exceeds explicit cap')
    torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
    internal=torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
    if isinstance(internal,bytes):internal=internal.decode()
    if torch.npu.matmul.allow_hf32 is not False or internal!='disable':raise ValueError('Strict precision readback failed')
    execution=dict(FIXED,world=world,shape={key:getattr(args,key) for key in ('B','H','I','E','K')},
                   seed=args.seed,weight_dtype='torch.bfloat16' if args.weight_dtype=='bf16' else 'torch.float32',
                   shared=args.shared,cases=cases,oracle_chunk=args.oracle_chunk,
                   max_reference_gib=args.max_reference_gib,limits=ref.LIMITS)
    return base,ref,candidate,root,reservation,execution


def check_shapes(args,world,counts,shapes):
    rows=sum(counts);h,i=args.H,args.I
    expected={'x':[args.B,h],'xe':[rows,h],'w1':[args.E//world,2*i,h],
              'gu':[rows,2*i],'a':[rows,i],'w2':[args.E//world,h,i],'z':[rows,h],'y':[args.B,h]}
    return len(counts)==args.E//world and shapes==expected


def phase(name,rank=None,**fields):
    """Best-effort flushed diagnostics; logging must not skip host convergence."""
    import json,os,time
    if rank is None:rank=os.environ.get('RANK','unknown')
    record=dict(phase=name,rank=rank,time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),**fields)
    try:print(json.dumps(record),flush=True)
    except OSError:pass


def close_case_host(engine,host,report,case,prior_error=None):
    """Host-only error convergence, not a repair of SHMEM finalization."""
    close_error=None
    phase('case_close_before',host.rank(),case=case)
    try:engine.close()
    except Exception as exc:close_error=exc
    phase('case_close_after',host.rank(),case=case,passed=close_error is None)
    case_index=len(report.get('case_cleanup',[]))
    stage='P1/case-close/'+str(case_index)+'/'+case
    local={'rank':host.rank(),'case':case,'case_index':case_index,'stage':stage,
           'close_error':None if close_error is None else repr(close_error),
           'body_error':None if prior_error is None else repr(prior_error)}
    item=dict(local);report.setdefault('case_cleanup',[]).append(item)
    try:
        peers=host.all_gather(local,stage)
        if ([value.get('rank') for value in peers]!=list(range(host.size())) or
                any(value.get('stage')!=stage or value.get('case')!=case for value in peers)):
            raise RuntimeError('Host close status payload/stage mismatch')
        item['peer_status']=peers
        item['all_ranks_closed']=not any(value['close_error'] for value in peers)
        item['all_ranks_case_passed']=not any(value['close_error'] or value['body_error'] for value in peers)
        phase('case_close_host_status',host.rank(),case=case,passed=item['all_ranks_case_passed'])
    except Exception as exc:
        report['all_passed']=False;item['host_exchange_error']=repr(exc)
        host.abort('case close status exchange failed: '+repr(exc))
        raise RuntimeError('Host close convergence failed; no next case permitted') from exc
    if not item['all_ranks_case_passed']:
        report['all_passed']=False
        failure=close_error if close_error is not None else prior_error
        raise RuntimeError('A case body/close failed; no next case permitted: '+
            (repr(failure) if failure is not None else 'peer failure; inspect peer_status')) from failure


def finish_host_report(host,report,dist,save,prior_error=None):
    """Keep host coordination alive until all ranks reach PG teardown."""
    errors=[]
    if prior_error is not None:report['all_passed']=False
    local={'rank':host.rank(),'error':None if prior_error is None else repr(prior_error)}
    try:
        peers=host.all_gather(local,'P1/before-PG-teardown')
        report['peer_before_PG_teardown']=peers
        if any(peer['error'] for peer in peers):
            report['all_passed']=False
            errors.append('A peer reported an original pre-teardown error')
    except Exception as exc:errors.append('host pre-teardown: '+repr(exc))
    try:
        if dist.is_initialized():dist.destroy_process_group()
    except Exception as exc:errors.append('PG teardown: '+repr(exc))
    try:
        peers=host.all_gather({'rank':host.rank(),'errors':errors},'P1/after-PG-teardown')
        if any(peer['errors'] for peer in peers):
            report['all_passed']=False
            errors.append('A peer reported PG/host teardown failure')
        report['peer_exit_status']=peers
    except Exception as exc:errors.append('host exit status: '+repr(exc))
    try:host.close()
    except Exception as exc:errors.append('host close: '+repr(exc))
    if errors:report['cleanup_errors']=errors;report['all_passed']=False
    # Report I/O runs after every required host exchange and close attempt.
    # A final I/O error is still nonzero and cannot skip communication above.
    try:save()
    except Exception as exc:
        report['all_passed']=False;report['report_io_error']=repr(exc)
        raise
    if errors:raise RuntimeError('Host/PG cleanup failed; inspect cleanup_errors')



def finish_validation_host(host,report,dist,save,collector,initial,base,prior_error=None):
    """Final identity and host cleanup run even after body failure; first error wins."""
    final_error=finish_identity_host(host,report,collector,initial,base)
    first=prior_error or final_error
    if first is not None:
        report['all_passed']=False;report.setdefault('error',repr(first))
    try:finish_host_report(host,report,dist,save,first)
    except Exception as exc:
        # finish_host_report already attempts all host/PG cleanup and final save.
        return first or exc
    return first


def main():
    args=parse_args()
    if not args.run_npu:
        print(json.dumps(dict(action='plan only, no Torch/NPU import',backend=FIXED['backend'],
                             world=args.world_size,shape={k:getattr(args,k) for k in ('B','H','I','E','K')},
                             shared=args.shared,cases=args.cases,scope='accuracy only; separately authorized device run required')))
        return
    phase('torch_import_before')
    import torch
    import torch_npu
    import torch.distributed as dist
    phase('torch_import_after')
    from deep_ep._store import StoreGroup
    phase('host_connect_before')
    host=StoreGroup.from_env()  # existing torchrun TCPStore; no device/PG transport
    rank,world=host.rank(),host.size()
    phase('host_connect_after',rank,world=world)
    local_rank=int(os.environ['LOCAL_RANK'])
    report={'rank':rank,'world':world,'backend':FIXED['backend'],'all_passed':False,'cases':[]}
    identity=None;base=None
    try:
        phase('set_device_before',rank,device=local_rank)
        torch.npu.set_device(local_rank)
        phase('set_device_after',rank,device=local_rank)
        phase('PG_init_before',rank)
        dist.init_process_group('hccl')
        phase('PG_init_after',rank)
        if (dist.get_rank(),dist.get_world_size())!=(rank,world):
            raise ValueError('Host/ProcessGroup rank-world mismatch')
        collective_checked(torch,dist,lambda:args.output.mkdir(parents=True,exist_ok=True),'report output directory')
        base,ref,candidate,root,reservation,execution=collective_checked(torch,dist,
            lambda:prepare(args,world,torch,torch_npu),'candidate configuration/import')
        # All loaders and configuration errors are exchanged BEFORE any SHMEM session.
        phase('identity_before',rank)
        identity,loaded_pointwise=collective_checked(torch,dist,
            lambda:collect_identity(root,args.pointwise_library,execution,base),'candidate identity/load')
        phase('identity_after',rank,digest=identity['content_digest'])
        report.update(shape=execution['shape'],shared=args.shared,weight_dtype=execution['weight_dtype'],
                      seed=args.seed,limits=ref.LIMITS,execution=execution,identity=identity,
                      reference_memory_reservation=reservation,scope='P1 pointwise replacements, P0 SHMEM/CATLASS, full-gradient accuracy only',
                      combine_difference='Independent FP32 sum oracle; native combine rounds BF16 server partials. Fixed tolerances unchanged.')
        # Prevent rank-specific config/roots/source differences entering collective engines.
        local_digest=torch.tensor(list(bytes.fromhex(identity['content_digest'])),device=f'npu:{local_rank}',dtype=torch.uint8)
        digests=[torch.empty_like(local_digest) for _ in range(world)]
        dist.all_gather(digests,local_digest)
        if not all(torch.equal(peer,local_digest) for peer in digests):
            raise ValueError('Ranks differ in candidate identity or execution configuration')
        args.weight_dtype={'bf16':torch.bfloat16,'fp32':torch.float32}[args.weight_dtype]
        for case in execution['cases']:
            phase('case_start',rank,case=case)
            data=ref.inputs(args,rank,world,case,f'npu:{local_rank}');x,w1,w2,ids,p,dy=data
            oracle,counts=ref.independent_oracle(*data,args.oracle_chunk)
            shared=None
            if args.shared:
                gen=torch.Generator(device=x.device).manual_seed(args.seed+9000+rank)
                sw1=torch.randn((2*args.I,args.H),device=x.device,generator=gen).mul_(1/math.sqrt(args.H)).to(args.weight_dtype).requires_grad_()
                sw2=torch.randn((args.H,args.I),device=x.device,generator=gen).mul_(1/math.sqrt(args.I)).to(args.weight_dtype).requires_grad_()
                shared=(sw1,sw2);sr=ref.shared_reference(x,sw1,sw2,dy)
                oracle['Y']=(oracle['Y']+sr['Y']).bfloat16();oracle['dx']=(oracle['dx']+sr['dx']).bfloat16()
                oracle.update({name:sr[name] for name in ('dSW1','dSW2')});del sr
            torch.npu.synchronize()
            engine=candidate.PointwiseTrainingMoE(pointwise_library=args.pointwise_library,group=dist.group.WORLD,
                max_tokens=args.B,hidden=args.H,topk=args.K,num_experts=args.E,cores=32)
            try:
                assert torch.is_grad_enabled(),'Full training validation requires autograd'
                y=engine(x,w1,w2,ids,p,**({} if shared is None else dict(zip(('shared_w1','shared_w2'),shared))))
                y.backward(dy);torch.npu.synchronize()
                dw1,dw2=engine.last_grads_fp32
                actual={'Y':y,'dx':x.grad,'dp':p.grad,'dW1':dw1,'dW2':dw2}
                item={'case':case,'oracle_counts':counts,'candidate_counts':ref.jsonable(engine.last_counts),
                      'candidate_shapes':ref.jsonable(engine.last_shapes),'metrics':{}}
                dtype_names={name:str(tensor.dtype) for name,tensor in actual.items()}
                dtype_names.update(leaf_dW1=str(w1.grad.dtype),leaf_dW2=str(w2.grad.dtype))
                for name in ref.LIMITS:item['metrics'][name]=ref.statistics(actual[name],oracle[name],ref.LIMITS[name])
                for name,leaf in (('dW1',w1),('dW2',w2)):
                    item['metrics']['leaf_'+name]=ref.statistics(leaf.grad,oracle[name].to(leaf.dtype),ref.LIMITS[name])
                if shared is not None:
                    for name,grad,leaf in zip(('dSW1','dSW2'),engine.last_shared_grads_fp32,shared):
                        dtype_names[name]=str(grad.dtype);dtype_names['leaf_'+name]=str(leaf.grad.dtype)
                        limit=ref.LIMITS['dW1' if name=='dSW1' else 'dW2']
                        item['metrics'][name]=ref.statistics(grad,oracle[name],limit)
                        item['metrics']['leaf_'+name]=ref.statistics(leaf.grad,oracle[name].to(leaf.dtype),limit)
                item['counts_match']=item['candidate_counts']==counts
                item['shapes_match']=check_shapes(args,world,counts,item['candidate_shapes'])
                item['dtypes']=dtype_names
                item['dtypes_match']=check_dtype_names(dtype_names,execution['weight_dtype'],args.shared)
                passed=item['counts_match'] and item['shapes_match'] and item['dtypes_match'] and all(v['passed'] for v in item['metrics'].values())
                gate=torch.tensor([int(passed)],device=x.device,dtype=torch.int32);dist.all_reduce(gate,op=dist.ReduceOp.MIN)
                item['all_ranks_numerical_pass']=bool(gate.item());report['cases'].append(item)
                (args.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2))
                if not gate.item():raise AssertionError('Full-gradient gate failed; no later cases permitted')
                del actual,oracle,dw1,dw2,y
            finally:
                close_case_host(engine,host,report,case,sys.exc_info()[1])
            del data,x,w1,w2,ids,p,dy
            if shared is not None:del sw1,sw2,shared
            gc.collect();torch.npu.empty_cache()
        report['all_passed']=True
    except Exception as exc:
        report['error']=repr(exc);raise
    finally:
        original=sys.exc_info()[1]
        def collect_final():
            if identity is None:raise RuntimeError('Initial P1 identity missing; final qualification unavailable')
            return collect_identity(root,args.pointwise_library,execution,base)[0]
        terminal=finish_validation_host(host,report,dist,
            lambda:(args.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2)),
            collect_final,identity,base,original)
        if original is None and terminal is not None:raise terminal


if __name__=='__main__':main()
