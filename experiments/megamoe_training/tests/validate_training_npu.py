"""Independent HCCL oracle and gated benchmark for TrainingMoE.

Run under torchrun, never directly on an unreserved environment. No SHMEM oracle
handles are used. This file creates no processes and installs no packages.
All numerical limits below are fixed before execution, not adapted to results.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from provenance import collective_identity

LIMITS = {
    'Y': {'atol': .003, 'rtol': .03, 'rms_relative': .02},
    'dx': {'atol': .003, 'rtol': .03, 'rms_relative': .02},
    'dp': {'atol': .003, 'rtol': .03, 'rms_relative': .02},
    'dW1': {'atol': .003, 'rtol': .03, 'rms_relative': .02},
    'dW2': {'atol': .003, 'rtol': .03, 'rms_relative': .02},
}


def allgather(t):
    values = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(values, t.contiguous())
    return values


@torch.no_grad()
def shared_reference(x,w1,w2,dy):
    """Unweighted shared expert; weights and gradients stay local to each rank."""
    wb1,wb2=w1.bfloat16(),w2.bfloat16()
    g,u=torch.mm(x,wb1.T).bfloat16().chunk(2,dim=-1)
    gf,uf=g.float(),u.float()
    sig=torch.sigmoid(gf)
    a=(gf*sig*uf).bfloat16()
    y=torch.mm(a,wb2.T).bfloat16()
    da=torch.mm(dy,wb2).bfloat16()
    dgu=torch.cat((da.float()*uf*sig*(1+gf*(1-sig)),da.float()*gf*sig),dim=-1).bfloat16()
    return {'Y':y,'dx':torch.mm(dgu,wb1).bfloat16(),
            'dSW1':torch.mm(dgu.float().T,x.float()),'dSW2':torch.mm(dy.float().T,a.float())}


@torch.no_grad()
def independent_oracle(x, w1, w2, ids, p, dy, chunk):
    """Explicit BF16 boundaries; dW is FP32 accumulation of BF16 operands.

    HCCL all_gather is independent of TrainingMoE's SHMEM routing. We process
    one source rank at a time and reduce that source's Y/dx/dp, bounding scratch.
    Oracle sums in FP32 then BF16. Native multi-server combine instead rounds
    server partials to BF16 before the source sum; this independent mathematical
    reference deliberately does not reuse that kernel or its reduction tree.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    xs, routes, ps, dys = map(allgather, (x.detach(), ids, p.detach(), dy))
    dw1 = torch.zeros_like(w1, dtype=torch.float32)
    dw2 = torch.zeros_like(w2, dtype=torch.float32)
    result = None
    counts = [0] * len(w1)
    for src in range(world):
        sx, sp, sd, si = xs[src], ps[src], dys[src], routes[src]
        sy = torch.zeros_like(sx, dtype=torch.float32)
        sdx = torch.zeros_like(sx, dtype=torch.float32)
        sdp = torch.zeros_like(sp, dtype=torch.float32)
        for local_e in range(len(w1)):
            token, slot = torch.where(si == rank * len(w1) + local_e)
            counts[local_e] += token.numel()
            ew1, ew2 = w1[local_e].to(torch.bfloat16), w2[local_e].to(torch.bfloat16)
            for start in range(0, token.numel(), chunk):
                t, k = token[start:start+chunk], slot[start:start+chunk]
                xx, dd = sx[t], sd[t]
                gu = torch.mm(xx, ew1.T).to(torch.bfloat16)
                g, u = gu.chunk(2, dim=-1)
                gf, uf = g.float(), u.float()
                sig = torch.sigmoid(gf)
                silu = gf * sig
                a = (silu * uf).to(torch.bfloat16)
                z = torch.mm(a, ew2.T).to(torch.bfloat16)
                weighted_z = (sp[t,k,None] * z.float()).to(torch.bfloat16)
                sy.index_add_(0, t, weighted_z.float())
                # p=0 is valid: dp still uses the unweighted forward Z.
                sdp[t,k] = (dd.float() * z.float()).sum(dim=-1)
                dz = (sp[t,k,None] * dd.float()).to(torch.bfloat16)
                da = torch.mm(dz, ew2).to(torch.bfloat16)
                dg = da.float() * uf * sig * (1 + gf * (1-sig))
                du = da.float() * silu
                dgu = torch.cat((dg,du), dim=-1).to(torch.bfloat16)
                dxx = torch.mm(dgu, ew1).to(torch.bfloat16)
                sdx.index_add_(0, t, dxx.float())
                dw1[local_e].add_(torch.mm(dgu.float().T, xx.float()))
                dw2[local_e].add_(torch.mm(dz.float().T, a.float()))
        dist.reduce(sy, dst=src)
        dist.reduce(sdx, dst=src)
        dist.reduce(sdp, dst=src)
        if src == rank:
            result = {'Y':sy.to(torch.bfloat16), 'dx':sdx.to(torch.bfloat16), 'dp':sdp}
        del sy, sdx, sdp
    result.update(dW1=dw1, dW2=dw2)
    return result, counts


@torch.no_grad()
def statistics(actual, expected, limit, chunk=1<<20):
    """Every element contributes; no sampling. Chunking bounds extra memory."""
    if actual.shape != expected.shape:
        return {'passed':False,'error':'shape mismatch','actual_shape':list(actual.shape),
                'expected_shape':list(expected.shape)}
    a, r = actual.detach().reshape(-1), expected.detach().reshape(-1)
    finite = True
    bad, absmax, relmax, refmax, errsq, refsq = 0, 0., 0., 0., 0., 0.
    for start in range(0,a.numel(),chunk):
        aa, rr = a[start:start+chunk].float(), r[start:start+chunk].float()
        finite = finite and bool((torch.isfinite(aa).all() & torch.isfinite(rr).all()).item())
        diff = (aa-rr).abs()
        bad += int((diff > limit['atol']+limit['rtol']*rr.abs()).sum().item())
        absmax = max(absmax, float(diff.max().item()))
        relmax = max(relmax, float((diff/rr.abs().clamp_min(1e-8)).max().item()))
        refmax = max(refmax, float(rr.abs().max().item()))
        # NPU float64 is not required. Reduce bounded FP32 chunks on device,
        # then accumulate only scalar summaries using host Python float64.
        errsq += float(diff.square().sum().item())
        refsq += float(rr.square().sum().item())
    rms = math.sqrt(errsq/max(a.numel(),1))
    ref_rms = math.sqrt(refsq/max(a.numel(),1))
    rms_relative = rms/max(ref_rms,1e-12)
    # An identically-zero reference uses absolute tolerance; no arbitrary ratio.
    rms_ok = rms_relative <= limit['rms_relative'] if ref_rms else rms <= limit['atol']
    return {'elements':a.numel(),'actual_dtype':str(actual.dtype),'reference_dtype':str(expected.dtype),
            'finite':finite,'max_abs_error':absmax,'max_relative_error_floor_1e_8':relmax,
            'max_reference_abs':refmax,'RMS_error':rms,'reference_RMS':ref_rms,
            'RMS_relative_error':rms_relative,'out_of_tolerance_elements':bad,
            'limits':limit,'passed':finite and bad==0 and rms_ok}


def inputs(args, rank, world, case, device):
    b,h,i,e,k = args.B,args.H,args.I,args.E,args.K
    gen = torch.Generator(device=device).manual_seed(args.seed+rank)
    x = torch.randn((b,h),generator=gen,device=device,dtype=torch.float32).mul_(.2).bfloat16()
    w1 = torch.randn((e//world,2*i,h),generator=gen,device=device).mul_(1/math.sqrt(h)).to(args.weight_dtype)
    w2 = torch.randn((e//world,h,i),generator=gen,device=device).mul_(1/math.sqrt(i)).to(args.weight_dtype)
    flat = torch.arange(b*k,device=device).reshape(b,k) + rank*b*k
    # Spread one token across rank/server ownership, rather than assigning all
    # eight slots to the same host. For final E64/K8 this remains exactly balanced.
    if e%k==0:
        base=torch.arange(b,device=device)[:,None]+rank*b
        ids=(base+torch.arange(k,device=device)[None,:]*(e//k)).remainder(e)
    else:
        ids=flat.remainder(e)
    if case == 'empty':
        ids = flat.remainder(e-1)  # Last global expert receives no rows.
    elif case == 'skew':
        ids = torch.arange(k,device=device).expand(b,k).contiguous()  # All tokens -> first K experts.
    p = torch.rand((b,k),generator=gen,device=device).add_(.1)
    p = (p/p.sum(dim=-1,keepdim=True)*2.5).float()
    if case == 'zero_gate':
        p[:,0] = 0
    dy = torch.randn((b,h),generator=gen,device=device).mul_(.1).bfloat16()
    return x.requires_grad_(),w1.requires_grad_(),w2.requires_grad_(),ids.long(),p.requires_grad_(),dy


def reference_memory_reservation(args,world):
    """Conservative additive tensor reservation, not total NPU peak memory.

    Shared reference and routed all-gather occur sequentially; summing their
    reservations intentionally overestimates simultaneous reference liveness.
    Native SHMEM/HCCL/workspace, routed candidate saved tensors/temporaries,
    allocator fragmentation, framework state and optimizer states are excluded.
    """
    b,h,i,e,k=args.B,args.H,args.I,args.E,args.K
    weights=(e//world)*3*h*i
    parts={
        'routed_gathered_inputs':world*(4*b*h+12*b*k),
        'routed_weight_state_existing_reservation':weights*14,
        'routed_outputs_existing_reservation':16*b*h+16*b*k,
        'routed_chunk_scratch':args.oracle_chunk*(16*h+40*i),
    }
    if getattr(args,'shared',False):
        p=3*h*i
        fp32=str(getattr(args,'weight_dtype','bf16')) in ('fp32','torch.float32')
        storage=4 if fp32 else 2
        parts.update({
            # FP32 leaf parameters ARE the master; no second master is invented.
            'shared_parameter_or_master':p*storage,
            'shared_bf16_compute_cast':p*2 if fp32 else 0,
            'shared_actual_fp32_wgrad':p*4,
            'shared_leaf_grad_reservation':p*storage,
            'shared_oracle_fp32_wgrad':p*4,
            # X/Z/dX BF16 plus GU/A/dA/dGU BF16. X aliases the existing source
            # input; counting it again is conservative, as are overlapping stages.
            'shared_bf16_saved_outputs_and_dgrad':b*(6*h+12*i),
            # Full-row shared reference is not oracle_chunk limited. Reserve
            # 3 FP32 [B,H] plus 16 FP32 [B,I] intermediates, covering operand
            # casts, sigmoid/derivative chains and FP32 concat/matmul operands.
            'shared_fp32_temporaries':b*(12*h+64*i),
        })
    return {'parts_bytes':parts,'total_bytes':sum(parts.values()),
            'total_GiB':sum(parts.values())/2**30}


def estimate_reference_gib(args,world):
    return reference_memory_reservation(args,world)['total_GiB']


def shared_precision_shape_allowed(args,world):
    # Preserve all previously allowed small shared cases. Only the requested
    # large balanced EP16 shape is newly admitted, bounding routed receive rows.
    return (args.B*args.H<=1024*1024 or
            ((args.B,args.H,args.I,args.E,args.K,world)==(8192,6144,2048,64,8,16)
             and args.cases=='balanced'))


def jsonable(value):
    if isinstance(value,torch.Tensor):
        if value.numel()>1024:
            return {'shape':list(value.shape),'dtype':str(value.dtype),'omitted':'metadata tensor exceeds 1024 elements'}
        return value.detach().cpu().tolist()
    if isinstance(value,dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value,(int,float,str,bool)) or value is None:
        return value
    return repr(value)


def clear_grads(tensors):
    for t in tensors:
        t.grad=None


def benchmark(engine, data, warmup, iters, shared=None):
    x,w1,w2,ids,p,dy=data
    def step():
        clear_grads((x,w1,w2,p)+(() if shared is None else shared))
        engine(x,w1,w2,ids,p,**({} if shared is None else dict(zip(('shared_w1','shared_w2'),shared)))).backward(dy)
    for _ in range(warmup):
        step()
    torch.npu.synchronize()
    dist.barrier()
    ms=[]
    for _ in range(iters):
        torch.npu.synchronize()
        start=time.perf_counter()
        step()
        torch.npu.synchronize()
        ms.append((time.perf_counter()-start)*1000)
    local=torch.tensor(ms,device=x.device,dtype=torch.float32)
    times=allgather(local)
    stacked=torch.stack(times)
    rank_ms=stacked.cpu().tolist()
    maxrank_ms=stacked.max(dim=0).values.cpu().tolist()
    return {'mode':'full forward+backward synchronized wall-clock; gradients reset each iteration',
            'warmup':warmup,'iterations':iters,'rank_ms':rank_ms,'maxrank_ms':maxrank_ms,
            'maxrank_mean_ms':sum(maxrank_ms)/len(maxrank_ms),
            'maxrank_p50_ms':float(torch.tensor(maxrank_ms).median())}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--B',type=int,default=32); ap.add_argument('--H',type=int,default=128)
    ap.add_argument('--I',type=int,default=128); ap.add_argument('--E',type=int,default=0)
    ap.add_argument('--K',type=int,default=4)
    ap.add_argument('--cases',default='balanced,zero_gate,empty,skew')
    ap.add_argument('--weight-dtype',choices=['bf16','fp32'],default='bf16')
    ap.add_argument('--shared',action='store_true',help='Unweighted shared precision: small cases or balanced EP16 target shape')
    ap.add_argument('--backend',choices=['shmem_catlass','cann_ai_cpu','cann_ccu'],default='shmem_catlass')
    ap.add_argument('--oracle-chunk',type=int,default=1024)
    ap.add_argument('--max-reference-gib',type=float,default=12.)
    ap.add_argument('--benchmark',action='store_true')
    ap.add_argument('--warmup',type=int,default=5); ap.add_argument('--iters',type=int,default=30)
    ap.add_argument('--seed',type=int,default=4317)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    import torch_npu  # noqa: F401 -- registers NPU backend
    local_rank=int(os.environ.get('LOCAL_RANK',0))
    torch.npu.set_device(local_rank)
    # Source API: Ascend/pytorch torch_npu/npu/npu_config.py, _allowHF32Matmul.
    # Fail explicitly if this deployed PTA cannot honor the oracle precision.
    if not hasattr(torch.npu,'matmul'):
        raise RuntimeError('PTA lacks torch.npu.matmul HF32 precision control')
    torch.npu.matmul.allow_hf32=False
    if torch.npu.matmul.allow_hf32:
        raise RuntimeError('Could not disable matmul HF32')
    torch.npu.config.allow_internal_format=False
    dist.init_process_group('hccl')
    rank,world=dist.get_rank(),dist.get_world_size()
    args.E=args.E or 4*world
    if args.shared and args.backend!='shmem_catlass':
        raise ValueError('--shared is implemented only for shmem_catlass candidate')
    if args.shared and not shared_precision_shape_allowed(args,world):
        raise ValueError('Large --shared precision requires balanced EP16 B8192 H6144 I2048 E64 K8')
    if args.shared and args.benchmark:
        raise ValueError('--shared is restricted to precision validation; omit --benchmark')
    if not (0<args.B<=16384 and args.E%world==0 and 0<args.K<=args.E):
        raise ValueError('Requires 0<B<=16384 (HT source limit), E%EP=0, 0<K<=E')
    if 'empty' in args.cases.split(',') and args.K>=args.E:
        raise ValueError('empty case requires K<E to preserve unique top-k indices')
    if min(args.H,args.I,args.oracle_chunk,args.iters)<=0 or args.warmup<0:
        raise ValueError('Invalid size or iteration count')
    known={'balanced','zero_gate','empty','skew'}
    cases=args.cases.split(',')
    if set(cases)-known:
        raise ValueError('Unknown cases')
    estimate=estimate_reference_gib(args,world)
    if estimate>args.max_reference_gib:
        raise MemoryError(f'Estimated reference footprint {estimate:.2f} GiB exceeds preflight cap {args.max_reference_gib:.2f} GiB')
    args.weight_dtype={'bf16':torch.bfloat16,'fp32':torch.float32}[args.weight_dtype]
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from megamoe_training import TrainingMoE
    def make_engine():
        kwargs=dict(group=dist.group.WORLD,max_tokens=args.B,hidden=args.H,topk=args.K,num_experts=args.E)
        if args.backend=='shmem_catlass':
            return TrainingMoE(**kwargs)
        from cann_training import CannTrainingMoE
        return CannTrainingMoE(**kwargs,comm_mode={'cann_ai_cpu':'ai_cpu','cann_ccu':'ccu'}[args.backend])
    args.output.mkdir(parents=True,exist_ok=True)
    report={'rank':rank,'world':world,'backend':args.backend,'shape':{'B':args.B,'H':args.H,'I':args.I,'E':args.E,'K':args.K},
            'backend_scope':('P0 separate SHMEM+CATLASS' if args.backend=='shmem_catlass' else 'CANN fused communication/GMM forward+dgrad, separate activation/pack/unpack, same CATLASS FP32 wgrad; unqualified until this report passes'),
            'weight_dtype':str(args.weight_dtype),'seed':args.seed,'limits':LIMITS,'shared':args.shared,
            'oracle_precision':{'matmul_allow_hf32':False,'allow_internal_format':False,'RMS_reduction':'device FP32 chunks, host float64 scalar accumulation'},
            'reference_estimated_GiB':estimate,'reference_chunk':args.oracle_chunk,
            'reference_memory_reservation':reference_memory_reservation(args,world),
            'scope':'NPU candidate versus independent HCCL all-gather manual oracle; no oracle SHMEM handles',
            'balanced_route':'idx=(token+rank*B+slot*(E//K))%E if E%K==0; flat consecutive assignments otherwise',
            'combine_difference':'Reference sums all contributions in FP32 then BF16; native multi-server combine also rounds per-server partials to BF16. Fixed numerical tolerances are not adapted.',
            'cases':[]}
    try:
        report['identity']=collective_identity(torch,dist)
        for case in cases:
            data=inputs(args,rank,world,case,f'npu:{local_rank}')
            x,w1,w2,ids,p,dy=data
            oracle,counts=independent_oracle(*data,args.oracle_chunk)
            shared=None
            if args.shared:
                gen=torch.Generator(device=x.device).manual_seed(args.seed+9000+rank)
                sw1=torch.randn((2*args.I,args.H),device=x.device,generator=gen).mul_(1/math.sqrt(args.H)).to(args.weight_dtype).requires_grad_()
                sw2=torch.randn((args.H,args.I),device=x.device,generator=gen).mul_(1/math.sqrt(args.I)).to(args.weight_dtype).requires_grad_()
                shared=(sw1,sw2)
                sr=shared_reference(x,sw1,sw2,dy)
                oracle['Y']=(oracle['Y']+sr['Y']).bfloat16()
                oracle['dx']=(oracle['dx']+sr['dx']).bfloat16()
                oracle.update({name:sr[name] for name in ('dSW1','dSW2')})
                del sr
            torch.npu.synchronize()
            engine=make_engine()
            try:
                y=engine(x,w1,w2,ids,p,**({} if shared is None else dict(zip(('shared_w1','shared_w2'),shared))))
                y.backward(dy)
                torch.npu.synchronize()
                dw1,dw2=engine.last_grads_fp32
                actual={'Y':y,'dx':x.grad,'dp':p.grad,'dW1':dw1,'dW2':dw2}
                record={'case':case,'oracle_counts':counts,'candidate_counts':jsonable(engine.last_counts),
                        'candidate_shapes':jsonable(engine.last_shapes),'metrics':{}}
                for name in LIMITS:
                    record['metrics'][name]=statistics(actual[name],oracle[name],LIMITS[name])
                record['metrics']['leaf_dW1']=statistics(w1.grad,oracle['dW1'].to(w1.dtype),LIMITS['dW1'])
                record['metrics']['leaf_dW2']=statistics(w2.grad,oracle['dW2'].to(w2.dtype),LIMITS['dW2'])
                if shared is not None:
                    for name,grad,leaf in zip(('dSW1','dSW2'),engine.last_shared_grads_fp32,shared):
                        limit=LIMITS['dW1' if name=='dSW1' else 'dW2']
                        record['metrics'][name]=statistics(grad,oracle[name],limit)
                        record['metrics']['leaf_'+name]=statistics(leaf.grad,oracle[name].to(leaf.dtype),limit)
                record['counts_match']=jsonable(engine.last_counts)==counts
                passed=record['counts_match'] and all(v['passed'] for v in record['metrics'].values())
                gate=torch.tensor([int(passed)],device=x.device,dtype=torch.int32)
                dist.all_reduce(gate,op=dist.ReduceOp.MIN)
                record['all_ranks_numerical_pass']=bool(gate.item())
                report['cases'].append(record)
                (args.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2))
                if not gate.item():
                    raise AssertionError('Numerical gate failed; benchmark is forbidden. See full-rank JSON metrics.')
                del actual,oracle,dw1,dw2,y
                gc.collect(); torch.npu.empty_cache()
            finally:
                engine.close()
            del data,x,w1,w2,ids,p,dy
            if shared is not None:
                del sw1,sw2,shared
            gc.collect(); torch.npu.empty_cache()
        # Every requested precision case on every rank must pass before timing.
        if args.benchmark:
            if args.shared:
                raise ValueError('Shared precision validation excludes timing; omit --benchmark')
            if 'balanced' not in cases:
                raise ValueError('Benchmark requires balanced precision case')
            data=inputs(args,rank,world,'balanced',f'npu:{local_rank}')
            engine=make_engine()
            try:
                report['benchmark']=benchmark(engine,data,args.warmup,args.iters)
            finally:
                engine.close()
        report['all_passed']=True
    except Exception as exc:
        report['all_passed']=False
        report['error']=repr(exc)
        raise
    finally:
        (args.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2))
        dist.destroy_process_group()


if __name__=='__main__':
    main()
