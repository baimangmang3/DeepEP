"""Forward-only ai_cpu / ccu CANN versus P0 SHMEM+CATLASS comparison.

Prepare code only until reserved devices and full precision qualification exist.
Example (both hosts, NODE_RANK 0/1, environment agent supplies MASTER_PORT):
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
 --rdzv-backend=static --master-addr=177.22.27.1 --master-port="$MASTER_PORT" \
 experiments/megamoe_training/tests/benchmark_compare.py --E 64 --K 8 \
 --B 32 --H 128 --I 128 --output /ABS/RESERVED_RUN/forward-compare-small

Only after small precision passes, use B8192 H6144 I2048 E64 K8, --cases
balanced --benchmark --warmup 5 --iters 30. No backward performance is compared.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import torch_npu

from cann_baseline import CannForward,forward_oracle
from validate_training_npu import LIMITS,inputs,statistics,jsonable
from provenance import collective_identity,require_training_qualification


def timing_round_robin(functions,group,warmup,iters,device):
    """Same order across ranks, rotating across paths to reduce order bias."""
    names=list(functions)
    for _ in range(warmup):
        for name in names:
            functions[name]()
    torch.npu.synchronize()
    dist.barrier(group=group)
    local={name:{'npu_event_ms':[],'sync_wall_ms':[]} for name in names}
    for iteration in range(iters):
        order=names[iteration%len(names):]+names[:iteration%len(names)]
        for name in order:
            dist.barrier(group=group)
            torch.npu.synchronize()
            st,en=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            before=time.perf_counter()
            st.record()
            out=functions[name]()
            en.record()
            torch.npu.synchronize()
            wall=(time.perf_counter()-before)*1000
            local[name]['sync_wall_ms'].append(wall)
            local[name]['npu_event_ms'].append(float(st.elapsed_time(en)))
            del out
    world=dist.get_world_size(group)
    result={}
    for name in names:
        result[name]={}
        for metric,values in local[name].items():
            tensor=torch.tensor(values,device=device,dtype=torch.float32)
            peers=[torch.empty_like(tensor) for _ in range(world)]
            dist.all_gather(peers,tensor,group=group)
            allvalues=torch.stack(peers).cpu()
            maximum=allvalues.max(dim=0).values
            result[name][metric]={'per_rank':allvalues.tolist(),'max_rank':maximum.tolist(),
                                 'max_rank_mean':float(maximum.mean()),'max_rank_p50':float(maximum.median())}
    return {'warmup':warmup,'iterations':iters,'order':'rotate one path per iteration',
            'barriers':'before each path, excluded from intervals',
            'event_scope':'current-stream NPU event interval, may include stream idle or communication waits; not pure kernel execution',
            'wall_scope':'synchronized Python call including output creation and device work; Event creation/barrier excluded',
            'paths':result}


def stage_functions(baselines):
    fns={}
    retained=[]
    with torch.no_grad():
        for name,base in baselines.items():
            packed=base.pack()
            gu=base.up(packed)
            act=base.activation(gu)
            retained.extend((packed,gu,act))
            fns[name+'/alltoallv_gmm']=lambda base=base,packed=packed:base.up(packed)
            fns[name+'/FP32_swiglu_to_BF16']=lambda base=base,gu=gu:base.activation(gu)
            fns[name+'/gmm_alltoallv']=lambda base=base,act=act:base.down(act)
    return fns,retained


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name,default in [('B',32),('H',128),('I',128),('E',0),('K',4)]:
        ap.add_argument('--'+name,type=int,default=default)
    ap.add_argument('--cases',default='balanced,zero_gate,empty')
    ap.add_argument('--modes',default='ai_cpu,ccu',help='Explicit modes; unsupported mode fails, never silently omitted')
    ap.add_argument('--weight-dtype',choices=['bf16','fp32'],default='bf16')
    ap.add_argument('--seed',type=int,default=4317)
    ap.add_argument('--oracle-chunk',type=int,default=1024)
    ap.add_argument('--max-reference-gib',type=float,default=12.)
    ap.add_argument('--benchmark',action='store_true')
    ap.add_argument('--training-qualification-dir',type=Path,help='Required for benchmark/profile: all rankN.json full-gradient reports matching world/B/H/I/E/K and BF16 weights')
    ap.add_argument('--cann-operator-scope',action='store_true',help='Separate prepared-input operator intervals; not kernel-only time')
    ap.add_argument('--cann-prepared-plan-scope',action='store_true',help='Separate CANN-only prepared-plan full forward, never a fresh-candidate comparison')
    ap.add_argument('--profile',action='store_true',help='Export per-path CPU/NPU profiler traces after precision gate')
    ap.add_argument('--warmup',type=int,default=5);ap.add_argument('--iters',type=int,default=30)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK',0)))
    torch.npu.matmul.allow_hf32=False
    if torch.npu.matmul.allow_hf32:
        raise RuntimeError('HF32 could not be disabled')
    torch.npu.config.allow_internal_format=False
    dist.init_process_group('hccl')
    rank,world=dist.get_rank(),dist.get_world_size()
    args.E=args.E or 4*world
    if not (0<args.B<=16384 and args.E%world==0 and 0<args.K<=min(args.E,16)):
        raise ValueError('Illegal HT input shape')
    cases=args.cases.split(',');modes=args.modes.split(',')
    if set(cases)-{'balanced','zero_gate','empty','skew'} or not modes or set(modes)-{'ai_cpu','ccu'}:
        raise ValueError('Invalid case or mode; modes must use audited ai_cpu/ccu spelling')
    if 'empty' in cases and args.K>=args.E:
        raise ValueError('Empty case requires K<E')
    if min(args.oracle_chunk,args.H,args.I,args.iters)<=0 or args.warmup<0:
        raise ValueError('Invalid dimension/iteration count')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    identity=collective_identity(torch,dist)
    qualification=[]
    if args.benchmark or args.profile:
        # All hosts must see all rank files; exchange the pass/fail flag so one
        # host cannot continue into collectives while a peer rejects its files.
        qualification_error=None
        try:
            qualification=require_training_qualification(args.training_qualification_dir,world,
                {key:getattr(args,key) for key in ('B','H','I','E','K')},identity)
        except Exception as exc:
            qualification_error=repr(exc)
        gate=torch.tensor([int(qualification_error is None)],device=f'npu:{int(os.environ.get("LOCAL_RANK",0))}',dtype=torch.int32)
        dist.all_reduce(gate,op=dist.ReduceOp.MIN)
        if not gate.item():
            raise ValueError(qualification_error or 'Another rank rejected full-gradient qualification; distribute matching rank0..world-1.json on every host')
    estimate=(world*(2*args.B*args.H+12*args.B*args.K)+args.B*args.K*args.H*8+
              4*args.B*args.H+args.E//world*3*args.H*args.I*8)/2**30
    if estimate>args.max_reference_gib:
        raise MemoryError(f'Rough reference footprint {estimate:.2f} GiB > cap {args.max_reference_gib}')
    args.weight_dtype={'bf16':torch.bfloat16,'fp32':torch.float32}[args.weight_dtype]
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from megamoe_training import TrainingMoE
    args.output.mkdir(parents=True,exist_ok=True)
    report={'rank':rank,'world':world,'shape':{key:getattr(args,key) for key in ('B','H','I','E','K')},
            'scope':'forward only; equal fixed inputs, BF16 compute, original FP32 gates; no backward timing',
            'candidate':'P0 separate SHMEM communication + CATLASS GMM; not final fused MegaMoE',
            'baseline':'CANN fused alltoallvGmm + separate FP32 SwiGLU/BF16 cast + fused gmmAlltoallv; separate pack/unpack/gating/topk',
            'modes_requested':modes,'limits':LIMITS['Y'],'oracle':'independent full-output HCCL all_gather + expert torch.mm; no SHMEM/fused-op reuse',
            'combine_rounding':'Oracle/CANN source FP32 sum->BF16; candidate native hierarchy additionally rounds server partials to BF16',
            'routing_plan':'Dynamic comparison refreshes CANN bincount/all_gather/host counts/CPU sort/order H2D each call and includes candidate fresh SHMEM routing; prepared-plan CANN scope is separate, no speedup versus fresh candidate',
            'weight_materialization':'BF16 inputs avoid baseline-only precomputed cast; FP32 parameter mode has baseline one-time BF16 cast and is precision-only unless scope is explicitly revisited',
            'precision':{'HF32':False,'internal_format':False},'reference_estimated_GiB':estimate,'cases':[]}
    report['training_qualification']=qualification
    report['identity']=identity
    ready=None
    try:
        for case in cases:
            data=inputs(args,rank,world,case,f'npu:{int(os.environ.get("LOCAL_RANK",0))}')
            x,w1,w2,ids,p,dy=data
            # Disable graph construction consistently: this compares forward only.
            x,w1,w2,p=(t.detach() for t in (x,w1,w2,p))
            ref=forward_oracle(x,w1,w2,ids,p,args.oracle_chunk)
            bases={mode:CannForward(x,w1,w2,ids,p,comm_mode=mode) for mode in modes}
            engine=TrainingMoE(group=dist.group.WORLD,max_tokens=args.B,hidden=args.H,
                               topk=args.K,num_experts=args.E)
            candidate=lambda:engine(x,w1,w2,ids,p)
            functions={**{'cann_'+mode:base for mode,base in bases.items()},'candidate_P0_forward':candidate}
            try:
                entry={'case':case,'baseline_metadata':{mode:base.metadata for mode,base in bases.items()},'validation':{}}
                with torch.no_grad():
                    for name,fn in functions.items():
                        actual=fn();torch.npu.synchronize()
                        entry['validation'][name]=statistics(actual,ref,LIMITS['Y'])
                        del actual
                entry['candidate_counts']=jsonable(engine.last_counts)
                entry['candidate_shapes']=jsonable(engine.last_shapes)
                expected_counts=next(iter(bases.values())).expert_counts
                entry['counts_match']=entry['candidate_counts']==expected_counts
                gate=torch.tensor([int(entry['counts_match'] and all(v['passed'] for v in entry['validation'].values()))],device=x.device,dtype=torch.int32)
                dist.all_reduce(gate,op=dist.ReduceOp.MIN)
                entry['all_ranks_precision_pass']=bool(gate.item())
                report['cases'].append(entry)
                (args.output/f'compare_rank{rank}.json').write_text(json.dumps(report,indent=2))
                if not gate.item():
                    raise AssertionError('Forward precision failed on at least one rank; no performance runs permitted')
                # Retain balanced shared inputs only after this case passes.
                if case=='balanced':
                    ready=(x,w1,w2,ids,p)
            finally:
                engine.close()
            del ref,functions,bases,data,dy,engine,candidate
            gc.collect();torch.npu.empty_cache()
        if args.benchmark or args.profile:
            if args.weight_dtype!=torch.bfloat16:
                raise ValueError('Forward performance comparison requires BF16 parameters for matched cast scope; FP32 weights are precision-only')
            if ready is None:
                raise ValueError('Timing/profiling requires a validated balanced case')
            x,w1,w2,ids,p=ready
            bases={mode:CannForward(x,w1,w2,ids,p,comm_mode=mode) for mode in modes}
            engine=TrainingMoE(group=dist.group.WORLD,max_tokens=args.B,hidden=args.H,
                               topk=args.K,num_experts=args.E)
            functions={**{'cann_'+mode:base for mode,base in bases.items()},'candidate_P0_forward':lambda:engine(x,w1,w2,ids,p)}
            try:
                with torch.no_grad():
                    if args.benchmark:
                        report['dynamic_routing_full_forward']=timing_round_robin(functions,dist.group.WORLD,args.warmup,args.iters,x.device)
                        if args.cann_prepared_plan_scope:
                            prepared={name+'/prepared_plan':base.prepared_forward for name,base in bases.items()}
                            report['CANN_only_prepared_plan_forward']=timing_round_robin(prepared,dist.group.WORLD,args.warmup,args.iters,x.device)
                        if args.cann_operator_scope:
                            ops,retained=stage_functions(bases)
                            report['CANN_prepared_input_operator_scope']=timing_round_robin(ops,dist.group.WORLD,args.warmup,args.iters,x.device)
                            del ops,retained
                    if args.profile:
                        from torch_npu.profiler import profile,ProfilerActivity
                        report['kernel_attribution_scope']='Profiler traces; do not equate prepared-input event intervals with kernel-only time'
                        for name,fn in functions.items():
                            for _ in range(3):fn()
                            torch.npu.synchronize();dist.barrier()
                            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],record_shapes=True) as prof:
                                for _ in range(3):fn()
                                torch.npu.synchronize()
                            prof.export_chrome_trace(str(args.output/f'{name}_rank{rank}_trace.json'))
            finally:
                engine.close()
        report['all_passed']=True
    except Exception as exc:
        report['all_passed']=False;report['error']=repr(exc)
        raise
    finally:
        (args.output/f'compare_rank{rank}.json').write_text(json.dumps(report,indent=2))
        dist.destroy_process_group()


if __name__=='__main__':
    main()
