"""Accuracy only, isolated original EP<=8 or adapted EXACT16 UDMA training.

Default plan imports no Torch. No timing, optimizer, fallback, worker launcher
or retry. Explicit run requires reserved devices and audited wheel contract.
"""
import argparse
import gc
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from qualification import check_config,check_contract,collect,require_module_paths,required_module_paths
from host_exit import finish_host_report


def phase(name,rank,**kw):
    try:print(json.dumps(dict(phase=name,rank=rank,time_ns=time.time_ns(),**kw)),flush=True)
    except OSError:pass


def host_stage(name,action,host,report,save,record=None):
    phase(name+'_before',host.rank());error=None
    try:action()
    except Exception as exc:error=exc;report.setdefault('first_exception',dict(stage=name,error=repr(exc)))
    item={'stage':name,'error':None if error is None else repr(error)};report['stages'].append(item)
    try:save()
    except Exception as exc:
        item['report_io_error']=repr(exc)
        if error is None:error=exc
    try:
        peers=host.all_gather(dict(rank=host.rank(),stage=name,error=None if error is None else repr(error)),'EP-GMM/'+name)
        item['peer_status']=peers
        ok=len(peers)==host.size() and [v.get('rank') for v in peers]==list(range(host.size())) and all(v.get('stage')==name and v.get('error') is None for v in peers)
    except Exception as exc:item['host_error']=repr(exc);error=error or exc;ok=False
    if record is not None:record['all_ranks_numerical_pass']=ok
    phase(name+'_after',host.rank(),passed=ok)
    if not ok:
        report['all_passed']=False
        raise RuntimeError('All-rank precision/lifecycle gate failed: '+name) from error


def cleanup_owner(engine,runtime,init_attempted,host,report):
    """Always try owned teardown, even after adapter.close or init errors."""
    errors=[]
    for name,action in [('adapter.close',lambda:None if engine is None else engine.close()),
                        ('UDMA.destroy',lambda:None if not init_attempted else runtime.destroy_udma_group())]:
        error=None
        try:action()
        except Exception as exc:error=repr(exc);errors.append(name+': '+error)
        item=dict(stage=name,rank=host.rank(),attempted=engine is not None if name=='adapter.close' else init_attempted,error=error)
        report.setdefault('owner_cleanup',[]).append(item)
        try:
            peers=host.all_gather(item,'EP-GMM/cleanup/'+name);item['peer_status']=peers
            if any(peer['error'] for peer in peers):errors.append('peer '+name+' failed')
        except Exception as exc:errors.append('host '+name+': '+repr(exc))
    if errors:
        report['all_passed']=False;report['owner_cleanup_errors']=errors
        raise RuntimeError('Owned adapter/UDMA teardown failed: '+repr(errors))


def run_case(args,ref,engine,rank,world,device,case,report,item):
    import torch
    data=ref.inputs(args,rank,world,case,device);x,w1,w2,ids,p,dy=data
    expected,counts=ref.independent_oracle(*data,args.oracle_chunk);shared=None
    if args.shared:
        gen=torch.Generator(device=x.device).manual_seed(args.seed+9000+rank)
        sw1=torch.randn((2*args.I,args.H),device=x.device,generator=gen).mul_(1/math.sqrt(args.H)).to(args.weight_dtype).requires_grad_()
        sw2=torch.randn((args.H,args.I),device=x.device,generator=gen).mul_(1/math.sqrt(args.I)).to(args.weight_dtype).requires_grad_()
        shared=(sw1,sw2);sr=ref.shared_reference(x,sw1,sw2,dy)
        expected['Y']=(expected['Y']+sr['Y']).bfloat16();expected['dx']=(expected['dx']+sr['dx']).bfloat16()
        expected.update({n:sr[n] for n in ('dSW1','dSW2')});del sr
    assert torch.is_grad_enabled(),'Real Function/autograd is required'
    y=engine(x,w1,w2,ids,p,**({} if shared is None else dict(zip(('shared_w1','shared_w2'),shared))))
    y.backward(dy);torch.npu.synchronize();dw1,dw2=engine.last_grads_fp32
    actual=dict(Y=y,dx=x.grad,dp=p.grad,dW1=dw1,dW2=dw2)
    item.update(case=case,oracle_counts=counts,candidate_counts=ref.jsonable(engine.last_counts),candidate_shapes=ref.jsonable(engine.last_shapes),metrics={})
    dtypes={n:str(v.dtype) for n,v in actual.items()};wanted={n:'torch.float32' if n in ('dp','dW1','dW2') else 'torch.bfloat16' for n in actual}
    for name in ref.LIMITS:item['metrics'][name]=ref.statistics(actual[name],expected[name],ref.LIMITS[name])
    for name,leaf in (('dW1',w1),('dW2',w2)):
        item['metrics']['leaf_'+name]=ref.statistics(leaf.grad,expected[name].to(leaf.dtype),ref.LIMITS[name]);dtypes['leaf_'+name]=str(leaf.grad.dtype);wanted['leaf_'+name]=str(leaf.dtype)
    if shared:
        for name,grad,leaf in zip(('dSW1','dSW2'),engine.last_shared_grads_fp32,shared):
            limit=ref.LIMITS['dW1' if name=='dSW1' else 'dW2'];item['metrics'][name]=ref.statistics(grad,expected[name],limit)
            item['metrics']['leaf_'+name]=ref.statistics(leaf.grad,expected[name].to(leaf.dtype),limit)
            dtypes[name]=str(grad.dtype);wanted[name]='torch.float32';dtypes['leaf_'+name]=str(leaf.grad.dtype);wanted['leaf_'+name]=str(leaf.dtype)
    rows=sum(counts);h,i,b=args.H,args.I,args.B
    shapes={'x':[b,h],'xe':[rows,h],'w1':[args.E//world,2*i,h],'gu':[rows,2*i],'a':[rows,i],
            'w2':[args.E//world,h,i],'packed_z':[b*args.K,h],'y':[b,h]}
    item.update(counts_match=item['candidate_counts']==counts,shapes_match=item['candidate_shapes']==shapes,dtypes=dtypes,dtypes_match=dtypes==wanted)
    item['local_numerical_pass']=item['counts_match'] and item['shapes_match'] and item['dtypes_match'] and all(v['passed'] for v in item['metrics'].values())
    report['cases'].append(item)
    # Release reporting references after full metrics; one engine/domain persists.
    del actual,expected,dw1,dw2,y,data,x,w1,w2,ids,p,dy
    if shared:del sw1,sw2,shared
    gc.collect();torch.npu.empty_cache()
    if not item['local_numerical_pass']:raise AssertionError('Fixed full-gradient numerical gate failed')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-npu',action='store_true')
    ap.add_argument('--variant',choices=['ep8_original','ep16_adapted'],default='ep8_original')
    ap.add_argument('--world-size',type=int)
    for key,value in [('B',32),('H',128),('I',128),('E',0),('K',4)]:ap.add_argument('--'+key,type=int,default=value)
    ap.add_argument('--training-source-root',type=Path,default=os.getenv('TRAINING_SOURCE_ROOT'))
    ap.add_argument('--candidate-parent',type=Path);ap.add_argument('--build-contract',type=Path);ap.add_argument('--output',type=Path)
    ap.add_argument('--cases',default='balanced,zero_gate,empty,skew');ap.add_argument('--shared',action='store_true')
    ap.add_argument('--weight-dtype',choices=['bf16','fp32'],default='bf16');ap.add_argument('--seed',type=int,default=4317)
    ap.add_argument('--oracle-chunk',type=int,default=1024);ap.add_argument('--max-reference-gib',type=float,default=12.)
    ap.add_argument('--heap-size',type=int,default=1<<30);a=ap.parse_args()
    if a.world_size is None:a.world_size=16 if a.variant=='ep16_adapted' else 8
    a.E=a.E or 4*a.world_size
    if not a.run_npu:
        print(json.dumps({'action':'plan only; no Torch import','variant':a.variant,'world':a.world_size,
            'shape':{k:getattr(a,k) for k in ('B','H','I','E','K')},'scope':'full-gradient accuracy only; no device authorization',
            'run_requires':['explicit frozen P0 root','candidate parent','content-bound audited wheel contract','reserved environment'],
            'owner':'one UDMA unique-ID owner throughout cases; adapter close then UDMA destroy then PG/host cleanup'}));return
    if a.output is None:raise ValueError('--output required for explicit run')
    import torch,torch_npu,torch.distributed as dist
    from host_store import StoreGroup
    host=StoreGroup.from_env();rank,world=host.rank(),host.size();device=f'npu:{int(os.environ["LOCAL_RANK"])}'
    state={'runtime':None,'engine':None,'init_attempted':False};report={'rank':rank,'world':world,'backend':a.variant,'cases':[],'stages':[],'all_passed':False}
    def save():a.output.mkdir(parents=True,exist_ok=True);(a.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        torch.npu.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('hccl')
        def prepare():
            if any(v is None for v in (a.training_source_root,a.candidate_parent,a.build_contract)):raise ValueError('Explicit roots/build contract required')
            if world!=a.world_size:raise ValueError('WORLD_SIZE differs from explicit world')
            root=a.training_source_root.resolve(strict=True);parent=a.candidate_parent.resolve(strict=True)
            if root.is_relative_to(parent) or parent.is_relative_to(root):raise ValueError('Frozen and candidate roots overlap')
            sys.path[:0]=[str(root/'tests'),str(root),str(parent)]
            base=importlib.import_module('provenance');ref=importlib.import_module('validate_training_npu')
            package_name='ep_gmm_training' if a.variant=='ep8_original' else 'ep_gmm_training_ep16'
            wrapper=importlib.import_module(package_name)
            require_module_paths(required_module_paths(root,parent,a.variant))
            for module,path in ((base,root/'tests/provenance.py'),(ref,root/'tests/validate_training_npu.py'),(wrapper,parent/package_name/'__init__.py')):
                if Path(module.__file__).resolve()!=path:raise ValueError('Import escaped explicit source root')
            ep=importlib.import_module('deep_ep_experimental.ep_gmm_fused');ep_core=importlib.import_module(ep.__name__+'._C')
            runtime=importlib.import_module(ep.__name__+'.runtime');ops=importlib.import_module(ep.__name__+'.ops')
            deep=importlib.import_module('deep_ep');core=importlib.import_module('deep_ep._native').require_native('EP-GMM training qualifier')
            if Path(deep.__file__).resolve().parents[1]!=Path(ep.__file__).resolve().parents[2]:raise ValueError('DeepEP/ep_gmm packages must share one isolated wheel root')
            if core.runtime_busy() or runtime._session is not None or runtime._config is not None or runtime._closing or runtime._leases:
                raise RuntimeError('A runtime owner/unclean state already exists; never adopt/destroy it')
            torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
            config=dict(variant=a.variant,world=world,shape={k:getattr(a,k) for k in ('B','H','I','E','K')},seed=a.seed,
                shared=a.shared,weight_dtype='torch.bfloat16' if a.weight_dtype=='bf16' else 'torch.float32',cases=a.cases.split(','),
                oracle_chunk=a.oracle_chunk,max_reference_gib=a.max_reference_gib,heap_size=a.heap_size,limits=ref.LIMITS,
                gemm_cores=32,owner='one_unique_id_udma',schedule='serial',recompute=False)
            check_config(config)
            if a.oracle_chunk<=0 or not math.isfinite(a.max_reference_gib) or not 400*1024**2+4096<a.heap_size<1<<63:raise ValueError('Invalid reference/heap size')
            if a.shared and not ref.shared_precision_shape_allowed(a,world):raise ValueError('Frozen shared shape restriction')
            reservation=ref.reference_memory_reservation(a,world)
            if reservation['total_GiB']>a.max_reference_gib:raise MemoryError('Frozen reference reservation exceeds cap')
            contract=json.loads(a.build_contract.read_text());check_contract(contract,a.variant)
            hardware=list(torch.ops.deep_ep_ep_gmm_fused._tiling_hardware(torch.empty(0,device=device,dtype=torch.uint8)))
            if len(hardware)!=2 or (a.variant=='ep16_adapted' and hardware[0]!=32):raise ValueError('EP16 requires actual32 AIC, queried by native host API')
            config['actual_hardware']={'aic_cores':hardware[0],'ub_bytes':hardware[1]}
            state.update(base=base,ref=ref,wrapper=wrapper,runtime=runtime,ops=ops,ep=ep,ep_core=ep_core,core=core,config=config,contract=contract,parent=parent)
            state['identity']=collect(config,contract,base,parent,wrapper,ep,ep_core,core,ops)
            report.update(execution=config,shape=config['shape'],seed=a.seed,weight_dtype=config['weight_dtype'],shared=a.shared,limits=ref.LIMITS,reference_memory_reservation=reservation,identity=state['identity'])
        host_stage('prepare-and-actual-identity',prepare,host,report,save)
        def agreement():
            tokens=host.all_gather(state['identity']['content_digest'],'EP-GMM/identity-agreement')
            if len(set(tokens))!=1:raise ValueError('Variant/config/source/native identities differ across ranks')
        host_stage('identity-agreement',agreement,host,report,save)
        def initialize():
            state['init_attempted']=True
            config=state['runtime'].init_udma_group(group=dist.group.WORLD,local_rank=int(os.environ['LOCAL_RANK']),heap_size=a.heap_size)
            if config.mode!='unique_id':raise RuntimeError('Unique-ID UDMA bootstrap required')
        host_stage('unique-ID-UDMA-init',initialize,host,report,save)
        def refresh_identity():
            fresh=collect(state['config'],state['contract'],state['base'],state['parent'],state['wrapper'],state['ep'],state['ep_core'],state['core'],state['ops'])
            # Previously bound sources/native objects cannot change mid-run;
            # genuinely lazy newly mapped dependencies are retained in final ID.
            def unchanged(old,new):
                before={v['role']:v for v in old};after={v['role']:v for v in new}
                if any(after.get(k)!=v for k,v in before.items()):raise ValueError('Previously loaded source/native identity changed')
            unchanged(state['identity']['content']['files'],fresh['content']['files'])
            unchanged(state['identity']['content']['base_identity']['content']['files'],fresh['content']['base_identity']['content']['files'])
            state['identity']=fresh;report['identity']=fresh
        host_stage('post-init-actual-identity',refresh_identity,host,report,save)
        host_stage('post-init-identity-agreement',agreement,host,report,save)
        def construct():
            name='EpGmmTrainingMoE' if a.variant=='ep8_original' else 'Ep16GmmTrainingMoE'
            state['engine']=getattr(state['wrapper'],name)(group=dist.group.WORLD,max_tokens=a.B,hidden=a.H,topk=a.K,num_experts=a.E,cores=32)
        host_stage('one-adapter-construct',construct,host,report,save)
        a.weight_dtype={'bf16':torch.bfloat16,'fp32':torch.float32}[a.weight_dtype]
        for case in a.cases.split(','):
            item={};host_stage('precision/'+case,lambda:run_case(a,state['ref'],state['engine'],rank,world,device,case,report,item),host,report,save,item)
        host_stage('final-actual-identity',refresh_identity,host,report,save)
        host_stage('final-identity-agreement',agreement,host,report,save)
        report['all_passed']=True
    except Exception as exc:report['error']=repr(exc);report['all_passed']=False;raise
    finally:
        prior=sys.exc_info()[1];cleanup_error=None
        try:cleanup_owner(state['engine'],state['runtime'],state['init_attempted'],host,report)
        except Exception as exc:cleanup_error=exc;report['all_passed']=False
        finish_host_report(host,report,dist,save,cleanup_error or prior)


if __name__=='__main__':main()
