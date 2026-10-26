"""CPU Torch only: actual adapter context/math with two-rank fused-op stubs.

No native transport, stream, allocator or NPU qualification. Rewrites only the
adapter runtime's expected device spelling npu:current -> cpu in an AST copy;
frozen P0 test loader independently changes its one NPU validator comparison.
Communication is simulated by synchronized CPU threads; no ProcessGroup exists.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import types
from unittest.mock import patch
import torch
from torch.autograd.function import once_differentiable

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]
P0=WORK/'DeepEP/experiments/megamoe_training/tests/test_training_math_cpu.py'
SPEC=importlib.util.spec_from_file_location('p0_cpu_helpers',P0)
P0TEST=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(P0TEST)
FROZEN=P0TEST.load_actual_training()
LIMIT={'atol':.003,'rtol':.03,'rms_relative':.02}


class Hub:
    world=2
    def __init__(self):
        self.local=threading.local();self.condition=threading.Condition();self.pending={};self.outputs={}
        self.configs=[types.SimpleNamespace(rank=r,world_size=2,mode='unique_id',members=(0,1),device='cpu') for r in range(2)]
        self.syncs=0
    def exchange(self,kind,payload,compute):
        rank=self.local.rank;seq=self.local.seq;self.local.seq+=1
        with self.condition:
            slot=self.pending.setdefault(seq,{})
            slot[rank]=(kind,payload)
            if len(slot)==2:
                assert len({v[0] for v in slot.values()})==1,'collective sequence mismatch'
                self.outputs[seq]=compute([slot[r][1] for r in range(2)]);self.condition.notify_all()
            if not self.condition.wait_for(lambda:seq in self.outputs,timeout=15):
                raise TimeoutError('CPU stub peer missing at '+kind)
            return self.outputs[seq][rank]
    def get_world_size(self,group=None):return 2
    def get_rank(self,group=None):return self.local.rank
    def get_process_group_ranks(self,group):return [0,1]
    ReduceOp=types.SimpleNamespace(MIN='min')
    def all_reduce(self,value,**kwargs):
        result=self.exchange('reduce',value.clone(),lambda xs:[torch.stack(xs).min(0).values]*2);value.copy_(result)
    def all_gather(self,out,value,**kwargs):
        result=self.exchange('gather',value.clone(),lambda xs:[xs]*2)
        for dest,src in zip(out,result):dest.copy_(src)
    def config(self):
        c=self.configs[self.local.rank]
        if c is None:raise RuntimeError('UDMA destroyed')
        return c
    def synchronize(self,device):self.syncs+=1
    def up(self,x,w,sc,rc,**kwargs):
        def compute(values):
            result=[];local=len(values[0][1])
            for owner,(_,weight,_,recv) in enumerate(values):
                chunks=[];ys=[]
                for expert in range(local):
                    for src,(source,_,send,_) in enumerate(values):
                        e=owner*local+expert;start=sum(send[:e]);n=send[e]
                        assert n==recv[src*local+expert]
                        xx=source[start:start+n];chunks.append(xx);ys.append((xx.float()@weight[expert].float()).bfloat16())
                xe=torch.cat(chunks);result.append((torch.cat(ys),xe.clone()))
            return result
        assert kwargs['output_all_to_allv'] is True and kwargs['received_rows']==sum(rc)
        return self.exchange('up',(x,w,sc,rc),compute)
    def down(self,x,w,sc,rc,**kwargs):
        def compute(values):
            local=len(values[0][1]);pieces={}
            for owner,(xx,weight,_,recv) in enumerate(values):
                offset=0
                for expert in range(local):
                    for src in range(2):
                        n=recv[src*local+expert];piece=xx[offset:offset+n];offset+=n
                        pieces[src,owner*local+expert]=(piece.float()@weight[expert].float()).bfloat16()
                assert offset==len(xx)
            result=[]
            for src,(_,_,send,_) in enumerate(values):
                assert all(len(pieces[src,e])==n for e,n in enumerate(send))
                result.append(torch.cat([pieces[src,e] for e in range(len(send))]))
            return result
        return self.exchange('down',(x,w,sc,rc),compute)


def load_adapter(hub):
    tree=ast.parse((ROOT/'training_adapter.py').read_text())
    tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    class CpuExpectedDevice(ast.NodeTransformer):
        count=0
        def visit_JoinedStr(self,n):
            if ast.unparse(n)=="f'npu:{torch.npu.current_device()}'":
                self.count+=1;return ast.copy_location(ast.Constant('cpu'),n)
            return self.generic_visit(n)
    rewrite=CpuExpectedDevice();tree=rewrite.visit(tree);assert rewrite.count==1
    ast.fix_missing_locations(tree)
    scope=dict(torch=torch,dist=hub,once_differentiable=once_differentiable,
        alltoallv_gmm=hub.up,gmm_alltoallv=hub.down,get_udma_config=hub.config,
        CatlassGemm=P0TEST.CpuGemm,FrozenTrainingMoE=FROZEN.TrainingMoE,
        _SharedExpert=FROZEN._SharedExpert,_activation=FROZEN._activation,_activation_grad=FROZEN._activation_grad)
    exec(compile(tree,str(ROOT/'training_adapter.py'),'exec'),scope)
    return types.SimpleNamespace(**scope)


class Context:
    def save_for_backward(self,*values):self.saved_tensors=values


def expert_vjp(x,w1,w2,dy,p=None):
    """Independent autograd VJPs, explicitly rounded between graph segments.

    FP32 weight operands are detached BF16 values, so autograd retains FP32 dW
    instead of allowing Torch's BF16 parameter-gradient cast to hide that result.
    """
    with torch.enable_grad():
        xx=x.float().detach().requires_grad_();first=w1.bfloat16().float().detach().requires_grad_()
        gu=xx@first.T;g,u=gu.detach().bfloat16().float().chunk(2,-1)
        g=g.requires_grad_();u=u.requires_grad_();activation=torch.nn.functional.silu(g)*u
        aa=activation.detach().bfloat16().float().requires_grad_();second=w2.bfloat16().float().detach().requires_grad_()
        zz=aa@second.T;z=zz.detach().bfloat16()
        if p is None:dz=dy
        else:dz=(dy.float()*p[:,None]).bfloat16()
        da,dw2=torch.autograd.grad(zz,(aa,second),dz.float())
        dg,du=torch.autograd.grad(activation,(g,u),da.bfloat16().float())
        dgu=torch.cat((dg,du),-1).bfloat16()
        dx,dw1=torch.autograd.grad(gu,(xx,first),dgu.float())
    return z,dx.bfloat16(),dw1,dw2


def oracle(data,shared):
    world=2;e=4;local=2;allw1=torch.cat([v[1].detach() for v in data]);allw2=torch.cat([v[2].detach() for v in data])
    totals=[torch.zeros_like(allw1,dtype=torch.float32),torch.zeros_like(allw2,dtype=torch.float32)];outputs=[]
    for rank,(x,w1,w2,idx,p,dy) in enumerate(data):
        yy=torch.zeros_like(x,dtype=torch.float32);dx=torch.zeros_like(yy);dp=torch.zeros_like(p)
        for expert in range(e):
            token,slot=torch.where(idx==expert)
            z,dd,dw1,dw2=expert_vjp(x[token],allw1[expert],allw2[expert],dy[token],p[token,slot])
            yy.index_add_(0,token,(z.float()*p[token,slot,None]).bfloat16().float());dx.index_add_(0,token,dd.float())
            dp[token,slot]=(dy[token].float()*z.float()).sum(-1);totals[0][expert]+=dw1;totals[1][expert]+=dw2
        result={'Y':yy.bfloat16(),'dx':dx.bfloat16(),'dp':dp}
        if shared is not None:
            sy,sd,sw1,sw2=expert_vjp(x,*shared[rank],dy)
            result.update(Y=result['Y']+sy,dx=result['dx']+sd,dSW1=sw1,dSW2=sw2)
        outputs.append(result)
    for rank,result in enumerate(outputs):
        result['dW1']=totals[0][rank*local:(rank+1)*local];result['dW2']=totals[1][rank*local:(rank+1)*local]
    return outputs


def stats(a,r):
    d=(a.float()-r.float()).abs();ref=r.float();bad=int((d>LIMIT['atol']+LIMIT['rtol']*ref.abs()).sum())
    rms=float(d.square().mean().sqrt()) if d.numel() else 0.;rr=float(ref.square().mean().sqrt()) if ref.numel() else 0.
    finite=bool(torch.isfinite(a).all() & torch.isfinite(r).all())
    return dict(elements=a.numel(),badcount=bad,max_abs=float(d.max()) if d.numel() else 0.,RMS=rms,
        passed=finite and bad==0 and (rms<=LIMIT['rms_relative']*rr if rr else rms<=LIMIT['atol']))


def run_case(case,dtype,with_shared):
    hub=Hub();module=load_adapter(hub);data=[];shared=[]
    for rank in range(2):
        gen=torch.Generator().manual_seed(903+rank);b,h,i=4,128,128
        x=(torch.randn(b,h,generator=gen)*.2).bfloat16();w1=(torch.randn(2,2*i,h,generator=gen)*.05).to(dtype);w2=(torch.randn(2,h,i,generator=gen)*.05).to(dtype)
        idx=(torch.arange(b)[:,None]+rank*b+torch.arange(2)[None,:]*2)%4
        if case=='empty':idx=(torch.arange(b*2).reshape(b,2)+rank*b*2)%3
        if case=='skew':idx=torch.tensor([[0,1]]).expand(b,2).contiguous()
        p=torch.rand(b,2,generator=gen)+.1
        if case=='zero_gate':p[:,0]=0
        dy=(torch.randn(b,h,generator=gen)*.1).bfloat16();data.append((x,w1,w2,idx,p,dy))
        shared.append(((torch.randn(2*i,h,generator=gen)*.05).to(dtype),(torch.randn(h,i,generator=gen)*.05).to(dtype)))
    expected=oracle(data,shared if with_shared else None)
    def worker(rank):
        hub.local.rank=rank;hub.local.seq=0
        owner=module.EpGmmTrainingMoE(group='cpu-stub',max_tokens=4,hidden=128,topk=2,num_experts=4)
        x,w1,w2,idx,p,dy=data[rank];ctx=Context()
        with torch.no_grad():
            y=module._EpGmmTraining.forward(ctx,x,w1,w2,idx,p,owner)
            dx,dw1,dw2,_,dp,_=module._EpGmmTraining.backward(ctx,dy)
            actual={'Y':y,'dx':dx,'dp':dp,'dW1':owner.last_grads_fp32[0],'dW2':owner.last_grads_fp32[1]}
            if with_shared:
                sc=Context();sy=FROZEN._SharedExpert.forward(sc,x,*shared[rank],owner)
                sdx,sw1,sw2,_=FROZEN._SharedExpert.backward(sc,dy)
                actual.update(Y=y+sy,dx=dx+sdx,dSW1=owner.last_shared_grads_fp32[0],dSW2=owner.last_shared_grads_fp32[1])
            metrics={n:stats(v,expected[rank][n]) for n,v in actual.items()}
            metrics['leaf_dW1']=stats(dw1,expected[rank]['dW1'].to(dtype));metrics['leaf_dW2']=stats(dw2,expected[rank]['dW2'].to(dtype))
            if with_shared:
                metrics['leaf_dSW1']=stats(sw1,expected[rank]['dSW1'].to(dtype));metrics['leaf_dSW2']=stats(sw2,expected[rank]['dSW2'].to(dtype))
        original=hub.configs[rank];hub.configs[rank]=copy.copy(original)
        try:owner._check_runtime()
        except RuntimeError:pass
        else:raise AssertionError('Different UDMA owner accepted')
        hub.configs[rank]=original;owner.close();owner.close()
        try:owner._check_runtime()
        except RuntimeError:pass
        else:raise AssertionError('Closed adapter accepted')
        assert hub.configs[rank] is original,'Adapter must not destroy caller domain'
        return dict(rank=rank,counts=owner.last_counts,metrics=metrics,passed=all(v['passed'] for v in metrics.values()),caller_owner_retained=True)
    with patch.object(torch,'npu',types.SimpleNamespace(current_device=lambda:0,synchronize=hub.synchronize),create=True):
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(worker,range(2)))
    assert hub.syncs==2,'Idempotent close synchronizes once per adapter'
    return dict(case=case,weight_dtype=str(dtype),shared=with_shared,ranks=results,passed=all(v['passed'] for v in results))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'tests/cpu_results.json');args=ap.parse_args()
    torch.set_num_threads(1)
    report={'scope':'CPU actual forward/backward bodies with simulated two-rank communications and CPU GEMM; independent segmented-autograd oracle; NOT NPU qualification',
            'adapter_sha256':hashlib.sha256((ROOT/'training_adapter.py').read_bytes()).hexdigest(),'torch':torch.__version__,'limits':LIMIT,'cases':[],'all_passed':False}
    try:
        for case in ('balanced','zero_gate','empty','skew'):
            for dtype,shared in ((torch.bfloat16,False),(torch.float32,True)):
                result=run_case(case,dtype,shared);report['cases'].append(result)
                if not result['passed']:raise AssertionError('Fixed CPU precision gate failed')
        report['all_passed']=True
    except Exception as exc:report['error']=repr(exc);raise
    finally:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,cases=len(report['cases']),scope='CPU stub only')))


if __name__=='__main__':main()
