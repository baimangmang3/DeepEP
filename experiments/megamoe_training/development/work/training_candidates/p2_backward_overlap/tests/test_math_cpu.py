"""CPU Torch real Function/autograd regression; not NPU/overlap qualification.

Loads actual P0/P1/P2 bodies. P0's existing CPU loader rewrites only its one
device validator. Communication/native pointwise/GEMM are CPU stubs; oracle
uses independently segmented autograd VJPs with explicit BF16 boundaries.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch
import torch
from torch.autograd.function import once_differentiable

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


P0TEST=load('actual_p0_cpu',WORK/'DeepEP/experiments/megamoe_training/tests/test_training_math_cpu.py')
FROZEN=P0TEST.ACTUAL
VJP=load('independent_autograd_vjp',ROOT.parent/'ep_gmm_training/tests/test_adapter_cpu.py').expert_vjp
sys.path.insert(0,str(WORK/'DeepEP/experiments/megamoe_training/tests'))
REF=load('frozen_statistics',WORK/'DeepEP/experiments/megamoe_training/tests/validate_training_npu.py')


class State:
    trace=[];fail_weight=None


class Event:
    def __init__(self):self.released=False
    def current_stream_wait(self,release_handle=False):
        State.trace.append('wait_release' if release_handle else 'wait')
        if release_handle:self.released=True


class CpuBuffer(P0TEST.CpuBuffer):
    last_event=None
    def combine(self,*args,**kwargs):
        async_mode=kwargs.pop('async_with_compute_stream',False)
        y,p,_=super().combine(*args,**kwargs)
        if not async_mode:return y,p,None
        State.trace.append('combine_async');self.last_event=Event()
        return y,p,self.last_event


class CpuGemm(P0TEST.CpuGemm):
    weight_calls=0
    def weight_grad(self,*args):
        self.weight_calls+=1;name='dW'+str(self.weight_calls);State.trace.append(name)
        if State.fail_weight==name:raise RuntimeError('injected '+name)
        return super().weight_grad(*args)


class CpuPointwise:
    def __init__(self,*args,**kwargs):pass
    def activation(self,gu):return FROZEN._activation(gu)
    def activation_grad(self,da,gu):return FROZEN._activation_grad(da,gu)
    def gate(self,z,p):return (z.float()*p[:,None]).bfloat16()


def source_module(path,namespace):
    tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(tree,str(path),'exec'),namespace);return types.SimpleNamespace(**namespace)


P1=source_module(ROOT.parent/'p1_pointwise/training_adapter.py',dict(torch=torch,once_differentiable=once_differentiable,
    FrozenTrainingMoE=FROZEN.TrainingMoE,TrainingPointwise=CpuPointwise))
P2=source_module(ROOT/'training_adapter.py',dict(torch=torch,once_differentiable=once_differentiable,
    PointwiseTrainingMoE=P1.PointwiseTrainingMoE,_RoutedP1=P1._RoutedP1,_SharedP1=P1._SharedP1))


def fixture(case,dtype,shared):
    gen=torch.Generator().manual_seed(9021);b,h,i=4,128,128
    x=(torch.randn(b,h,generator=gen)*.2).bfloat16().requires_grad_()
    w1=(torch.randn(4,2*i,h,generator=gen)*.05).to(dtype).requires_grad_()
    w2=(torch.randn(4,h,i,generator=gen)*.05).to(dtype).requires_grad_()
    idx=(torch.arange(b)[:,None]+torch.arange(2)[None,:]*2)%4
    if case=='empty':idx=torch.arange(b*2).reshape(b,2)%3
    if case=='skew':idx=torch.tensor([[0,1]]).expand(b,2).contiguous()
    p=(torch.rand(b,2,generator=gen)+.1)
    if case=='zero_gate':p[:,0]=0
    p.requires_grad_();dy=(torch.randn(b,h,generator=gen)*.1).bfloat16()
    sw=None
    if shared:sw=tuple((torch.randn(*shape,generator=gen)*.05).to(dtype).requires_grad_() for shape in ((2*i,h),(h,i)))
    return (x,w1,w2,idx,p,dy),sw


def independent(data,shared):
    x,w1,w2,ids,p,dy=data;yy=torch.zeros_like(x,dtype=torch.float32);dx=torch.zeros_like(yy);dp=torch.zeros_like(p)
    dw1=torch.zeros_like(w1,dtype=torch.float32);dw2=torch.zeros_like(w2,dtype=torch.float32)
    for e in range(len(w1)):
        token,slot=torch.where(ids==e)
        z,dxx,dw1[e],dw2[e]=VJP(x[token],w1[e],w2[e],dy[token],p[token,slot])
        yy.index_add_(0,token,(z.float()*p[token,slot,None]).bfloat16().float());dx.index_add_(0,token,dxx.float())
        with torch.enable_grad():
            gate=p[token,slot].detach().requires_grad_()
            dp[token,slot]=torch.autograd.grad(z.detach().float()*gate[:,None],gate,dy[token].float())[0]
    result=dict(Y=yy.bfloat16(),dx=dx.bfloat16(),dp=dp,dW1=dw1,dW2=dw2)
    if shared:
        sy,sd,sw1,sw2=VJP(x,*shared,dy)
        result.update(Y=result['Y']+sy,dx=result['dx']+sd,dSW1=sw1,dSW2=sw2)
    return result


def engine(schedule):
    return P2.BackwardOverlapTrainingMoE(backward_schedule=schedule,pointwise_library='CPU-stub-only',group=None,
        max_tokens=4,hidden=128,topk=2,num_experts=4)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'tests/math_cpu_results.json');args=ap.parse_args()
    torch.set_num_threads(1);records=[]
    report=dict(scope='CPU real custom Function with CPU communication/GEMM/pointwise stubs; independent segmented-autograd oracle; NOT NPU or overlap proof',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'__init__.py',ROOT/'training_adapter.py')},
        torch=torch.__version__,limits=REF.LIMITS,checks=records,all_passed=False)
    try:
        with patch.object(torch.distributed,'get_world_size',return_value=1),patch.object(FROZEN,'ElasticBuffer',CpuBuffer),patch.object(FROZEN,'CatlassGemm',CpuGemm):
            for schedule in ('serial','overlap'):
                for case in ('balanced','zero_gate','empty','skew'):
                    for dtype in (torch.bfloat16,torch.float32):
                        for shared in (False,True):
                            State.trace=[];State.fail_weight=None;data,sw=fixture(case,dtype,shared);expected=independent(data,sw)
                            x,w1,w2,ids,p,dy=data;owner=engine(schedule)
                            try:
                                y=owner(x,w1,w2,ids,p,**({} if sw is None else dict(zip(('shared_w1','shared_w2'),sw))))
                                assert y.requires_grad and y.grad_fn is not None,'Must test real Function.apply/autograd'
                                y.backward(dy)
                                actual=dict(Y=y,dx=x.grad,dp=p.grad,dW1=owner.last_grads_fp32[0],dW2=owner.last_grads_fp32[1])
                                metrics={n:REF.statistics(v,expected[n],REF.LIMITS[n]) for n,v in actual.items()}
                                for n,leaf in (('dW1',w1),('dW2',w2)):
                                    metrics['leaf_'+n]=REF.statistics(leaf.grad,expected[n].to(dtype),REF.LIMITS[n]);assert leaf.grad.dtype==dtype
                                if sw:
                                    for n,grad,leaf in zip(('dSW1','dSW2'),owner.last_shared_grads_fp32,sw):
                                        limit=REF.LIMITS['dW1' if n=='dSW1' else 'dW2'];metrics[n]=REF.statistics(grad,expected[n],limit)
                                        metrics['leaf_'+n]=REF.statistics(leaf.grad,expected[n].to(dtype),limit);assert leaf.grad.dtype==dtype
                                assert actual['dx'].dtype==torch.bfloat16 and actual['dp'].dtype==torch.float32
                                assert all(actual[n].dtype==torch.float32 for n in ('dW1','dW2'))
                                if case=='zero_gate':assert torch.count_nonzero(p.grad[p.detach()==0])>0
                                assert owner.buffer.last_event.released and State.trace[-1]=='wait_release'
                                if schedule=='serial':assert State.trace.index('wait')<State.trace.index('dW1')
                                else:assert 'wait' not in State.trace
                                passed=all(v['passed'] for v in metrics.values());records.append(dict(schedule=schedule,case=case,dtype=str(dtype),shared=shared,metrics=metrics,passed=passed))
                                if not passed:raise AssertionError('Fixed CPU precision gate failed')
                            finally:owner.close()
                for failure in ('dW1','dW2'):
                    State.trace=[];data,_=fixture('balanced',torch.float32,False);owner=engine(schedule)
                    try:
                        y=owner(*data[:5]);State.fail_weight=failure
                        try:y.backward(data[-1])
                        except RuntimeError as exc:assert 'injected '+failure in str(exc)
                        else:raise AssertionError('Missing injected failure')
                        assert owner.buffer.last_event.released and State.trace[-1]=='wait_release'
                        records.append(dict(schedule=schedule,failure=failure,trace=State.trace[:],passed=True))
                    finally:State.fail_weight=None;owner.close()
        report['all_passed']=True
    except Exception as exc:report['error']=repr(exc);raise
    finally:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,checks=len(records),scope='CPU stubs only')))


if __name__=='__main__':main()
