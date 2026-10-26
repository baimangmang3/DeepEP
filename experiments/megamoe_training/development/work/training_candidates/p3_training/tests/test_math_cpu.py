"""Real Torch CPU Function/autograd wiring, not native/SHMEM/device qualification.

Requires CPU Torch supplied by the execution owner. Reuses existing independent
segmented autograd VJP fixture, not the candidate backward as the oracle.
"""
import argparse,hashlib,importlib.util,json,types
from pathlib import Path
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
BASE=load('p3_existing_cpu_fixture',ROOT.parent/'p2_backward_overlap/tests/test_math_cpu.py')
FROZEN=BASE.FROZEN

class Trace:
    fused=[];activation=0;backward_activation=0;grouped=[];destroyed=0
    @classmethod
    def reset(cls):cls.fused=[];cls.activation=0;cls.backward_activation=0;cls.grouped=[];cls.destroyed=0

class CpuUp:
    def __init__(self,*args,**kwargs):pass
    def __call__(self,xe,weights,counts):
        assert weights.shape[0]==4 and sum(counts)==len(xe)
        assert xe.dtype==weights.dtype==torch.bfloat16
        parts=[];start=0
        for weight,count in zip(weights,counts):
            parts.append((xe[start:start+count].float()@weight.float().T).bfloat16());start+=count
        gu=torch.cat(parts);g,u=gu.float().chunk(2,-1)
        a=(torch.nn.functional.silu(g)*u).bfloat16()
        invocation=types.SimpleNamespace(gu=gu,a=a,flags=object())
        Trace.fused.append(invocation);return invocation

class CpuGemm(BASE.P0TEST.CpuGemm):
    def grouped(self,x,w,counts,**kw):
        Trace.grouped.append(dict(x_shape=list(x.shape),w_shape=list(w.shape),transpose=kw.get('transpose_weights',True)))
        return super().grouped(x,w,counts,**kw)

class CpuPointwise(BASE.CpuPointwise):
    def activation(self,gu):Trace.activation+=1;return super().activation(gu)
    def activation_grad(self,da,gu):Trace.backward_activation+=1;return super().activation_grad(da,gu)

class CpuBuffer(BASE.P0TEST.CpuBuffer):
    def destroy(self):Trace.destroyed+=1

P1=BASE.source_module(ROOT.parent/'p1_pointwise/training_adapter.py',dict(torch=torch,
    once_differentiable=BASE.once_differentiable,FrozenTrainingMoE=FROZEN.TrainingMoE,TrainingPointwise=CpuPointwise))
P3=BASE.source_module(ROOT/'training_adapter.py',dict(torch=torch,PointwiseTrainingMoE=P1.PointwiseTrainingMoE,
    _RoutedP1=P1._RoutedP1,_SharedP1=P1._SharedP1,GroupedUpSwiGLU=CpuUp))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'tests/math_cpu_results.json');args=ap.parse_args()
    torch.set_num_threads(1);records=[]
    dependencies=(ROOT/'__init__.py',ROOT/'training_adapter.py',Path(__file__),ROOT.parent/'p1_pointwise/training_adapter.py',
        ROOT.parent/'p2_backward_overlap/tests/test_math_cpu.py',ROOT.parent/'ep_gmm_training/tests/test_adapter_cpu.py',
        BASE.WORK/'DeepEP/experiments/megamoe_training/tests/test_training_math_cpu.py',
        BASE.WORK/'DeepEP/experiments/megamoe_training/tests/validate_training_npu.py',
        BASE.WORK/'DeepEP/experiments/megamoe_training/megamoe_training/training.py')
    report=dict(scope=__doc__,torch=str(torch.__version__),limits=BASE.REF.LIMITS,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},checks=records,all_passed=False)
    try:
        assert P3._RoutedUpSwiGLU.backward is P1._RoutedP1.backward
        with patch.object(torch.distributed,'get_world_size',return_value=1),patch.object(FROZEN,'ElasticBuffer',CpuBuffer),patch.object(FROZEN,'CatlassGemm',CpuGemm):
            for case in ('balanced','zero_gate','empty','skew'):
                for dtype in (torch.bfloat16,torch.float32):
                    for shared in (False,True):
                        Trace.reset();data,sw=BASE.fixture(case,dtype,shared);expected=BASE.independent(data,sw)
                        x,w1,w2,ids,p,dy=data
                        owner=P3.UpSwiGLUTrainingMoE(up_swiglu_library='CPU-stub-only',pointwise_library='CPU-stub-only',
                            group=None,max_tokens=4,hidden=128,topk=2,num_experts=4)
                        try:
                            y=owner(x,w1,w2,ids,p,**({} if sw is None else dict(zip(('shared_w1','shared_w2'),sw))))
                            assert y.requires_grad and y.grad_fn is not None
                            assert len(Trace.fused)==1 and Trace.activation==int(shared)
                            assert len(Trace.grouped)==1 and Trace.grouped[0]['w_shape']==[4,128,128]
                            y.backward(dy)
                            assert Trace.backward_activation==1+int(shared) and len(Trace.fused)==1
                            # Routed forward GMM2, backward dA/dX only: no separate routed up GEMM.
                            assert len(Trace.grouped)==3
                            assert [v['transpose'] for v in Trace.grouped]==[True,False,False]
                            actual=dict(Y=y,dx=x.grad,dp=p.grad,dW1=owner.last_grads_fp32[0],dW2=owner.last_grads_fp32[1])
                            metrics={n:BASE.REF.statistics(v,expected[n],BASE.REF.LIMITS[n]) for n,v in actual.items()}
                            for name,leaf in (('dW1',w1),('dW2',w2)):
                                metrics['leaf_'+name]=BASE.REF.statistics(leaf.grad,expected[name].to(dtype),BASE.REF.LIMITS[name])
                                assert leaf.grad.dtype==dtype
                            if sw:
                                for name,grad,leaf in zip(('dSW1','dSW2'),owner.last_shared_grads_fp32,sw):
                                    limit=BASE.REF.LIMITS['dW1' if name=='dSW1' else 'dW2']
                                    metrics[name]=BASE.REF.statistics(grad,expected[name],limit)
                                    metrics['leaf_'+name]=BASE.REF.statistics(leaf.grad,expected[name].to(dtype),limit)
                                    assert grad.dtype==torch.float32 and leaf.grad.dtype==dtype
                            assert actual['Y'].dtype==actual['dx'].dtype==torch.bfloat16
                            assert all(actual[n].dtype==torch.float32 for n in ('dp','dW1','dW2'))
                            if case=='zero_gate':assert torch.count_nonzero(p.grad[p.detach()==0])>0
                            assert owner.last_shapes['gu']==[8,256] and owner.last_shapes['a']==[8,128]
                            passed=all(v['passed'] for v in metrics.values())
                            records.append(dict(case=case,weight_dtype=str(dtype),shared=shared,fused_calls=len(Trace.fused),
                                activation_calls=Trace.activation,activation_grad_calls=Trace.backward_activation,
                                inherited_backward=True,metrics=metrics,passed=passed))
                            if not passed:raise AssertionError('Fixed full-element CPU precision gate failed')
                        finally:owner.close()
                        assert Trace.destroyed==1
        report['all_passed']=True
    except BaseException as exc:report['error']=repr(exc);raise
    finally:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,checks=len(records),scope='CPU stubs; no native/device qualification')))

if __name__=='__main__':main()
