"""Unqualified CANN training baseline; explicit autograd and CATLASS wgrad.

Forward: fused alltoallv+GMM1, separate FP32 SwiGLU->BF16, fused GMM2+
alltoallv, separate unpack/gating/topk. Backward: fused alltoallv+dA GEMM,
separate activation derivative and TWO CATLASS FP32 weight-gradient groups,
fused dX GEMM+alltoallv, separate dP/unpack/topk. The same CATLASS wgrad
backend as the candidate isolates communication/GMM fusion, not wgrad speed.
No torch.mm production fallback. 950 trans=False/export paths still require
hardware qualification; successful import/syntax is not support evidence.
"""
import torch
import torch.distributed as dist
import torch_npu
from torch.autograd.function import once_differentiable
from cann_baseline import CannForward
from megamoe_training.gemm import CatlassGemm


def exported_alltoallv_gmm(base,x,w,*,trans):
    # Deployed schema is (gmmY, mmY, permute_out). Xe/dZe is the THIRD
    # return; mmY is absent because no shared mm_x/mm_weight is supplied.
    out,_,exported=torch_npu.npu_alltoallv_gmm(
        x,w,base.hcom,base.world,base.sc,base.rc,
        trans_gmm_weight=trans,permute_out_flag=True,comm_mode=base.mode)
    if (exported is None or exported.ndim!=2 or
            tuple(exported.shape)!=(sum(base.rc),x.shape[1]) or
            exported.dtype!=torch.bfloat16):
        raise RuntimeError('CANN permute_out=True did not return the required BF16 expert/source-major export in tuple[2]')
    return out,exported.contiguous()


class _CannTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w1,w2,idx,p,owner):
        if owner.closed:
            raise RuntimeError('CannTrainingMoE is closed')
        if (x.ndim!=2 or not 0<x.shape[0]<=owner.max_tokens or x.shape[1]!=owner.hidden or
                idx.shape!=(x.shape[0],owner.topk) or p.shape!=idx.shape or
                w1.shape[0]!=owner.num_experts//owner.world or w1.shape[2]!=owner.hidden or
                w1.shape[1]!=2*w2.shape[2] or w2.shape[:2]!=(w1.shape[0],owner.hidden)):
            raise ValueError('CANN training shape contract mismatch')
        if any(t.device!=x.device or not t.is_contiguous() for t in (x,w1,w2,idx,p)):
            raise ValueError('All inputs must be contiguous on the same device')
        if w1.dtype not in (torch.bfloat16,torch.float32) or w2.dtype not in (torch.bfloat16,torch.float32):
            raise ValueError('Weights must be BF16 or FP32 master parameters')
        # Constructor refreshes routing/counts on EACH forward. No static-plan
        # shortcut is used in the full training comparison.
        base=CannForward(x,w1,w2,idx,p,comm_mode=owner.comm_mode,group=owner.group)
        packed_x=base.pack()
        gu,xe=exported_alltoallv_gmm(base,packed_x,base.w1,trans=True)
        a=base.activation(gu)
        packed_z=base.down(a)
        y=base.restore(packed_z)
        ctx.owner,ctx.base=owner,base
        ctx.weight_dtypes=(w1.dtype,w2.dtype)
        sorted_p=p.detach().flatten().index_select(0,base.order)
        ctx.save_for_backward(gu,xe,a,packed_z,base.w1,base.w2,sorted_p)
        owner.last_counts=list(base.expert_counts)
        owner.last_shapes={'x':list(x.shape),'xe':list(xe.shape),'w1':list(w1.shape),
                           'gu':list(gu.shape),'a':list(a.shape),'w2':list(w2.shape),
                           'packed_z':list(packed_z.shape),'y':list(y.shape)}
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        owner,base=ctx.owner,ctx.base
        if owner.closed:
            raise RuntimeError('CannTrainingMoE closed before backward')
        gu,xe,a,packed_z,w1,w2,sorted_p=ctx.saved_tensors
        d=dy.detach().bfloat16().index_select(0,base.source_token).contiguous()
        dz_packed=(d.float()*sorted_p[:,None]).bfloat16()
        dp_sorted=(d.float()*packed_z.float()).sum(dim=-1)
        dp=dp_sorted.index_select(0,base.inverse).reshape(base.b,base.k).contiguous()
        da,dz_e=exported_alltoallv_gmm(base,dz_packed,w2,trans=False)
        g,u=gu.float().chunk(2,dim=-1)
        sig=torch.sigmoid(g)
        dgu=torch.cat((da.float()*u*sig*(1+g*(1-sig)),da.float()*g*sig),dim=-1).bfloat16()
        # Both backends use this exact CATLASS FP32 wgrad implementation.
        dw2=owner.gemm.weight_grad(dz_e,a,base.expert_counts)
        dw1=owner.gemm.weight_grad(dgu,xe,base.expert_counts)
        packed_dx,_=torch_npu.npu_gmm_alltoallv(
            dgu,w1,base.hcom,base.world,base.rc,base.sc,
            trans_gmm_weight=False,comm_mode=base.mode)
        dx=packed_dx.index_select(0,base.inverse).reshape(base.b,base.k,base.h).float().sum(dim=1).bfloat16()
        owner.last_grads_fp32=(dw1,dw2)
        return dx,dw1.to(ctx.weight_dtypes[0]),dw2.to(ctx.weight_dtypes[1]),None,dp,None


class CannTrainingMoE:
    """Test-only baseline. Shared experts are intentionally not implemented."""
    def __init__(self,*,group,max_tokens,hidden,topk,num_experts,comm_mode,library=None,cores=32):
        if comm_mode not in ('ai_cpu','ccu'):
            raise ValueError('Only audited ai_cpu/ccu mode spellings are accepted')
        self.group,self.world=group,dist.get_world_size(group)
        if num_experts%self.world or not 0<max_tokens<=16384 or not 0<topk<=min(num_experts,16):
            raise ValueError('Invalid expert/routing dimensions')
        self.max_tokens,self.hidden,self.topk,self.num_experts=max_tokens,hidden,topk,num_experts
        self.comm_mode=comm_mode
        self.gemm=CatlassGemm(library,cores)
        self.closed=False
        self.last_counts=self.last_shapes=self.last_grads_fp32=None

    def __call__(self,x,w1,w2,idx,p):
        return _CannTraining.apply(x,w1,w2,idx,p,self)

    def close(self):
        self.closed=True
