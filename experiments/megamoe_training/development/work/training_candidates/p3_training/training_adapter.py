"""P3 routes through SHMEM and fuses only grouped W1 with SwiGLU.

Canonical BF16 GU/A are retained for the inherited P1 backward. Shared experts,
GMM2, gating, communication and backward remain separate. This module alone is
not evidence of native or complete training qualification.
"""
import torch
from p1_pointwise.training_adapter import PointwiseTrainingMoE, _RoutedP1, _SharedP1
from p3_up_swiglu.up_swiglu import GroupedUpSwiGLU


class _RoutedUpSwiGLU(_RoutedP1):
    @staticmethod
    def forward(ctx, x, w1, w2, idx, p, engine):
        engine._validate(x, w1, w2, idx, p)
        wb1, wb2 = w1.detach().to(torch.bfloat16), w2.detach().to(torch.bfloat16)
        xe, _, pe, handle, _ = engine.buffer.dispatch(
            x.detach(), topk_idx=idx.detach(), topk_weights=p.detach(),
            num_experts=engine.num_experts, num_compute_units=32, num_qps=2,
            expert_alignment=1, do_cpu_sync=True)
        if handle.is_low_latency or pe is None:
            raise RuntimeError('Training requires HT expanded BF16 routing and explicit gate weights')
        counts = tuple(handle.num_recv_tokens_per_expert_list)
        # Private ready flags are stream-recorded by the launcher; no flag reuse
        # or extra training save is needed. GU/A keep the same rounded boundary.
        invocation = engine.up_swiglu(xe, wb1, counts)
        gu, a = invocation.gu, invocation.a
        z = engine.gemm.grouped(a, wb2, counts)
        weighted = engine.pointwise.gate(z, pe)
        y, _, _ = engine.buffer.combine(weighted, handle)
        ctx.engine, ctx.handle, ctx.counts = engine, handle, counts
        ctx.weight_dtypes = (w1.dtype, w2.dtype)
        ctx.save_for_backward(xe, gu, a, z, pe, wb1, wb2)
        engine.last_counts = list(counts)
        engine.last_shapes = {'x': list(x.shape), 'xe': list(xe.shape),
            'w1': list(w1.shape), 'gu': list(gu.shape), 'a': list(a.shape),
            'w2': list(w2.shape), 'z': list(z.shape), 'y': list(y.shape)}
        return y

    # Inherited once_differentiable backward consumes exactly the saved tensors
    # above: unweighted Z for dp, one gate on dZ, FP32 dW, cached routing for dx.


class UpSwiGLUTrainingMoE(PointwiseTrainingMoE):
    candidate_name = 'P3_grouped_up_swiglu_v1'

    def __init__(self, *, up_swiglu_library, **kwargs):
        # Reject incompatible EP/expert configuration before SHMEM construction.
        world = torch.distributed.get_world_size(kwargs.get('group'))
        experts = kwargs.get('num_experts')
        if type(experts) is not int or world <= 0 or experts != 4 * world:
            raise ValueError('P3 requires exactly four local experts on every EP rank')
        if kwargs.get('cores', 32) != 32:
            raise ValueError('P3 requires 32 AIC and 64 AIV')
        self.up_swiglu = GroupedUpSwiGLU(up_swiglu_library)
        super().__init__(**kwargs)

    def __call__(self, x, w1, w2, idx, p, shared_w1=None, shared_w2=None):
        if (shared_w1 is None) != (shared_w2 is None):
            raise ValueError('Supply both shared-expert weights')
        routed = _RoutedUpSwiGLU.apply(x, w1, w2, idx, p, self)
        if shared_w1 is None:
            return routed
        return routed + _SharedP1.apply(x, shared_w1, shared_w2, self)
