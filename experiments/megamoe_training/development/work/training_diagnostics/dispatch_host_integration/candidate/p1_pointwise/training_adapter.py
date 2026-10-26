"""P1 changes only activation/activation-gradient and row-gate vector operations.

Communication, CATLASS GEMMs, saved tensors, dP and dW ordering match frozen P0.
This is NOT a fused communication/compute pipeline. Caller must first qualify
the standalone pointwise library and then all full-layer gradients separately.
"""
import torch
from dispatch_host_integration.hooks import routed_phase
from torch.autograd.function import once_differentiable
from megamoe_training.training import TrainingMoE as FrozenTrainingMoE
from .pointwise import TrainingPointwise


class _RoutedP1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, idx, p, engine):
        with routed_phase(engine, "forward"):
            engine._validate(x, w1, w2, idx, p)
            gemm = engine.gemm
            wb1, wb2 = w1.detach().to(torch.bfloat16), w2.detach().to(torch.bfloat16)
            xe, _, pe, handle, _ = engine.buffer.dispatch(
                x.detach(), topk_idx=idx.detach(), topk_weights=p.detach(),
                num_experts=engine.num_experts, num_compute_units=32, num_qps=2,
                expert_alignment=1, do_cpu_sync=True)
            if handle.is_low_latency or pe is None:
                raise RuntimeError('Training requires HT expanded BF16 routing and explicit gate weights')
            counts = tuple(handle.num_recv_tokens_per_expert_list)
            gu = gemm.grouped(xe, wb1, counts)
            a = engine.pointwise.activation(gu)
            z = gemm.grouped(a, wb2, counts)
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

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        engine, handle, counts = ctx.engine, ctx.handle, ctx.counts
        with routed_phase(engine, "backward"):
            if engine.closed:
                raise RuntimeError('TrainingMoE.close() was called before backward')
            xe, gu, a, z, pe, w1, w2 = ctx.saved_tensors
            de, _, _, _, _ = engine.buffer.dispatch(
                dy.detach().to(torch.bfloat16).contiguous(), handle=handle,
                num_compute_units=32, num_qps=2, do_cpu_sync=False)
            # Keep unweighted rounded Z, including when p == 0. No quotient by p.
            dp_e = (de.float() * z.float()).sum(dim=-1).contiguous()
            dz = engine.pointwise.gate(de, pe)
            dw2 = engine.gemm.weight_grad(dz, a, counts)
            da = engine.gemm.grouped(dz, w2, counts, transpose_weights=False)
            dgu = engine.pointwise.activation_grad(da, gu)
            dw1 = engine.gemm.weight_grad(dgu, xe, counts)
            dx_e = engine.gemm.grouped(dgu, w1, counts, transpose_weights=False)
            dx, dp, _ = engine.buffer.combine(dx_e, handle, topk_weights=dp_e)
            if dp is None:
                raise RuntimeError('SHMEM did not return selected router-weight gradients')
            engine.last_grads_fp32 = (dw1, dw2)
            return dx, dw1.to(ctx.weight_dtypes[0]), dw2.to(ctx.weight_dtypes[1]), None, dp, None


class _SharedP1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, engine):
        wb1, wb2 = w1.detach().to(torch.bfloat16), w2.detach().to(torch.bfloat16)
        if (w1.ndim != 2 or w2.ndim != 2 or w1.shape != (w2.shape[1] * 2, x.shape[1])
                or w2.shape[0] != x.shape[1] or not w1.is_contiguous() or not w2.is_contiguous()):
            raise ValueError('Shared expert expects w1[2I,H], w2[H,I]')
        gu = engine.gemm(x.detach(), wb1, trans_b=True)
        a = engine.pointwise.activation(gu)
        y = engine.gemm(a, wb2, trans_b=True)
        ctx.save_for_backward(x, gu, a, wb1, wb2)
        ctx.engine, ctx.dtypes = engine, (w1.dtype, w2.dtype)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, gu, a, w1, w2 = ctx.saved_tensors
        gemm = ctx.engine.gemm
        d = dy.detach().to(torch.bfloat16).contiguous()
        dw2 = gemm(d, a, trans_a=True, fp32=True)
        da = gemm(d, w2)
        dgu = ctx.engine.pointwise.activation_grad(da, gu)
        dw1 = gemm(dgu, x.detach(), trans_a=True, fp32=True)
        dx = gemm(dgu, w1)
        ctx.engine.last_shared_grads_fp32 = (dw1, dw2)
        return dx, dw1.to(ctx.dtypes[0]), dw2.to(ctx.dtypes[1]), None


class PointwiseTrainingMoE(FrozenTrainingMoE):
    candidate_name = 'P1_native_pointwise_v1'

    def __init__(self, *, pointwise_library, **kwargs):
        # Fixed configuration, recorded in P1 qualification; no unbound ablation switches.
        self.pointwise = TrainingPointwise(pointwise_library, vector_cores=64)
        super().__init__(**kwargs)

    def _validate(self, x, w1, w2, idx, p):
        super()._validate(x, w1, w2, idx, p)
        if any(n <= 0 or n % 128 or n > 2**20 for n in (x.shape[1], w2.shape[2])):
            raise ValueError('P1 requires native H and I multiples of128, each <=2**20')

    def __call__(self, x, w1, w2, idx, p, shared_w1=None, shared_w2=None):
        if (shared_w1 is None) != (shared_w2 is None):
            raise ValueError('Supply both shared-expert weights')
        routed = _RoutedP1.apply(x, w1, w2, idx, p, self)
        if shared_w1 is None:
            return routed
        shared = _SharedP1.apply(x, shared_w1, shared_w2, self)
        return routed + shared
