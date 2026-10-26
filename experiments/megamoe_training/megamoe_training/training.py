"""Training MoE: SHMEM expanded routing and CATLASS forward/dgrad/wgrad.

P0 deliberately keeps communication, GEMM and pointwise operations separate.
It is a correctness baseline for the subsequent fused pipeline, not a claim of
fused MegaMoE performance. Routing indices are nondifferentiable; their selected
weights receive gradients, including at zero weight. Second derivatives are not
supported. FP32 master weights are supported with BF16 compute operands.
"""
import torch
from torch.autograd.function import once_differentiable
from deep_ep import ElasticBuffer
from .gemm import CatlassGemm


def _activation(gu):
    g, u = gu.float().chunk(2, dim=-1)
    return (torch.nn.functional.silu(g) * u).to(torch.bfloat16)


def _activation_grad(da, gu):
    g, u = gu.float().chunk(2, dim=-1)
    s = torch.sigmoid(g)
    d = da.float()
    return torch.cat((d * u * s * (1 + g * (1 - s)), d * g * s), dim=-1).to(torch.bfloat16)


class _RoutedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, idx, p, engine):
        engine._validate(x, w1, w2, idx, p)
        gemm = engine.gemm
        wb1, wb2 = w1.detach().to(torch.bfloat16), w2.detach().to(torch.bfloat16)
        xe, _, pe, handle, _ = engine.buffer.dispatch(
            x.detach(), topk_idx=idx.detach(), topk_weights=p.detach(),
            num_experts=engine.num_experts, num_compute_units=32, num_qps=2,
            expert_alignment=1, do_cpu_sync=True)
        if handle.is_low_latency or pe is None:
            raise RuntimeError("Training requires HT expanded BF16 routing and explicit gate weights")
        counts = tuple(handle.num_recv_tokens_per_expert_list)
        gu = gemm.grouped(xe, wb1, counts)
        a = _activation(gu)
        z = gemm.grouped(a, wb2, counts)
        weighted = (z.float() * pe[:, None]).to(torch.bfloat16)
        y, _, _ = engine.buffer.combine(weighted, handle)
        ctx.engine, ctx.handle, ctx.counts = engine, handle, counts
        ctx.weight_dtypes = (w1.dtype, w2.dtype)
        ctx.save_for_backward(xe, gu, a, z, pe, wb1, wb2)
        engine.last_counts = list(counts)
        engine.last_shapes = {"x": list(x.shape), "xe": list(xe.shape),
            "w1": list(w1.shape), "gu": list(gu.shape), "a": list(a.shape),
            "w2": list(w2.shape), "z": list(z.shape), "y": list(y.shape)}
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        engine, handle, counts = ctx.engine, ctx.handle, ctx.counts
        if engine.closed:
            raise RuntimeError("TrainingMoE.close() was called before backward")
        xe, gu, a, z, pe, w1, w2 = ctx.saved_tensors
        # Cached routing uses the forward's exact permutation without a new notify.
        de, _, _, _, _ = engine.buffer.dispatch(
            dy.detach().to(torch.bfloat16).contiguous(), handle=handle,
            num_compute_units=32, num_qps=2, do_cpu_sync=False)
        # Compute dp before weighting dZ. Never divide by p: p may be zero.
        dp_e = (de.float() * z.float()).sum(dim=-1).contiguous()
        dz = (de.float() * pe[:, None]).to(torch.bfloat16)
        dw2 = engine.gemm.weight_grad(dz, a, counts)
        da = engine.gemm.grouped(dz, w2, counts, transpose_weights=False)
        dgu = _activation_grad(da, gu)
        dw1 = engine.gemm.weight_grad(dgu, xe, counts)
        dx_e = engine.gemm.grouped(dgu, w1, counts, transpose_weights=False)
        dx, dp, _ = engine.buffer.combine(dx_e, handle, topk_weights=dp_e)
        if dp is None:
            raise RuntimeError("SHMEM did not return selected router-weight gradients")
        engine.last_grads_fp32 = (dw1, dw2)
        return dx, dw1.to(ctx.weight_dtypes[0]), dw2.to(ctx.weight_dtypes[1]), None, dp, None


class _SharedExpert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, engine):
        wb1, wb2 = w1.detach().to(torch.bfloat16), w2.detach().to(torch.bfloat16)
        if (w1.ndim != 2 or w2.ndim != 2 or w1.shape != (w2.shape[1] * 2, x.shape[1])
                or w2.shape[0] != x.shape[1] or not w1.is_contiguous() or not w2.is_contiguous()):
            raise ValueError("Shared expert expects w1[2I,H], w2[H,I]")
        gu = engine.gemm(x.detach(), wb1, trans_b=True)
        a = _activation(gu)
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
        dgu = _activation_grad(da, gu)
        dw1 = gemm(dgu, x.detach(), trans_a=True, fp32=True)
        dx = gemm(dgu, w1)
        ctx.engine.last_shared_grads_fp32 = (dw1, dw2)
        return dx, dw1.to(ctx.dtypes[0]), dw2.to(ctx.dtypes[1]), None


class TrainingMoE:
    """One EP group, sequential collective use, caller-owned model parameters.

    BF16 parameter gradients returned by autograd are BF16, as required by Torch.
    Use FP32 master parameters to retain FP32 weight gradients in `.grad`.
    `last_grads_fp32` also retains the latest FP32 accumulation for verification.
    Call close collectively after all pending forwards/backwards have completed.
    """
    def __init__(self, *, group, max_tokens, hidden, topk, num_experts,
                 num_bytes=None, library=None, cores=32, debug_collectives=False):
        self.gemm = CatlassGemm(library, cores)
        self.group = group
        self.world_size = torch.distributed.get_world_size(group)
        if num_experts % self.world_size or not 1 <= topk <= min(16, num_experts):
            raise ValueError("Experts must divide EP size; topk must be in [1,min(E,16)]")
        self.num_experts, self.max_tokens, self.hidden, self.topk = num_experts, max_tokens, hidden, topk
        self.buffer = ElasticBuffer(group=group, num_bytes=num_bytes,
            num_max_tokens_per_rank=max_tokens, hidden=hidden, num_topk=topk,
            use_fp8_dispatch=False, explicitly_destroy=True, debug_collectives=debug_collectives)
        self.closed = False
        self.last_grads_fp32 = self.last_shared_grads_fp32 = None
        self.last_counts = self.last_shapes = None

    def _validate(self, x, w1, w2, idx, p):
        if self.closed:
            raise RuntimeError("TrainingMoE has been closed")
        if x.dtype != torch.bfloat16 or x.ndim != 2 or x.device.type != "npu":
            raise ValueError("x must be BF16 [B,H] on NPU")
        b, h = x.shape
        e = self.num_experts // self.world_size
        if not 1 <= b <= self.max_tokens or h != self.hidden:
            raise ValueError("x shape exceeds configured bounds")
        if (w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != e or w2.shape[0] != e
                or w1.shape[2] != h or w2.shape[1] != h or w1.shape[1] != 2 * w2.shape[2]):
            raise ValueError("Weights must be w1[E_local,2I,H], w2[E_local,H,I]")
        if w1.dtype not in (torch.bfloat16, torch.float32) or w2.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Weights must be BF16 or FP32 master parameters")
        if idx.shape != (b, self.topk) or idx.dtype != torch.int64:
            raise ValueError("idx must be int64 [B,K]")
        if p.shape != idx.shape or p.dtype != torch.float32:
            raise ValueError("p must be FP32 [B,K]")
        if any(t.device != x.device or not t.is_contiguous() for t in (x, w1, w2, idx, p)):
            raise ValueError("All inputs must be contiguous on the same NPU")

    def __call__(self, x, w1, w2, idx, p, shared_w1=None, shared_w2=None):
        if (shared_w1 is None) != (shared_w2 is None):
            raise ValueError("Supply both shared-expert weights")
        routed = _RoutedMoE.apply(x, w1, w2, idx, p, self)
        if shared_w1 is None:
            return routed
        shared = _SharedExpert.apply(x, shared_w1, shared_w2, self)
        return routed + shared

    def close(self):
        if not self.closed:
            self.buffer.destroy()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
