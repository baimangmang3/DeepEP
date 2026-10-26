"""Unqualified training adapter around the pinned EP<=8 ep_gmm_fused API.

This file does not widen the pinned operator's EP limit or implement EP16.
The caller owns an initialized UDMA group, with one owner per process. No
ElasticBuffer is created. Use an isolated wheel/process for this baseline.
Weight layout conversions, fresh routing and exact Xe exports occur on every
forward, and therefore belong in full training timing. No weight cache or
PyTorch GEMM fallback is used. Full-gradient qualification is still required.
"""
import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable
from deep_ep_experimental.ep_gmm_fused import alltoallv_gmm, gmm_alltoallv
from deep_ep_experimental.ep_gmm_fused.runtime import get_udma_config
from megamoe_training.gemm import CatlassGemm
from megamoe_training.training import (
    TrainingMoE as FrozenTrainingMoE, _SharedExpert, _activation, _activation_grad,
)


class _RoutePlan:
    """Same dynamic host route plan/cast/reduction scope as the CANN control."""
    def __init__(self, x, idx, owner):
        self.b, self.h = x.shape
        self.k, self.e = owner.topk, owner.num_experts
        local_e = self.e // owner.world_size
        # Validate IDs before bincount so an invalid rank cannot create a
        # differently-sized collective. The status exchange is fixed-size.
        flat = idx.detach().cpu().flatten()
        valid = flat.numel() == self.b * self.k and bool(((flat >= 0) & (flat < self.e)).all())
        status = torch.tensor([int(valid)], dtype=torch.int32, device=x.device)
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=owner.group)
        if not status.item():
            raise ValueError('Invalid expert IDs on at least one EP rank')
        local = torch.bincount(idx.flatten(), minlength=self.e).to(torch.int64)
        peers = [torch.empty_like(local) for _ in range(owner.world_size)]
        dist.all_gather(peers, local, group=owner.group)
        table = torch.stack(peers).cpu()
        self.sc = table[owner.rank].tolist()
        local_table = table[:, owner.rank * local_e:(owner.rank + 1) * local_e]
        self.rc = local_table.flatten().tolist()  # [source rank, local expert]
        self.counts = tuple(local_table.sum(0).tolist())
        self.order = torch.argsort(flat, stable=True).to(x.device)
        self.inverse = torch.argsort(self.order)
        self.token = self.order // self.k
        self.ep, self.experts = owner.world_size, owner.num_experts

    def up(self, x, weight):
        # Host route lists deliberately select the pinned metadata-export path.
        # All per-call H2D route/metadata copies remain in the measured API scope.
        y, exported = alltoallv_gmm(
            x, weight, self.sc, self.rc, ep_size=self.ep,
            expert_num=self.experts, received_rows=sum(self.rc),
            output_all_to_allv=True)
        if (exported.shape != (sum(self.rc), x.shape[1]) or
                exported.dtype != torch.bfloat16 or exported.device != x.device or
                not exported.is_contiguous()):
            raise RuntimeError('ep_gmm export did not provide exact contiguous BF16 expert/source rows')
        return y, exported

    def down(self, x, weight):
        # ep_gmm API always takes source-local then received tables, unlike
        # CANN down API's positional receive/send count direction.
        return gmm_alltoallv(x, weight, self.sc, self.rc,
                            ep_size=self.ep, expert_num=self.experts)

    def restore(self, packed):
        return packed.index_select(0, self.inverse).reshape(self.b, self.k, self.h)


class _EpGmmTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, idx, p, owner):
        owner._validate(x, w1, w2, idx, p)
        base = _RoutePlan(x, idx, owner)
        wb1, wb2 = w1.detach().bfloat16(), w2.detach().bfloat16()
        # Canonical training [E,N,K] -> pinned ep_gmm [E,K,N]. This copy is
        # repeated after every parameter update, inside every forward scope.
        ew1 = wb1.transpose(1, 2).contiguous()
        ew2 = wb2.transpose(1, 2).contiguous()
        packed_x = x.detach().index_select(0, base.token).contiguous()
        gu, xe = base.up(packed_x, ew1)
        a = _activation(gu)
        packed_z = base.down(a, ew2)
        z = base.restore(packed_z)
        contributions = (z.float() * p.detach()[:, :, None]).bfloat16()
        y = contributions.float().sum(dim=1).bfloat16()
        sorted_p = p.detach().flatten().index_select(0, base.order)
        ctx.owner, ctx.base = owner, base
        ctx.weight_dtypes = (w1.dtype, w2.dtype)
        ctx.save_for_backward(gu, xe, a, packed_z, wb1, wb2, sorted_p)
        owner.last_counts = list(base.counts)
        owner.last_shapes = {'x': list(x.shape), 'xe': list(xe.shape),
            'w1': list(w1.shape), 'gu': list(gu.shape), 'a': list(a.shape),
            'w2': list(w2.shape), 'packed_z': list(packed_z.shape), 'y': list(y.shape)}
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        owner, base = ctx.owner, ctx.base
        owner._check_runtime()
        gu, xe, a, packed_z, w1, w2, sorted_p = ctx.saved_tensors
        d = dy.detach().bfloat16().index_select(0, base.token).contiguous()
        # Unweighted BF16 Z is retained even where gate==0; no divide by gate.
        dp_sorted = (d.float() * packed_z.float()).sum(dim=-1)
        dp = dp_sorted.index_select(0, base.inverse).reshape(base.b, base.k).contiguous()
        dz_packed = (d.float() * sorted_p[:, None]).bfloat16()
        # W2 canonical [E,H,I] already is [E,K,N] for the dA matmul.
        da, dz_e = base.up(dz_packed, w2)
        dgu = _activation_grad(da, gu)
        dw2 = owner.gemm.weight_grad(dz_e, a, base.counts)
        dw1 = owner.gemm.weight_grad(dgu, xe, base.counts)
        # W1 canonical [E,2I,H] already is [E,K,N] for dX. Gate was
        # applied to dZ; do not multiply it again while returning dX.
        packed_dx = base.down(dgu, w1)
        dx = base.restore(packed_dx).float().sum(dim=1).bfloat16()
        owner.last_grads_fp32 = (dw1, dw2)
        return dx, dw1.to(ctx.weight_dtypes[0]), dw2.to(ctx.weight_dtypes[1]), None, dp, None


class EpGmmTrainingMoE:
    """Borrow a caller-owned UDMA group; close before destroying that group.

    Closing this adapter synchronizes its outstanding work but never destroys
    the caller's UDMA domain or ProcessGroup. Original EP8 operators remain
    unchanged. Shared experts use the exact frozen P0 math/CATLASS path.
    """
    candidate_name = 'pinned_ep8_ep_gmm_training_adapter_v1'

    def __init__(self, *, group, max_tokens, hidden, topk, num_experts,
                 library=None, cores=32):
        self.group, self.world_size = group, dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if (not 1 <= self.world_size <= 8 or num_experts <= 0 or
                num_experts % self.world_size or num_experts // self.world_size > 32 or
                not 1 <= topk <= min(16, num_experts) or not 0 < max_tokens <= 16384 or
                hidden <= 0 or hidden % 128):
            raise ValueError('Invalid pinned EP8 training dimensions (down local experts<=32)')
        self.max_tokens, self.hidden = max_tokens, hidden
        self.topk, self.num_experts = topk, num_experts
        self.closed = False
        self._config = get_udma_config()
        self._check_runtime()
        self.gemm = CatlassGemm(library, cores)
        self.last_counts = self.last_shapes = self.last_grads_fp32 = None
        self.last_shared_grads_fp32 = None

    def _check_runtime(self):
        if self.closed:
            raise RuntimeError('EpGmmTrainingMoE has been closed')
        config = get_udma_config()
        if (config is not self._config or config.rank != self.rank or
                config.world_size != self.world_size or
                config.mode != 'unique_id' or
                config.members != tuple(dist.get_process_group_ranks(self.group)) or
                config.device != f'npu:{torch.npu.current_device()}'):
            raise RuntimeError('Caller UDMA group/device does not match this training adapter')

    def _validate(self, x, w1, w2, idx, p):
        self._check_runtime()
        FrozenTrainingMoE._validate(self, x, w1, w2, idx, p)
        if x.device != torch.device(self._config.device) or w2.shape[2] <= 0 or w2.shape[2] % 128:
            raise ValueError('Pinned training uses current UDMA device and native I multiples of128')

    def __call__(self, x, w1, w2, idx, p, shared_w1=None, shared_w2=None):
        if (shared_w1 is None) != (shared_w2 is None):
            raise ValueError('Supply both shared-expert weights')
        routed = _EpGmmTraining.apply(x, w1, w2, idx, p, self)
        if shared_w1 is None:
            return routed
        return routed + _SharedExpert.apply(x, shared_w1, shared_w2, self)

    def close(self):
        if not self.closed:
            self._check_runtime()
            torch.npu.synchronize(torch.device(self._config.device))
            self.closed = True
