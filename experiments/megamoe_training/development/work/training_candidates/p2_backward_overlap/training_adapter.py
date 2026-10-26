"""Unqualified P2: reordered dgrad, then dW concurrent with dX/dP return.

The serial control has the same arithmetic and launch order, but waits for
combine before dW. This is stream overlap, not a fused communication/GMM kernel.
P0 and P1 sources/libraries are dependencies and must be bound by a separate
qualification identity before this candidate is allowed into a benchmark.
"""
import torch
from torch.autograd.function import once_differentiable
from p1_pointwise.training_adapter import PointwiseTrainingMoE, _RoutedP1, _SharedP1


class _RoutedP2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, idx, p, engine):
        return _RoutedP1.forward(ctx, x, w1, w2, idx, p, engine)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        engine, handle, counts = ctx.engine, ctx.handle, ctx.counts
        if engine.closed:
            raise RuntimeError('TrainingMoE.close() was called before backward')
        xe, gu, a, z, pe, w1, w2 = ctx.saved_tensors
        de, _, _, _, _ = engine.buffer.dispatch(
            dy.detach().to(torch.bfloat16).contiguous(), handle=handle,
            num_compute_units=32, num_qps=2, do_cpu_sync=False)
        # Preserve the rounded, unweighted Z and the zero-gate derivative.
        dp_e = (de.float() * z.float()).sum(dim=-1).contiguous()
        dz = engine.pointwise.gate(de, pe)
        da = engine.gemm.grouped(dz, w2, counts, transpose_weights=False)
        dgu = engine.pointwise.activation_grad(da, gu)
        dx_e = engine.gemm.grouped(dgu, w1, counts, transpose_weights=False)

        # ElasticBuffer records a producer event on the current compute stream,
        # launches combine on its existing comm stream and retains/record_streams
        # all input/output/route tensors. Its return event covers final transport
        # completion, not just producer readiness. No readiness protocol changes.
        dx, dp, event = engine.buffer.combine(
            dx_e, handle, topk_weights=dp_e, async_with_compute_stream=True)
        if event is None:
            raise RuntimeError('P2 requires a real combine completion event')
        try:
            if dp is None:
                raise RuntimeError('SHMEM did not return selected router gradients')
            if engine.backward_schedule == 'serial':
                event.current_stream_wait()
            dw1 = engine.gemm.weight_grad(dgu, xe, counts)
            dw2 = engine.gemm.weight_grad(dz, a, counts)
        finally:
            # Queue the wait even when a weight-gradient launch raises. Waiting
            # also protects consumer-stream lifetimes; it is not a host sync.
            # The event is kept alive through both dW submissions above.
            event.current_stream_wait(release_handle=True)
        engine.last_grads_fp32 = (dw1, dw2)
        return dx, dw1.to(ctx.weight_dtypes[0]), dw2.to(ctx.weight_dtypes[1]), None, dp, None


class BackwardOverlapTrainingMoE(PointwiseTrainingMoE):
    candidate_name = 'P2_backward_return_overlap_v1_UNQUALIFIED'

    def __init__(self, *, backward_schedule, **kwargs):
        if backward_schedule not in ('serial', 'overlap'):
            raise ValueError('P2 schedule must be serial or overlap')
        self._backward_schedule = backward_schedule
        super().__init__(**kwargs)

    @property
    def backward_schedule(self):
        return self._backward_schedule

    def __call__(self, x, w1, w2, idx, p, shared_w1=None, shared_w2=None):
        if (shared_w1 is None) != (shared_w2 is None):
            raise ValueError('Supply both shared-expert weights')
        routed = _RoutedP2.apply(x, w1, w2, idx, p, self)
        if shared_w1 is None:
            return routed
        # Shared expert remains P1, with no claimed shared/communication overlap.
        return routed + _SharedP1.apply(x, shared_w1, shared_w2, self)
