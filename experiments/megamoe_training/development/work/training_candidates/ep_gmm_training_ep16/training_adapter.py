"""Unqualified training adapter for the isolated, fully patched EP16 wheel.

Uses exactly the EP8 adapter's dynamic route plan, weight copies, saved Xe and
autograd arithmetic. This file does not adapt native rank tables or protocols.
It requires the separately built ep16.patch implementation and a caller-owned
EP16 UDMA group. Native/Python hashes and full precision must be checked before
benchmarking; metadata constants alone do not establish native compatibility.
"""
import torch.distributed as dist
from deep_ep_experimental.ep_gmm_fused import ops
from deep_ep_experimental.ep_gmm_fused.runtime import get_udma_config
from megamoe_training.gemm import CatlassGemm
from ep_gmm_training.training_adapter import EpGmmTrainingMoE


def require_ep16_metadata(module):
    expected = {
        '_MOE_ROUTE_MAX_RANKS': 16,
        '_MOE_ROUTE_MAX_LOCAL_EXPERTS': 64,
        '_MOE_ROUTE_MAX_SEGMENTS': 1024,
        '_MOE_ROUTE_PEER_ROWS_OFFSET': 16,
        '_MOE_ROUTE_EXPERT_ROWS_OFFSET': 144,
        '_MOE_ROUTE_SEGMENTS_OFFSET': 656,
        '_MOE_ROUTE_SEGMENT_SIZE': 24,
        '_MOE_ROUTE_METADATA_SIZE': 25232,
    }
    if any(getattr(module, key, None) != value for key, value in expected.items()):
        raise ValueError('EP16 adapter requires the isolated adapted Python metadata ABI')


class Ep16GmmTrainingMoE(EpGmmTrainingMoE):
    candidate_name = 'adapted_ep16_ep_gmm_training_v1_UNQUALIFIED'

    def __init__(self, *, group, max_tokens, hidden, topk, num_experts,
                 library=None, cores=32):
        # Do not call the pinned parent's constructor or widen its EP8 guard.
        # The unchanged inherited methods enforce caller runtime/device identity
        # and retain exact forward/backward/close semantics.
        require_ep16_metadata(ops)
        self.group, self.world_size = group, dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if (self.world_size != 16 or num_experts <= 0 or num_experts % 16
                or num_experts // 16 > 32 or not 1 <= topk <= min(16, num_experts)
                or not 0 < max_tokens <= 16384 or hidden <= 0 or hidden % 128
                or cores != 32):
            raise ValueError('Adapted training requires EP16,32 AIC,local experts<=32 and legal native128 dimensions')
        self.max_tokens, self.hidden = max_tokens, hidden
        self.topk, self.num_experts = topk, num_experts
        self.closed = False
        self._config = get_udma_config()
        self._check_runtime()
        self.gemm = CatlassGemm(library, cores)
        self.last_counts = self.last_shapes = self.last_grads_fp32 = None
        self.last_shared_grads_fp32 = None
