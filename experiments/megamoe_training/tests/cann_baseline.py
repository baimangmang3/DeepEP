"""CANN forward baseline: two fused communication/GMM ops + separate SwiGLU.

API/count directions verified against deployed work/torch-native-apis.json and
the previously audited bench_verified.py, lines 46-108. Old A3 qualification is
not a qualification of the current 950. Here modes are spelled ai_cpu and ccu.
No backward is provided; compare only with candidate forward.
"""
import torch
import torch.distributed as dist
import torch_npu


class CannForward:
    def __init__(self, x, w1, w2, ids, p, *, comm_mode, group=None):
        if comm_mode not in ('ai_cpu','ccu'):
            raise ValueError('950 audited comm_mode spelling is ai_cpu or ccu; AIV is not qualified')
        self.group=dist.group.WORLD if group is None else group
        self.rank=dist.get_rank(self.group)
        self.world=dist.get_world_size(self.group)
        self.x,self.ids,self.p=x,ids,p
        self.w1,self.w2=w1.detach().bfloat16(),w2.detach().bfloat16()
        self.mode=comm_mode
        self.b,self.h=x.shape
        self.k=ids.shape[1]
        self.local_e=len(w1)
        self.e=self.world*self.local_e
        self.last_exported_x=None
        if x.dtype!=torch.bfloat16 or p.dtype!=torch.float32 or ids.dtype!=torch.int64:
            raise ValueError('Requires BF16 X, int64 ids, FP32 gates')
        # Deployed schema uses positional hcom; avoids historical docs' group kw.
        self.hcom=self.group._get_backend(torch.device('npu')).get_hccl_comm_name(dist.get_rank())
        self.refresh_routing()
        self.metadata={
            'comm_mode':comm_mode,'send_counts':self.sc,'recv_counts':self.rc,
            'received_rows':sum(self.rc),'received_rows_per_expert':self.expert_counts,
            'counts_layout':'send[dest_rank,local_expert], recv[source_rank,local_expert]',
            'inverse_count_direction':'gmm_alltoallv receives recv_counts,send_counts',
            'input_layout':'source rows sorted by global expert; stable token/topk slot order',
            'forward_internal_layout':'local expert then source rank',
            'return_layout':'source global expert sorted order before explicit inverse permutation',
            'weight_layout':'PyTorch [Elocal,2I,H]/[Elocal,H,I], trans_gmm_weight=True',
            'exports_included':'alltoallv_gmm permute_out_flag=True; tuple[2] BF16 Xe retained as last_exported_x for training-forward scope',
            'fused':['alltoallv+GMM1','GMM2+alltoallv'],
            'separate':['X pack','FP32 SwiGLU -> BF16','unpermute','FP32 gate times BF16 Z -> BF16','FP32 topk sum -> BF16'],
            'backward':'not implemented; forward comparison only',
            'dynamic_forward_includes':['device bincount','count all_gather','counts D2H','CPU stable route argsort','order H2D','X pack','unpack/gating/topk'],
            'prepared_plan_only_excludes':['count all_gather','CPU stable route argsort','weight BF16 materialization'],
        }

    def refresh_routing(self):
        """Fresh count exchange/sort; INCLUDED in dynamic full-forward timing.

        CANN takes host counts, so host synchronization here is part of this
        implementation's end-to-end cost, not a native operator kernel cost.
        """
        local=torch.bincount(self.ids.flatten(),minlength=self.e).to(torch.int64)
        peers=[torch.empty_like(local) for _ in range(self.world)]
        dist.all_gather(peers,local,group=self.group)
        self.table=torch.stack(peers).cpu()
        self.sc=self.table[self.rank].tolist()
        self.rc=self.table[:,self.rank*self.local_e:(self.rank+1)*self.local_e].flatten().tolist()
        self.expert_counts=self.table[:,self.rank*self.local_e:(self.rank+1)*self.local_e].sum(0).tolist()
        flat=self.ids.detach().cpu().flatten()
        if flat.numel()!=self.b*self.k or flat.min()<0 or flat.max()>=self.e:
            raise ValueError('Invalid routes')
        self.order=torch.argsort(flat,stable=True).to(self.x.device)
        self.inverse=torch.argsort(self.order)
        self.source_token=self.order//self.k

    def pack(self):
        return self.x.index_select(0,self.source_token).contiguous()

    def up(self, packed):
        gu,_,xe=torch_npu.npu_alltoallv_gmm(
            packed,self.w1,self.hcom,self.world,self.sc,self.rc,
            trans_gmm_weight=True,permute_out_flag=True,comm_mode=self.mode)
        if (xe is None or xe.ndim!=2 or tuple(xe.shape)!=(sum(self.rc),packed.shape[1]) or
                xe.dtype!=torch.bfloat16):
            raise RuntimeError('CANN training-forward scope requires BF16 Xe export in tuple[2] with [received_rows,H] shape')
        self.last_exported_x=xe
        return gu

    @staticmethod
    def activation(gu):
        g,u=gu.float().chunk(2,dim=-1)
        return (torch.nn.functional.silu(g)*u).bfloat16()

    def down(self, a):
        z,_=torch_npu.npu_gmm_alltoallv(
            a,self.w2,self.hcom,self.world,self.rc,self.sc,
            trans_gmm_weight=True,comm_mode=self.mode)
        return z

    def core(self,packed):
        return self.down(self.activation(self.up(packed)))

    def restore(self,packed_z):
        z=packed_z.index_select(0,self.inverse).reshape(self.b,self.k,self.h)
        # FP32 original gates, never silently quantized to BF16 probs.
        contributions=(z.float()*self.p[:,:,None]).bfloat16()
        return contributions.float().sum(dim=1).bfloat16()

    @torch.no_grad()
    def prepared_forward(self):
        """Prepared routing plan scope, never compared with fresh candidate."""
        return self.restore(self.core(self.pack()))

    @torch.no_grad()
    def __call__(self):
        self.refresh_routing()
        return self.prepared_forward()


@torch.no_grad()
def forward_oracle(x,w1,w2,ids,p,chunk=1024,group=None):
    """Independent all-gather/expert torch.mm reference, checking every Y row.

    Uses no CannForward sorting, counts, fused op or SHMEM handle. Accumulates
    weighted BF16 contributions in FP32 then BF16; native SHMEM hierarchy may
    additionally round server partials. The fixed precision limits are external.
    """
    group=dist.group.WORLD if group is None else group
    rank,world=dist.get_rank(group),dist.get_world_size(group)
    def gather(t):
        peers=[torch.empty_like(t) for _ in range(world)]
        dist.all_gather(peers,t.contiguous(),group=group)
        return peers
    xs,indices,gates=map(gather,(x,ids,p))
    result=None
    for src in range(world):
        y=torch.zeros_like(xs[src],dtype=torch.float32)
        for le in range(len(w1)):
            token,slot=torch.where(indices[src]==rank*len(w1)+le)
            for start in range(0,token.numel(),chunk):
                t,k=token[start:start+chunk],slot[start:start+chunk]
                gu=torch.mm(xs[src][t],w1[le].bfloat16().T).bfloat16()
                g,u=gu.float().chunk(2,dim=-1)
                a=(torch.nn.functional.silu(g)*u).bfloat16()
                z=torch.mm(a,w2[le].bfloat16().T).bfloat16()
                value=(z.float()*gates[src][t,k,None]).bfloat16()
                y.index_add_(0,t,value.float())
        dist.reduce(y,dst=src,group=group)
        if rank==src:
            result=y.bfloat16()
    return result
