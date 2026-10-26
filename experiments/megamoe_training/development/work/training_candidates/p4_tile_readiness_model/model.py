"""Protocol arithmetic only. No hardware memory-order/Notify extraction claim."""
from dataclasses import dataclass

def integer(value, minimum=0):
    if type(value) is not int or value < minimum:raise ValueError('invalid integer')
    return value

@dataclass(frozen=True)
class Domain:
    kind: str
    owner: str
    queue: str = ''
    def __post_init__(self):
        if self.kind not in ('qp','local') or not isinstance(self.owner,str) or not self.owner or not isinstance(self.queue,str) or (self.kind=='qp') != bool(self.queue):
            raise ValueError('explicit QP or local writer domain required')

@dataclass(frozen=True)
class Write:
    source: int
    expert: int
    row: int                 # expert-local final occurrence, not input token
    domain: Domain
    seq: int                # submission order within this exact domain/epoch
    epoch: int

class Model:
    def __init__(self, counts, records, *, epoch, rowbytes, world, leases=()):
        self.epoch=integer(epoch,1);self.rowbytes=integer(rowbytes,1);self.world=integer(world,1)
        self.counts=tuple(integer(n) for n in counts);self.records=tuple(records)
        self.by_occurrence={};self.by_submission={};self.dependencies={};last={}
        for w in self.records:
            if not isinstance(w,Write) or not isinstance(w.domain,Domain):raise ValueError('normalized Write/Domain required')
            if integer(w.source)>=self.world or integer(w.expert)>=len(self.counts) or integer(w.row)>=self.counts[w.expert]:raise ValueError('occurrence outside actual extent')
            if w.epoch!=self.epoch or type(w.epoch) is not int:raise ValueError('wrong record epoch')
            if integer(w.seq,1)!=last.get(w.domain,0)+1:raise ValueError('submission order must be strict, gapless per domain')
            occurrence=(w.expert,w.row);submission=(w.domain,w.seq)
            if occurrence in self.by_occurrence:raise ValueError('final occurrence has multiple writers')
            self.by_occurrence[occurrence]=w;self.by_submission[submission]=w;last[w.domain]=w.seq
            dep=self.dependencies.setdefault((w.expert,w.row//128),{})
            dep[w.domain]=max(dep.get(w.domain,0),w.seq)
        if len(self.by_occurrence)!=sum(self.counts):raise ValueError('final occurrence coverage incomplete')
        self.frontier={d:0 for d in last};self.completed=set();self.leases={}
        for name,occurrence in leases:
            if not isinstance(name,str) or not name or name in self.leases or tuple(occurrence) not in self.by_occurrence:raise ValueError('invalid primary read lease')
            self.leases[name]=tuple(occurrence)
        self.lease_spec=tuple(self.leases.items());self.backward_done=False;self.outer_epoch_retired=False;self.reclaimed=False

    def check_epoch(self,epoch):
        if type(epoch) is not int or epoch!=self.epoch or self.reclaimed:raise ValueError('old/future/reclaimed epoch')

    def retire_write(self,domain,seq,*,epoch,complete_bytes):
        self.check_epoch(epoch);integer(seq,1);integer(complete_bytes)
        key=(domain,seq)
        if key not in self.by_submission:raise ValueError('unknown producer submission')
        if complete_bytes>self.rowbytes:raise ValueError('completion exceeds full rowbytes')
        if complete_bytes==self.rowbytes:self.completed.add(key)
        # Out-of-order full completion is buffered, never crosses a gap.
        while (domain,self.frontier[domain]+1) in self.completed:self.frontier[domain]+=1
        return self.frontier[domain]

    def ready(self,expert,tile,*,epoch):
        self.check_epoch(epoch);integer(expert);integer(tile)
        key=(expert,tile)
        if key not in self.dependencies:raise ValueError('no actual tile: empty/tail padding is not a consumer')
        return all(self.frontier[d]>=seq for d,seq in self.dependencies[key].items())

    def retire_lease(self,name,*,epoch):
        self.check_epoch(epoch)
        if name not in dict(self.lease_spec):raise ValueError('unknown primary read lease')
        self.leases.pop(name,None) # idempotent, epoch still checked

    def finish_backward(self,*,epoch):self.check_epoch(epoch);self.backward_done=True

    def retire_outer_epoch(self,*,epoch,evidence):
        """Caller assertion of ALL hidden/aux/descriptor termination; not a device proof."""
        self.check_epoch(epoch)
        required={'hidden_terminal','aux_complete','descriptor_epoch_retired'}
        if not isinstance(evidence,dict) or set(evidence)!=required or any(v is not True for v in evidence.values()):
            raise ValueError('all external termination assertions required')
        self.outer_epoch_retired=True

    def can_reclaim(self):
        return not self.reclaimed and self.outer_epoch_retired and self.backward_done and not self.leases and len(self.completed)==len(self.records)

    def next_epoch(self):
        if not self.can_reclaim():raise ValueError('saved Xe / write / primary DMA lease still alive')
        records=[Write(w.source,w.expert,w.row,w.domain,w.seq,self.epoch+1) for w in self.records]
        result=Model(self.counts,records,epoch=self.epoch+1,rowbytes=self.rowbytes,world=self.world,leases=self.lease_spec)
        self.reclaimed=True
        return result
