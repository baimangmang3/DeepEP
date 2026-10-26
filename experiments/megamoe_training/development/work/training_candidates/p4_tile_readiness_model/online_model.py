"""Online issue-order layer. Static expectations do NOT assign transport seq.

No Notify extraction, WQE transport, device seal publication or visibility proof.
"""
from dataclasses import dataclass
from model import Model,Domain,Write,integer

@dataclass(frozen=True)
class Expected:
    source: int
    expert: int
    row: int
    domain: Domain

class OnlineModel(Model):
    """Reuse checked epoch/retire/lifetime operations, assign seq only in issue()."""
    def __init__(self,counts,expected,*,epoch,rowbytes,world,leases=()):
        self.epoch=integer(epoch,1);self.rowbytes=integer(rowbytes,1);self.world=integer(world,1)
        self.counts=tuple(integer(n) for n in counts);self.records=tuple(expected)
        self.by_occurrence={};self.expected_by_tile={}
        for e in self.records:
            if not isinstance(e,Expected) or not isinstance(e.domain,Domain):raise ValueError('normalized Expected/Domain required')
            if integer(e.source)>=world or integer(e.expert)>=len(self.counts) or integer(e.row)>=self.counts[e.expert]:raise ValueError('expected occurrence outside actual extent')
            key=(e.expert,e.row)
            if key in self.by_occurrence:raise ValueError('multiple expected writers')
            self.by_occurrence[key]=e
            self.expected_by_tile.setdefault((e.expert,e.row//128),{}).setdefault(e.domain,set()).add(key)
        if len(self.by_occurrence)!=sum(self.counts):raise ValueError('expected coverage incomplete')
        self.by_submission={};self.issued={};self.frontier={e.domain:0 for e in self.records}
        self.last={d:0 for d in self.frontier};self.completed=set();self.dependencies={};self.leases={}
        for name,occurrence in leases:
            if not isinstance(name,str) or not name or name in self.leases or tuple(occurrence) not in self.by_occurrence:raise ValueError('invalid primary lease')
            self.leases[name]=tuple(occurrence)
        self.lease_spec=tuple(self.leases.items());self.backward_done=False;self.outer_epoch_retired=False;self.reclaimed=False

    def issue(self,source,expert,row,domain,*,epoch):
        self.check_epoch(epoch);integer(source);integer(expert);integer(row)
        key=(expert,row);e=self.by_occurrence.get(key)
        if e is None or e.source!=source or e.domain!=domain:raise ValueError('actual source/writer does not match static expectation')
        if key in self.issued:raise ValueError('duplicate final occurrence issue')
        seq=self.last[domain]+1
        self.last[domain]=seq;self.issued[key]=seq
        self.by_submission[(domain,seq)]=Write(source,expert,row,domain,seq,epoch)
        return seq

    def seal(self,expert,tile,domain,*,epoch):
        self.check_epoch(epoch);integer(expert);integer(tile)
        key=(expert,tile);expected=self.expected_by_tile.get(key,{}).get(domain)
        if expected is None:raise ValueError('no expected tile/domain')
        if not expected.issubset(self.issued):raise ValueError('unissued expected occurrence prevents seal')
        watermark=max(self.issued[occurrence] for occurrence in expected)
        self.dependencies.setdefault(key,{})[domain]=watermark
        return watermark # duplicate same-epoch seal is idempotent

    def ready(self,expert,tile,*,epoch):
        self.check_epoch(epoch);integer(expert);integer(tile)
        key=(expert,tile)
        if key not in self.expected_by_tile:raise ValueError('no actual tile')
        sealed=self.dependencies.get(key,{})
        return set(sealed)==set(self.expected_by_tile[key]) and all(self.frontier[d]>=seq for d,seq in sealed.items())

    def next_epoch(self):
        if not self.can_reclaim():raise ValueError('writes/saved Xe/readers/outer epoch still live')
        result=OnlineModel(self.counts,self.records,epoch=self.epoch+1,rowbytes=self.rowbytes,world=self.world,leases=self.lease_spec)
        self.reclaimed=True
        return result
