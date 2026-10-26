"""Small discriminating schedules; no Torch/native/PG imports."""
import dataclasses,hashlib,json,pathlib,sys
from model import Domain,Write,Model
checks=[]
def check(name,f):f();checks.append(name)
def reject(f):
    try:f()
    except ValueError:return
    raise AssertionError('invalid protocol accepted')
def build(counts,rows,leases=()):return Model(counts,rows,epoch=7,rowbytes=12288,world=16,leases=leases)
q0=Domain('qp','sender0','QP0');q1=Domain('qp','sender0','QP1');l0=Domain('local','rank1/core24');l1=Domain('local','rank1/core25')
def complete(m,w,n=12288):return m.retire_write(w.domain,w.seq,epoch=7,complete_bytes=n)

# Non-row-sorted submission order; tile0 includes source0 and source9.
records=[Write(0,0,r,q0,i+1,7) for i,r in enumerate([128]+list(range(64)))]+[Write(9,0,r,q1,i+1,7) for i,r in enumerate(range(64,128))]
m=build([129,0],records)
assert m.dependencies[(0,0)]=={q0:65,q1:64} and m.dependencies[(0,1)]=={q0:1};checks.append('cross-source tile and non-row-sorted sequence max dependencies')
complete(m,records[0]);assert m.ready(0,1,epoch=7) and not m.ready(0,0,epoch=7);checks.append('actual one-row tail ready independently, not final-row ordering')
for w in reversed(records[1:]):complete(m,w)
assert m.ready(0,0,epoch=7);checks.append('out-of-order notices close gaps only after all earlier submissions retire')
for epoch in (6,8,'7'):reject(lambda:m.retire_write(q0,1,epoch=epoch,complete_bytes=12288))
checks.append('old future and wrong-type epoch reject')
gaprows=[Write(0,0,i,q0,i+1,7) for i in range(3)];gap=build([3],gaprows)
assert complete(gap,gaprows[2])==0 and complete(gap,gaprows[0],12287)==0
assert complete(gap,gaprows[0])==1 and complete(gap,gaprows[1])==3
checks.append('explicit out-of-order gap blocks frontier, partial row never closes gap')
reject(lambda:m.ready(1,0,epoch=7));reject(lambda:m.ready(0,2,epoch=7));checks.append('empty expert and padded tail have no wait target')
# Preserve 3:1 independent QPs and interleaved local owners.
rows=[];seq={d:0 for d in (q0,q1,l0,l1)}
for r in range(129):
    d=(q0 if r%4<3 else q1) if r<128 else l0;seq[d]+=1;rows.append(Write(r%16,0,r,d,seq[d],7))
a=build([129],rows)
for w in rows:
    if w.domain!=q1:complete(a,w)
assert not a.ready(0,0,epoch=7);checks.append('3to1 QP frontier cannot merge even after QP0 and tail local completion')
for w in rows:
    if w.domain==q1:complete(a,w)
assert a.ready(0,0,epoch=7);checks.append('all independent QP contributions required')
brows=[Write(0,0,r,l0 if r%2==0 else l1,r//2+1,7) for r in range(129)];b=build([129],brows)
for w in brows:
    if w.domain==l0:complete(b,w)
assert not b.ready(0,0,epoch=7) and b.ready(0,1,epoch=7);checks.append('interleaved local owners and actual tail extent')
crows=[Write(0,0,0,q0,1,7),Write(0,0,1,l0,1,7)];c=build([2],crows,[('FM-DMA', (0,0)),('local-expand',(0,0))])
complete(c,crows[0]);assert not c.ready(0,0,epoch=7);checks.append('primary arrival does not retire downstream final expansion')
assert complete(c,crows[1],12287)==0 and not c.ready(0,0,epoch=7)
complete(c,crows[1]);front=dict(c.frontier);complete(c,crows[1]);assert c.frontier==front;checks.append('full rowbytes required and duplicate full notification idempotent')
assert c.ready(0,0,epoch=7) and not c.can_reclaim();reject(c.next_epoch);checks.append('compute ready cannot release training saved Xe')
c.finish_backward(epoch=7);assert not c.can_reclaim();c.retire_lease('FM-DMA',epoch=7);assert not c.can_reclaim();checks.append('backward finish still retains outstanding DMA/local primary readers')
c.retire_lease('local-expand',epoch=7);c.retire_lease('local-expand',epoch=7);assert not c.can_reclaim();reject(c.next_epoch)
outer=dict(hidden_terminal=True,aux_complete=True,descriptor_epoch_retired=True)
for epoch in (6,8,'7'):reject(lambda:c.retire_outer_epoch(epoch=epoch,evidence=outer))
reject(lambda:c.retire_outer_epoch(epoch=7,evidence=dict(outer,aux_complete=False)))
reject(lambda:c.retire_outer_epoch(epoch=7,evidence=dict(hidden_terminal=True,aux_complete=True)))
assert not c.can_reclaim();checks.append('all writes readers backward done still cannot reuse without complete exact-epoch outer termination')
c.retire_outer_epoch(epoch=7,evidence=outer);c.retire_outer_epoch(epoch=7,evidence=outer);assert c.can_reclaim();n=c.next_epoch()
reject(lambda:n.retire_write(q0,1,epoch=7,complete_bytes=12288));reject(lambda:c.ready(0,0,epoch=7));assert not n.ready(0,0,epoch=8);checks.append('all reader retirement permits next epoch; ABA stale completion cannot authorize reuse')
for bad in ([crows[0],dataclasses.replace(crows[1],row=0)],crows[:1],[dataclasses.replace(crows[0],seq=2),crows[1]],[dataclasses.replace(crows[0],epoch=8),crows[1]]):reject(lambda:build([2],bad))
checks.append('duplicate writer missing occurrence sequence gap wrong-record epoch reject')
reject(lambda:Domain('qp','sender0'));reject(lambda:Domain('local','core24','QP0'));checks.append('producer domain kind and QP identity explicit')
z=build([0,0],[]);assert not z.can_reclaim();z.finish_backward(epoch=7);assert not z.can_reclaim();reject(z.next_epoch)
z.retire_outer_epoch(epoch=7,evidence=outer);assert z.can_reclaim();z2=z.next_epoch();assert not z2.outer_epoch_retired
checks.append('all-empty still requires outer retirement; valid caller evidence permits next epoch and resets proof')
assert not any(k=='torch' or k.startswith(('torch.','torch_npu','deep_ep')) for k in sys.modules)
result=dict(passed=True,checks=len(checks),items=checks,scope='stdlib normalized final-write protocol model only; not Notify extraction/native memory ordering or hardware validation')
p=pathlib.Path(__file__).with_name('cpu_results.json');p.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result,indent=2))
