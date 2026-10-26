"""Only new online ordering/sealing schedules; no replay of prior18 checks."""
import json,pathlib,sys
from model import Domain
from online_model import Expected,OnlineModel
checks=[]
def reject(f):
    try:f()
    except ValueError:return
    raise AssertionError('bad online protocol accepted')
q0=Domain('qp','gateway','QP0');q1=Domain('qp','gateway','QP1')
expected=[Expected(r%2,0,r,q0) for r in range(129)]
def fresh(table=expected,counts=(129,)):return OnlineModel(counts,table,epoch=11,rowbytes=12288,world=16)
def issue(m,e):return m.issue(e.source,e.expert,e.row,e.domain,epoch=11)
def retire(m,d,seq,n=12288):return m.retire_write(d,seq,epoch=11,complete_bytes=n)
orders=[sorted(expected,key=lambda e:e.source),sorted(expected,key=lambda e:-e.source)]
models=[]
for order in orders:
    m=fresh()
    for e in order:retire(m,e.domain,issue(m,e))
    assert not m.ready(0,0,epoch=11) # completion alone is not a seal
    for tile in (0,1):m.seal(0,tile,q0,epoch=11);assert m.ready(0,tile,epoch=11)
    models.append(m)
assert models[0].issued[(0,0)]!=models[1].issued[(0,0)]
checks.append('identical static expectations plus opposite source arrivals yield different actual seq and correct tiles')
checks.append('retire before seal remains not-ready until explicit complete expected-set seal')
m=fresh()
for e in orders[0]:
    if e.source==0:retire(m,e.domain,issue(m,e))
assert m.frontier[q0]==65 and not m.ready(0,0,epoch=11)
reject(lambda:m.seal(0,0,q0,epoch=11));checks.append('high early-source frontier cannot hide unissued later source or permit seal')
before=(len(m.issued),dict(m.last))
reject(lambda:issue(m,expected[0]));assert before==(len(m.issued),m.last)
reject(lambda:m.issue(9,0,1,q0,epoch=11));reject(lambda:m.issue(1,0,1,q1,epoch=11))
assert not m.ready(0,0,epoch=11);checks.append('duplicate issue and source/domain mismatch reject atomically, cannot reduce expected set')
table=[Expected(0,0,0,q0),Expected(1,0,1,q1)];m=fresh(table,(2,))
for e in table:retire(m,e.domain,issue(m,e))
assert m.seal(0,0,q0,epoch=11)==1 and not m.ready(0,0,epoch=11)
assert m.seal(0,0,q1,epoch=11)==1 and m.ready(0,0,epoch=11)
assert m.seal(0,0,q1,epoch=11)==1;checks.append('cross-QP tile requires independent seals and frontier; duplicate seal idempotent')
m=fresh(table,(2,));a=issue(m,table[0]);b=issue(m,table[1]);m.seal(0,0,q0,epoch=11);m.seal(0,0,q1,epoch=11)
retire(m,q0,a);retire(m,q1,b,12287);assert not m.ready(0,0,epoch=11)
retire(m,q1,b);assert m.ready(0,0,epoch=11);checks.append('seal before completion still requires full rowbytes on each frontier')
m=fresh([Expected(0,0,r,q0) for r in range(3)],(3,))
for e in m.records:issue(m,e)
m.seal(0,0,q0,epoch=11);retire(m,q0,3);assert m.frontier[q0]==0 and not m.ready(0,0,epoch=11)
retire(m,q0,1);retire(m,q0,2);assert m.ready(0,0,epoch=11);checks.append('online seq completion gaps buffer until true contiguous prefix')
for epoch in (10,12):
    reject(lambda:m.seal(0,0,q0,epoch=epoch));reject(lambda:m.issue(0,0,0,q0,epoch=epoch))
reject(lambda:m.retire_write(q0,4,epoch=11,complete_bytes=12288));checks.append('wrong epochs and completion before issue cannot mutate online state')
m.finish_backward(epoch=11);reject(m.next_epoch)
m.retire_outer_epoch(epoch=11,evidence=dict(hidden_terminal=True,aux_complete=True,descriptor_epoch_retired=True));n=m.next_epoch()
assert not n.issued and not n.dependencies and not n.outer_epoch_retired
reject(lambda:n.retire_write(q0,1,epoch=11,complete_bytes=12288));checks.append('persistent Xe/outer obligations and ABA guard inherited; next epoch clears seals and issued order')
empty=fresh([],(0,));reject(lambda:empty.ready(0,0,epoch=11));reject(lambda:empty.seal(0,0,q0,epoch=11));checks.append('empty has no synthetic tile or domain seal')
assert not any(k=='torch' or k.startswith(('torch.','torch_npu','deep_ep')) for k in sys.modules)
out=dict(passed=True,checks=len(checks),items=checks,scope='new independent stdlib online issue/seal model only; prior18 model tests not rerun')
path=pathlib.Path(__file__).with_name('online_cpu_results.json');path.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out,indent=2))
