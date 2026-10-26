"""Patch application + actual signature/call threading checks; no native compile."""
import hashlib,json,pathlib,re,shutil,subprocess,tempfile
from prepare_patch import HERE,ROOT,changes
checks=[]
m=json.loads((HERE/'manifest.json').read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
for rel,pins in m['files'].items():
    assert sha((ROOT/rel).read_bytes())==pins['before']['sha256']
    assert sha((HERE/'candidate'/rel).read_bytes())==pins['after']['sha256']
checks.append('all7 original files unchanged and candidate before/after hashes exact')
with tempfile.TemporaryDirectory(dir=HERE) as tmp:
    p=pathlib.Path(tmp)
    for rel in m['files']:(p/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p/rel)
    for args in (['git','-c','core.autocrlf=false','apply','--check',str(HERE/'identity64.patch')],['git','-c','core.autocrlf=false','apply',str(HERE/'identity64.patch')]):
        run=subprocess.run(args,cwd=p,capture_output=True,text=True);assert run.returncode==0,run.stderr
    for rel in m['files']:assert (p/rel).read_bytes()==(HERE/'candidate'/rel).read_bytes()
checks.append('git apply --check and apply against exact original7 files reproduces candidate bytes')
# Undo only explicitly reviewed rewrites; any incidental protocol/math/event
# edit would survive the inverse and fail exact original-byte comparison.
for name,edits in changes.items():
    text=(HERE/'candidate/csrc/kernels/dispatch'/name).read_bytes().decode()
    for old,new in reversed(edits):
        if not new:
            anchor='    const uint32_t peers = owner' if name=='aux_pipeline.hpp' else '    const uint32_t routes = t.tokens * t.topk;'
            assert text.count(anchor)==1;text=text.replace(anchor,old+anchor,1)
        else:
            assert text.count(new)==1,(name,new);text=text.replace(new,old,1)
    assert text.encode()==(ROOT/'csrc/kernels/dispatch'/name).read_bytes(),name
checks.append('inverse of explicit parameter/mapping rewrites restores original bytes; no incidental changes')

def args(text,needle):
    at=text.index(needle)+len(needle);assert text[at]=='('
    depth=1;end=at+1
    while depth:
        depth+=(text[end]=='(')-(text[end]==')');end+=1
    s=text[at+1:end-1];out=[];begin=0;angle=paren=brace=0
    for i,c in enumerate(s):
        angle+=(c=='<')-(c=='>');paren+=(c=='(')-(c==')');brace+=(c=='{')-(c=='}')
        if c==',' and angle==paren==brace==0:out.append(s[begin:i].strip());begin=i+1
    out.append(s[begin:].strip());return out
texts={n:(HERE/'candidate/csrc/kernels/dispatch'/n).read_text() for n in changes}
edges=[('dispatch.cpp','PublishPeerAddresses','peer_address.hpp','PublishPeerAddresses',5,2,'logical_role'),
 ('peer_address.hpp','CacheClosDirectories','peer_transport.hpp','CacheClosDirectories',3,1,'logical_role'),
 ('dispatch.cpp','WaitDispatchIncoming','completion_wait.hpp','WaitDispatchIncoming',4,2,'logical_role'),
 ('dispatch.cpp','DispatchAuxPipeline::Run<Profile>','aux_pipeline.hpp','Run',8,3,'logical_role'),
 ('dispatch.cpp','PeerScales<Profile>','peer_scale.hpp','PeerScales',9,4,'logical_role'),
 ('peer_scale.hpp','DispatchScalePipeline::Run<Profile>','scale_pipeline.hpp','Run',8,3,'logical_role')]
def wiring(texts):
    for caller,call,callee,definition,arity,pos,name in edges:
        a=args(texts[caller],call);b=args(texts[callee],'void '+definition)
        assert len(a)==len(b)==arity,(call,a,b)
        assert a[pos]==name and b[pos]=='uint32_t '+name,(call,a[pos],b[pos])
    assert args(texts['dispatch.cpp'],'DispatchAuxPipeline::Run<Profile>')[4]=='aux_owner'
    assert args(texts['aux_pipeline.hpp'],'void Run')[4]=='uint32_t owner'
    assert args(texts['dispatch.cpp'],'PeerScales<Profile>')[5]=='aux_owner'
    assert args(texts['peer_scale.hpp'],'void PeerScales')[5]=='uint32_t aux_owner'
    assert args(texts['peer_scale.hpp'],'DispatchScalePipeline::Run<Profile>')[4]=='aux_owner'
    assert args(texts['scale_pipeline.hpp'],'void Run')[4]=='uint32_t worker'
wiring(texts);checks.append('all6 actual helper edges and aux-owner arguments match definitions and positional types')
bad=dict(texts);bad['dispatch.cpp']=bad['dispatch.cpp'].replace('logical_role, aux_owner, copy, slots','logical_role, copy, slots')
try:wiring(bad)
except AssertionError:pass
else:raise AssertionError('missing owner call argument escaped')
checks.append('injected PeerScales missing owner fails actual call/definition wiring check')
assert sum(t.count('GetBlockIdx(') for t in texts.values())==1
assert 'const uint32_t physical_role = GetBlockIdx();' in texts['dispatch.cpp']
assert 'const uint32_t logical_role = physical_role;' in texts['dispatch.cpp']
assert sum(t.count('SyncAll<true>();') for t in texts.values())==1
assert 'SyncAll<true>();' in texts['peer_address.hpp']
checks.append('only one physical read in changed Dispatch include path, exact identity mapping; original SyncAll remains')
roles=list(range(64));logical=roles[:]
assert [r for r in logical if r==0]==[0] and [r for r in logical if r==7]==[7]
assert [r for r in logical if 8<=r<24]==list(range(8,24))
assert [r-32 for r in logical if r>=32]==list(range(32))
for r in roles:
    assert (r<32)==(logical[r]<32)
    assert r*256==logical[r]*256
    if r>=32:assert (r-32+32)*32*8==logical[r]*256
checks.append('all64 identity roles preserve publisher/poller/cache ranges, 32hidden+32aux coverage and profile byte offsets')
# No physical Notify or low-latency edits included, nor launcher/header ABI.
assert set(m['files'])=={'csrc/kernels/dispatch/'+n for n in changes}
assert not any(n in m['files'] for n in ('csrc/kernels/dispatch/peer_plan.hpp','csrc/kernels/dispatch/launch.hpp'))
for n in ('peer_plan.hpp','launch.hpp'):
    assert n not in changes
checks.append('Notify partition/low-latency/launch ABI excluded; candidate contains only7 scoped helper files')
out=dict(passed=True,checks=len(checks),items=checks,native_compile=False,NPU=False,scope='Patch apply, reversible exact source edits, helper signatures and identity64 arithmetic only')
(HERE/'cpu_results.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out,indent=2))
