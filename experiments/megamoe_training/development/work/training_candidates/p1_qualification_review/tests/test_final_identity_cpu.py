"""Stdlib fault models/archive checks only; creates no training qualification."""
import ast,copy,hashlib,importlib.util,json,pathlib,sys,tempfile,types
ROOT=pathlib.Path(__file__).resolve().parents[1]
WS=ROOT.parents[2]
sys.path.insert(0,str(ROOT/'p1_pointwise'))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
P=load('candidate_provenance',ROOT/'p1_pointwise/candidate_provenance.py')
sys.modules['candidate_provenance']=P
V=load('review_validator',ROOT/'p1_pointwise/validate_training_p1.py')
G=load('review_gate',ROOT/'comparison/gate.py')
B=load('base_provenance',WS/'work/DeepEP/experiments/megamoe_training/tests/provenance.py')
passed=[]
def check(name,action):action();passed.append(name)
def rejects(action):
    try:action()
    except Exception:return
    raise AssertionError('failure was not rejected')
def sealed(value):
    value['content']['base_identity']['content_digest']=B.canonical_digest(value['content']['base_identity']['content'])
    value['content_digest']=P.digest(value['content']);return value
MAIN=WS/'outputs/training_results_20261026_0153_p1_main_routed_once/p1-host-v2-main-bf16-balanced'
original=json.loads((MAIN/'rack05os7/rank0.json').read_bytes())
initial=copy.deepcopy(original['identity']);initial['schema']='p1-training-2'
initial['content']['candidate_files'].append(dict(role='source/mapped_identity.py',bytes=(ROOT/'p1_pointwise/mapped_identity.py').stat().st_size,sha256=hashlib.sha256((ROOT/'p1_pointwise/mapped_identity.py').read_bytes()).hexdigest()))
initial['content']['mapped_libraries']={'loaded/lib_model.so':dict(bytes=1,sha256='c'*64)};sealed(initial)
final=copy.deepcopy(initial)
final['content']['base_identity']['content']['files'].append(dict(role='loaded_dependency/lib_test_lazy.so',bytes=1,sha256='a'*64));sealed(final)
validate=lambda value:P.validate_identity(value,B)
def peerlist(identity):return [dict(rank=r,error=None,digest=identity['content_digest']) for r in range(16)]
def report(a=initial,b=final):
    r=copy.deepcopy(original);r.update(identity=copy.deepcopy(a),identity_final=copy.deepcopy(b),final_identity_peer_status=peerlist(b));return r
check('legal lazy addition transition',lambda:P.final_identity(initial,final,B))
check('legal lazy addition final matches current',lambda:G.require_p1_final_identity(report(),final,validate,16))
check('unchanged identity',lambda:G.require_p1_final_identity(report(initial,initial),initial,validate,16))
for change in ('source','pointwise','selected_native','dependency','removed_dependency','version','precision','execution','invalid_added_role'):
    def test(change=change):
        z=copy.deepcopy(final);c=z['content'];bc=c['base_identity']['content'];fs=bc['files']
        if change=='source':next(v for v in c['candidate_files'] if v['role']=='source/training_adapter.py')['sha256']='b'*64
        elif change=='pointwise':next(v for v in c['candidate_files'] if v['role']=='native.pointwise')['sha256']='b'*64
        elif change=='selected_native':next(v for v in fs if v['role']=='native.catlass')['sha256']='b'*64
        elif change=='dependency':next(v for v in fs if v['role'].startswith('loaded_dependency/') and v['role']!='loaded_dependency/lib_test_lazy.so')['sha256']='b'*64
        elif change=='removed_dependency':fs.remove(next(v for v in fs if v['role'].startswith('loaded_dependency/') and v['role']!='loaded_dependency/lib_test_lazy.so'))
        elif change=='version':bc['versions']['torch']='different'
        elif change=='precision':bc['precision']['matmul_allow_hf32']=True
        elif change=='execution':c['execution']['seed']+=1
        else:fs.append(dict(role='source/new_unreviewed',sha256='b'*64,bytes=1))
        sealed(z)
        rejects(lambda:P.final_identity(initial,z,B));rejects(lambda:G.require_p1_final_identity(report(initial,z),z,validate,16))
    check('reject '+change+' in both producer and consumer',test)
for change in ('missing_final','peer_error','peer_missing','peer_digest','local_error','final_current'):
    def test(change=change):
        r=report()
        if change=='missing_final':r.pop('identity_final')
        elif change=='peer_error':r['final_identity_peer_status'][12]['error']='collector failed'
        elif change=='peer_missing':r['final_identity_peer_status'].pop()
        elif change=='peer_digest':r['final_identity_peer_status'][12]['digest']='b'*64
        elif change=='local_error':r['final_identity_error']='failed'
        rejects(lambda:G.require_p1_final_identity(r,initial if change=='final_current' else final,validate,16))
    check('reject '+change,test)

class Host:
    def __init__(self,mode=None):self.mode=mode;self.events=[]
    def rank(self):return 0
    def size(self):return 2
    def all_gather(self,value,stage):
        self.events.append(stage);peers=[copy.deepcopy(value),copy.deepcopy(value)];peers[1]['rank']=1
        if self.mode=='exchange_error' and stage=='P1/final-identity':raise OSError('host exchange failure')
        if self.mode=='remote_final' and stage=='P1/final-identity':peers[1]['error']='remote collector failure'
        if self.mode=='remote_prior' and stage=='P1/before-PG-teardown':peers[1]['error']='remote body failure'
        if self.mode=='remote_pg' and stage=='P1/after-PG-teardown':peers[1]['errors']=['remote PG failure']
        return peers
    def close(self):self.events.append('host.close')
class Dist:
    def __init__(self,h,fail=False):self.h=h;self.fail=fail
    def is_initialized(self):return True
    def destroy_process_group(self):
        self.h.events.append('PG.destroy')
        if self.fail:raise OSError('PG destroy failure')
def fault(name,mode=None,body=False,collector=False,pg=False,save=False):
    h=Host(mode);r={'all_passed':True};first=ValueError('first body failure') if body else None
    def collect():
        if collector:raise LookupError('first collector failure')
        return final
    def write():
        h.events.append('save')
        if save:raise OSError('save failure')
    error=V.finish_validation_host(h,r,Dist(h,pg),write,collect,initial,B,first)
    if first is not None:assert error is first
    elif collector:assert isinstance(error,LookupError)
    else:assert error is not None
    assert r['all_passed'] is False and h.events[-1]=='save'
    assert h.events.index('PG.destroy')<h.events.index('host.close')<h.events.index('save')
    assert h.events[0]=='P1/final-identity' and 'P1/before-PG-teardown' in h.events and 'P1/after-PG-teardown' in h.events
for name,kw in (('body+collector+PG first error',dict(body=True,collector=True,pg=True)),('collector+exchange first error',dict(collector=True,mode='exchange_error')),('remote final failure',dict(mode='remote_final')),('remote prior failure',dict(mode='remote_prior')),('remote PG failure',dict(mode='remote_pg')),('PG failure',dict(pg=True)),('report IO failure',dict(save=True))):
    check(name,lambda name=name,kw=kw:fault(name,**kw))

def archive_compat():
    for label,backend in (('20261025_0658_ep16-user-shape-bf16','shmem_catlass'),('20261025_0703_ep16-user-shape-shared-bf16','shmem_catlass'),('20261025_0714_ep16-cann-ai-cpu-user-shape','cann_ai_cpu')):
        directory=WS/'outputs/training_qualifications'/label;r=json.loads((directory/'rank0.json').read_bytes())
        evidence=G.qualify(directory,backend,16,r['shape'],r['seed'],r['shared'],r['identity'],r['limits'],B.validate_identity)
        assert len(evidence)==16
check('three actual P0 shared/routed and CANN archives remain compatible',archive_compat)
def full_gate():
    with tempfile.TemporaryDirectory() as tmp:
        d=pathlib.Path(tmp)
        for path in MAIN.rglob('rank*.json'):
            r=json.loads(path.read_bytes());r.update(identity=initial,identity_final=final,final_identity_peer_status=peerlist(final));(d/path.name).write_bytes(json.dumps(r).encode())
        assert len(G.qualify(d,original['backend'],16,original['shape'],original['seed'],False,final,original['limits'],validate))==16
        p=d/'rank12.json';r=json.loads(p.read_bytes());r.pop('identity_final');p.write_bytes(json.dumps(r).encode())
        rejects(lambda:G.qualify(d,original['backend'],16,original['shape'],original['seed'],False,final,original['limits'],validate))
        r['identity_final']=final
        for field in ('case_cleanup','metrics'):
            bad=copy.deepcopy(r)
            if field=='case_cleanup':bad.pop(field)
            else:bad['cases'][0]['metrics'].pop('dx')
            p.write_bytes(json.dumps(bad).encode())
            rejects(lambda:G.qualify(d,original['backend'],16,original['shape'],original['seed'],False,final,original['limits'],validate))
        p.unlink()
        rejects(lambda:G.qualify(d,original['backend'],16,original['shape'],original['seed'],False,final,original['limits'],validate))
check('actual main metrics fixture full16 gate with modeled lazy final; missing rank12 final rejects',full_gate)
def immutable_math():
    old=ast.parse((WS/'work/training_candidates/p1_host_v2/p1_pointwise/validate_training_p1.py').read_text());new=ast.parse((ROOT/'p1_pointwise/validate_training_p1.py').read_text())
    def case(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Subscript) and ast.unparse(n.iter)=="execution['cases']")
    assert ast.dump(case(old))==ast.dump(case(new))
    for name in ('close_case_host','finish_host_report','prepare'):
        assert ast.dump(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name))==ast.dump(next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name))
    assert not any(k.startswith(('torch','deep_ep')) for k in sys.modules)
check('case math and close/PG protocol AST unchanged; no Torch/DeepEP imported',immutable_math)
result=dict(passed=True,checks=len(passed),items=passed,scope='Offline stdlib fault models and archived metric fixtures; no new training qualification/device/performance result',unchanged_originals=True)
(ROOT/'tests/cpu_results.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf8'))
print(json.dumps(result,indent=2))
