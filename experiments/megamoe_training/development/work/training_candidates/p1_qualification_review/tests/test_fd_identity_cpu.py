"""Archived real FD snapshots/policy only; no native imports or live capture."""
import ast,copy,hashlib,importlib.util,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1];WS=R.parents[2]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=load('fd_p1_provenance',R/'p1_pointwise/candidate_provenance.py')
B=load('fd_base',WS/'work/DeepEP/experiments/megamoe_training/tests/provenance.py')
G=load('fd_gate',R/'comparison/gate.py');checks=[]
def reject(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('not rejected')
maps=[];archives=[]
for host in ('rack05os7','rack05os8'):
    path=WS/'outputs/p2_qualification_deployment'/host/'collision-cpu.json';r=json.loads(path.read_bytes());raw=r['raw_mapping_result']
    collision=next(iter(r['collision_groups'].values()));roots={pathlib.Path(v['path']).parent.name:pathlib.Path(v['path']).parent for v in collision}
    objects=P.checked_maps(raw,roots);canonical=P.canonical_maps(objects);maps.append(canonical);archives.append(str(path.relative_to(WS)))
    assert len(objects)==len(raw['libraries'])
    for v in collision:
        assert objects[v['path']]['sha256']==v['sha256']
        assert all(objects[v['path']][k]==v[k] for k in ('device','inode','map_inode','map_device_major','map_device_minor','map_entries'))
    checks.append('actual dual_C all FD records preserved:'+host)
    moved=copy.deepcopy(raw);newroots={name:pathlib.Path('/model/moved')/name for name in roots}
    for v in moved['libraries']:
        for name,old in roots.items():
            if pathlib.Path(v['path']).is_relative_to(old):v['path']=str(newroots[name]/pathlib.Path(v['path']).relative_to(old));break
    assert P.canonical_maps(P.checked_maps(moved,newroots))==canonical
    checks.append('cross-root logical identity model:'+host)
    unknown=copy.deepcopy(raw);v=copy.deepcopy(collision[0]);v['path']='/unknown/'+pathlib.Path(v['path']).name;unknown['libraries'].append(v)
    reject(lambda:P.checked_maps(unknown,roots));checks.append('unknown same basename rejected:'+host)
    z=copy.deepcopy(raw);z['errors']=['Deleted actual mapping'];reject(lambda:P.checked_maps(z,roots));checks.append('FD collector error rejected:'+host)
    z=copy.deepcopy(raw);z['libraries'][0]['map_inode']+=1;reject(lambda:P.checked_maps(z,roots));checks.append('FD inode mismatch rejected:'+host)
    base=r['base_identity'];selected=next(v for v in raw['libraries'] if pathlib.Path(v['path']).name=='libmegamoe_pointwise.so')
    model=dict(content=dict(candidate_files=[dict(role='native.pointwise',bytes=selected['bytes'],sha256=selected['sha256'])]))
    P.bind_selected_maps(model,base,selected['path'],objects);checks.append('all actual base/PW exact-path bindings:'+host)
    core=base['diagnostic_paths']['native.deepep_core']
    z=copy.deepcopy(objects);z.pop(core);reject(lambda:P.bind_selected_maps(model,base,selected['path'],z));checks.append('missing exact selected FD rejected:'+host)
    z=copy.deepcopy(objects);z[core]['sha256']='f'*64;reject(lambda:P.bind_selected_maps(model,base,selected['path'],z));checks.append('selected extension replaced rejected:'+host)
assert maps[0]==maps[1];checks.append('two actual hosts equal canonical identity despite physical dev/inode differences')
unit=WS/'outputs/training_results_20261026_0153_p1_main_routed_once/p1-host-v2-main-bf16-balanced/rack05os7/rank0.json'
old=json.loads(unit.read_bytes())['identity'];reject(lambda:P.validate_identity(old,B));checks.append('old P1 schema rejected')
a=copy.deepcopy(old);a['schema']='p1-training-2';f=R/'p1_pointwise/mapped_identity.py'
a['content']['candidate_files'].append(dict(role='source/mapped_identity.py',sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size));a['content']['mapped_libraries']=maps[0]
def seal(v):v['content_digest']=P.digest(v['content']);return v
seal(a);P.validate_identity(a,B)
for change in ('removed','replaced','lazy_added'):
    b=copy.deepcopy(a);m=b['content']['mapped_libraries'];role=next(k for k in m if k.startswith('package/torch_npu/') and '/_C.' in k)
    if change=='removed':m.pop(role)
    elif change=='replaced':m[role]['sha256']='a'*64
    else:m['loaded/lib_fd_lazy_example.so']=dict(bytes=7,sha256='d'*64)
    seal(b);report=dict(identity=a,identity_final=b,final_identity_peer_status=[dict(rank=n,error=None,digest=b['content_digest']) for n in range(16)])
    if change=='lazy_added':P.final_identity(a,b,B);G.require_p1_final_identity(report,b,lambda v:P.validate_identity(v,B),16)
    else:reject(lambda:P.final_identity(a,b,B));reject(lambda:G.require_p1_final_identity(report,b,lambda v:P.validate_identity(v,B),16))
    checks.append('producer and consumer mapped '+change)
assert f.read_bytes()==(WS/'work/training_candidates/p3_training/mapped_identity.py').read_bytes();checks.append('reviewed fstat FD helper copied byte-exact')
source=(R/'p1_pointwise/candidate_provenance.py').read_text();tree=ast.parse(source)
p2=(WS/'work/training_candidates/p2_backward_overlap/candidate_provenance.py').read_text();other=ast.parse(p2)
for name in ('valid_map_role','imported_package_roots','checked_maps','canonical_maps'):
    get=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(get(tree))==ast.dump(get(other))
checks.append('namespace helper AST equal reviewed P2')
assert not any(n=='torch' or n.startswith(('torch.','torch_npu','deep_ep')) for n in sys.modules)
result=dict(passed=True,checks=len(checks),items=checks,archives=archives,scope='Offline archived FD evidence and identity models; no current collector/device/qualification run')
(R/'tests/fd_cpu_results.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf8'));print(json.dumps(result))
