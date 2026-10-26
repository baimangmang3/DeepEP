"""Only new component-entry/identity/ABI policy; no Torch/native/device execution."""
import ast,copy,ctypes,hashlib,importlib.util,json,math,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
V=load('p4_component_entry',ROOT/'validate_component.py');I=load('p4_component_identity',ROOT/'component_identity.py')
from p4_wave_compute.contract import plan,I64,U32
checks=[]
def reject(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('Expected refusal')
r=subprocess.run([sys.executable,'-I',str(ROOT/'validate_component.py')],capture_output=True,text=True);assert r.returncode==0,r.stderr
p=json.loads(r.stdout);assert len(p['configuration']['cases'])==8 and p['action'].startswith('plan');checks.append('actual default eight-small-case plan no Torch')
r=subprocess.run([sys.executable,'-I',str(ROOT/'validate_component.py'),'--run-npu'],capture_output=True,text=True);assert r.returncode==2;checks.append('explicit run missing SO/SHA/output rejected before Torch')
assert {(v['H'],v['I']) for v in V.CASES}>={(128,384),(384,128)} and V.CASES[-1].get('concurrent') is True
assert len({v['name'] for v in V.CASES})==8 and max(sum(v['counts']) for v in V.CASES)==1536
for case in V.CASES:plan(sum(case['counts']),case['H'],case['I'],case['counts'])
checks.append('distinct fixed coverage K-switch/Ntails/512tails/empty/skew and last bounded dual-stream')
metric=lambda **kw:V.summary(kw.get('n',1),kw.get('bad',0),kw.get('finite',True),0.,0.,kw.get('err2',0.),kw.get('ref2',1.),[1,128],'torch.bfloat16')
assert metric(err2=.02**2)['passed'] and not metric(err2=.02001**2)['passed'];checks.append('fixed RMS2percent boundary')
assert metric(err2=.003**2,ref2=0.)['passed'] and not metric(err2=.00301**2,ref2=0.)['passed'];checks.append('zero-reference fixed absolute RMS boundary')
assert not metric(bad=1)['passed'] and not metric(finite=False)['passed'];checks.append('badcount and finite always fail')
assert metric(n=0,ref2=0.)['passed'];checks.append('empty metric no max reduction on empty tensors')

def fd(path='/native/libmegamoe_wave_compute.so',sha='a'*64,ino=1):
    return dict(path=path,sha256=sha,bytes=7,inode=ino,map_inode=ino,map_device_major=1,map_device_minor=2,map_entries=[{'line':1}])
def identity():
    content=dict(sources={'native':'f'*64},configuration={'seed':4317},precision={'hf32':False})
    return dict(schema=I.SCHEMA,content=content,content_digest=I.digest(content),actual_maps=dict(errors=[],libraries=[fd()]))
a=identity();b=copy.deepcopy(a);b['actual_maps']['libraries'].append(fd('/native/liblazy.so','c'*64,2));I.compare(a,b);checks.append('lazy mapped library addition accepted')
for mode in ('removed','replaced','inode','source','schema','maperror'):
    b=copy.deepcopy(a)
    if mode=='removed':b['actual_maps']['libraries']=[]
    elif mode=='replaced':b['actual_maps']['libraries'][0]['sha256']='b'*64
    elif mode=='inode':b['actual_maps']['libraries'][0].update(inode=2,map_inode=2)
    elif mode=='source':b['content']['sources']['native']='b'*64;b['content_digest']=I.digest(b['content'])
    elif mode=='schema':b['schema']='p3-training-2'
    else:b['actual_maps']['errors']=['deleted']
    reject(lambda:I.compare(a,b));checks.append('final rejects '+mode)
table=I.mapped_table(a['actual_maps']);I.selected_library(table,'/native/libmegamoe_wave_compute.so','a'*64);checks.append('exact selected actual FD positive')
reject(lambda:I.selected_library(table,'/missing/libmegamoe_wave_compute.so','a'*64));reject(lambda:I.selected_library(table,'/native/libmegamoe_wave_compute.so','b'*64));checks.append('selected exact path/SHA mismatch reject')
conflict=dict(table);conflict['/other/libmegamoe_wave_compute.so']=fd('/other/libmegamoe_wave_compute.so','b'*64,2)
reject(lambda:I.selected_library(conflict,'/native/libmegamoe_wave_compute.so','a'*64));checks.append('conflicting P4 SO basename rejects')
first=RuntimeError('native first');item={}
def fail():raise ValueError('final identity second')
assert I.observe_final(item,fail,a,first) is first and item['passed'] is False and 'identity_final_traceback' in item;checks.append('first native error survives final identity failure')
assert isinstance(I.observe_final({},fail,a),ValueError);checks.append('final-only failure rejects successful body')

class Params(ctypes.Structure):
    _fields_=[(n,ctypes.c_void_p) for n in ('x','w1','w2','gu','a','z','flags')]+[(n,ctypes.c_uint32) for n in ('rows','h','i')]+[('counts',ctypes.c_uint32*4),('rowBegin',ctypes.c_uint32*4),('jobBegin',ctypes.c_uint32*5)]
assert ctypes.sizeof(ctypes.c_void_p)==8 and ctypes.sizeof(Params)==120
assert {n:getattr(Params,n).offset for n in ('rows','h','i','counts','rowBegin','jobBegin')}==dict(rows=56,h=60,i=64,counts=68,rowBegin=84,jobBegin=100);checks.append('LP64 model Params120 and fixed-width field offsets')
t=ast.parse((ROOT/'wave_compute.py').read_text());assignment=next(n for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Attribute) and v.attr=='argtypes' for v in n.targets))
argtypes=eval(compile(ast.Expression(assignment.value),'ctypes_argtypes','eval'),{'ctypes':ctypes})
assert len(argtypes)==13 and argtypes[:7]==[ctypes.c_void_p]*7 and argtypes[7:10]==[ctypes.c_uint32]*3 and argtypes[11:]==[ctypes.c_uint64,ctypes.c_void_p];checks.append('13 external C ABI pointer/dimension/count/flagbytes/stream types')
big=plan((1<<24)+1,128,128,[(1<<24)+1,0,0,0],I64)
offset=(big['rows']-1)*256+255;assert offset>U32 and offset<I64
reject(lambda:plan(U32+1,128,128,[U32+1,0,0,0],I64));reject(lambda:plan(U32,128,1<<20,[U32,0,0,0],I64));checks.append('large signed64 arithmetic model accepts >U32 element offset; row/job overflow rejects')
layout=ROOT.parents[1]/'DeepEP/third-party/catlass/include/tla/layout.hpp';text=layout.read_text(encoding='utf8')
assert 'MakeStride((int64_t)cols, Int<1>{})' in text and 'MakeStride(Int<1>{}, (int64_t)rows)' in text and 'return coord * stride;' in text;checks.append('actual Row/Col strides are int64 before coordinate multiply; no U32 linear-offset bug established')
original=json.loads((ROOT/'manifest.json').read_bytes())
for name,v in original['source_inventory'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==v['sha256']
assert hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()=='deec8aea2e29c9a365fbbc6ddbc51fe3a64328db4173f5b9b72f97ead3e3f9d9';checks.append('build-authorized original manifest and six files untouched')
assert not any(n=='torch' or n.startswith(('torch.','torch_npu','deep_ep')) for n in sys.modules)
result=dict(passed=True,checks=len(checks),items=checks,scope='New entry/identity/ABI arithmetic policy CPU only; LP64 model not compiled ABI, no Torch/native/device/numerical qualification.')
(ROOT/'tests/component_validator_cpu_results.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result,indent=2))
