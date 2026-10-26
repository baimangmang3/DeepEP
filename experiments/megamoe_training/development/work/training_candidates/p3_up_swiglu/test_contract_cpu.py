"""Offline schedule/ABI/wrapper guard checks. Never imports real Torch."""
import concurrent.futures,hashlib,json,sys,tempfile,types
from pathlib import Path
from contract import plan,jobs,product

def coverage(p):
    js=list(jobs(p));assert len(js)==p['jobs']
    assert sorted(q['job'] for q in js)==list(range(len(js)))
    seen=set();areas=[0,0];consumers=0
    for q in js:
        assert p['counts'][q['expert']]>0 and q['aic']==q['job']%32
        assert q['cols'] in (128,256) and q['cols']%64==0
        key=(q['expert'],(q['global_row']-p['row_prefix'][q['expert']])//128,q['col']//256)
        assert key not in seen;seen.add(key)
        assert q['global_row']+q['rows']<=p['row_prefix'][q['expert']+1]
        assert q['col']+q['cols']<=p['I']
        # Independent explicit vector row enumeration, including 8-row tails.
        consumed=[]
        for half in (0,1):
            start=half*64;end=min(q['rows'],(half+1)*64)
            if start>=end:continue
            consumers+=1
            assert (q['aic']*2+half)//2==q['aic']
            for r in range(start,end,8):consumed.extend(range(r,min(end,r+8)))
        assert consumed==list(range(q['rows']))
        areas[0]+=q['rows']*q['cols']*2;areas[1]+=len(consumed)*q['cols']
    # Unique rectangular row/column tiles exactly partition actual expert domains.
    expected={(e,r,c) for e,n in enumerate(p['counts']) for r in range((n+127)//128)
              for c in range((p['I']+255)//256)}
    assert seen==expected and areas==[p['rows']*p['I']*2,p['rows']*p['I']]
    return dict(jobs=len(js),nonempty_consumers=consumers,GU_elements=areas[0],A_elements=areas[1])

def run():
    checks=[]
    def ok(name,fn):fn();checks.append(name)
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Expected rejection')
    for n in (0,1,7,8,9,17,63,64,65,127,128,129,255,256,257):
        for i in (128,256,384,2048,6144):
            for counts in ([n]*4,[0,n,0,0],[n,0,1,0] if n else [0]*4):
                ok(f'partition:{counts}:I{i}',lambda c=counts,i=i:coverage(plan(sum(c),128,i,c)))
    main=coverage(plan(65536,6144,2048,[16384]*4))
    checks.append('main65536_exact_partition')
    for f in (lambda:plan(1,128,128,[0]*4),lambda:plan(1,128,128,[True,0,0,0]),
              lambda:plan(0,127,128,[0]*4),lambda:plan(0,128,129,[0]*4),
              lambda:plan(1<<32,128,128,[1<<32,0,0,0]),
              lambda:plan(0,128,128,[0]*3),lambda:plan(0,128,128,[0,0,0,-1]),
              lambda:plan(0,128,128,[0]*4,1),lambda:product(1<<62,4),
              lambda:plan((1<<32)-1,128,1<<20,[(1<<32)-1,0,0,0],(1<<63)-1)):
        ok('reject_boundary',lambda f=f:reject(f))
    p=plan(1,128,128,[1,0,0,0]);ok('exact_cap',lambda:plan(1,128,128,[1,0,0,0],p['total_bytes']))
    ok('first_byte_over_cap',lambda:reject(lambda:plan(1,128,128,[1,0,0,0],p['total_bytes']-1)))
    # Real wrapper executed against metadata-only Torch stand-in. Every device action logged.
    events=[];alloc=[]
    class Dev:
        type='npu'
    device=Dev()
    class Tensor:
        def __init__(self,shape,dtype='bf16',ptr=None):
            self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device=device
            self.requires_grad=False;self.ptr=ptr or id(self);self.records=[]
        def is_contiguous(self):return True
        def untyped_storage(self):return self
        def data_ptr(self):return self.ptr
        def record_stream(self,s):self.records.append(s)
    class Ctx:
        def __enter__(self):events.append('device')
        def __exit__(self,*a):pass
    stream=types.SimpleNamespace(npu_stream=17)
    def allocate(shape,**kw):
        t=Tensor(shape,kw['dtype']);alloc.append(t);events.append('allocate');return t
    torch=types.SimpleNamespace(Tensor=Tensor,bfloat16='bf16',int32='i32',empty=allocate,zeros=allocate,
        npu=types.SimpleNamespace(device=lambda d:Ctx(),current_stream=lambda d:stream))
    sys.modules['torch']=torch
    from up_swiglu import GroupedUpSwiGLU
    op=object.__new__(GroupedUpSwiGLU);op.cap_bytes=12<<30
    calls=[]
    def fn(*args):calls.append(args);return 0
    op.fn=fn
    x=Tensor((17,128));w=Tensor((4,256,128))
    before=len(events)
    for args in ((x,w,[1,0,0,0]),(x,Tensor((3,256,128)),[17,0,0,0]),(Tensor((17,128),ptr=w.ptr),w,[17,0,0,0])):
        reject(lambda args=args:op(*args))
    assert len(events)==before;checks.append('basic_rejections_before_any_device_action')
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda _:op(x,w,[17,0,0,0]),range(2)))
    results.append(op(x,w,[17,0,0,0]))
    ptrs=[t.data_ptr() for r in results for t in (r.gu,r.a,r.flags)]
    assert len(set(ptrs))==9 and len(calls)==3
    assert all(len(t.records)==1 for r in results for t in (r.gu,r.a,r.flags))
    assert len(x.records)==len(w.records)==3
    checks.extend(['repeat_and_concurrent_private_buffers','all_inputs_outputs_flags_record_stream'])
    before=len(calls);r=op(Tensor((0,128)),w,[0]*4)
    assert len(calls)==before and r.flags.shape==(0,32);checks.append('zero_rows_no_native_launch')
    def failed(*args):return 3
    op.fn=failed;before=len(x.records)
    try:op(x,w,[17,0,0,0])
    except RuntimeError:pass
    else:raise AssertionError('Native rejection swallowed')
    assert len(x.records)==before+1 and all(t.records for t in alloc[-3:])
    checks.append('native_rejection_still_records_all_buffers')
    source=Path(__file__).with_name('native')/'up_swiglu.cpp'
    text=source.read_text()
    for fragment in ('kM = 128, kN = 256, kCores = 32, kFlagStride = 32',
                     'kExperts = 4, kVectorRows = 8','core_ratio(1, 2)',
                     'SetFlag<HardEvent::FIX_S>','SetFlag<HardEvent::S_MTE2>',
                     'static_assert(!Policy::ASYNC','GetBlockIdx()/GetSubBlockNum()'):
        assert fragment in text
    checks.append('native_model_constants_and_sync_markers_bound')
    from test_up_swiglu_npu import source_inventory,p0_sources,finish_identity
    inventory=source_inventory() # Actual run-mode construction, including nested native path.
    assert inventory['native/up_swiglu.cpp']==hashlib.sha256(source.read_bytes()).hexdigest() and len(inventory)==6
    checks.append('actual_run_mode_source_inventory_nested_path')
    initial={'mapped':{'libraries':[dict(path='/a.so',sha256='old')]}}
    final={'mapped':{'libraries':[dict(path='/a.so',sha256='new'),dict(path='/lazy.so',sha256='lazy')]}}
    report=dict(all_passed=True,identity_initial=initial)
    assert not finish_identity(report,lambda:final,lambda:dict(errors=[],libraries=[])) and not report['all_passed']
    assert report['identity_difference']['added']=={'/lazy.so':'lazy'}
    assert report['identity_difference']['changed']['/a.so']==dict(before='old',after='new')
    checks.append('final_changed_identity_rejected_with_difference')
    removed={'mapped':{'libraries':[]}}
    report=dict(all_passed=True,identity_initial=initial)
    assert not finish_identity(report,lambda:removed,lambda:dict(errors=[],libraries=[])) and not report['all_passed']
    assert report['identity_difference']['removed']=={'/a.so':'old'}
    checks.append('final_removed_identity_rejected')
    added={'mapped':{'libraries':[dict(path='/a.so',sha256='old'),dict(path='/lazy.so',sha256='lazy')]}}
    report=dict(all_passed=True,identity_initial=initial)
    assert finish_identity(report,lambda:added,lambda:dict(errors=[],libraries=[])) and report['all_passed']
    assert report['identity_difference']==dict(added={'/lazy.so':'lazy'},removed={},changed={})
    checks.append('final_only_lazy_added_identity_accepted')
    def fail():raise RuntimeError('identity collection injected failure')
    report=dict(all_passed=False,error='ORIGINAL NATIVE ERROR')
    assert not finish_identity(report,fail,lambda:dict(errors=[],libraries=['ACTUAL MAP EVIDENCE']))
    assert report['error']=='ORIGINAL NATIVE ERROR' and report['identity_final_maps']['libraries']==['ACTUAL MAP EVIDENCE']
    checks.append('failed_final_identity_retains_original_exception_and_maps')
    report=dict(all_passed=True)
    assert not finish_identity(report,fail,lambda:dict(errors=[],libraries=[])) and not report['all_passed']
    checks.append('success_rejected_without_final_identity')
    for mapper in (fail,lambda:dict(errors=['inode mismatch'])):
        report=dict(all_passed=True)
        assert not finish_identity(report,lambda:final,mapper) and not report['all_passed']
        checks.append('final_maps_error_rejects_success')
    names=('megamoe_training','megamoe_training.gemm');previous={n:sys.modules.get(n) for n in names}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'megamoe_training').mkdir()
            for name,relative in zip(names,('megamoe_training/__init__.py','megamoe_training/gemm.py')):
                f=root/relative;f.write_text(name)
                sys.modules[name]=types.SimpleNamespace(__file__=str(f))
            actual=p0_sources(root)
            assert all(v['sha256']==hashlib.sha256(Path(v['path']).read_bytes()).hexdigest() for v in actual.values())
            checks.append('actual_imported_p0_init_and_gemm_source_hash')
            for name in names:
                original=sys.modules[name];outside=root/'escaped.py';outside.write_text('escape')
                sys.modules[name]=types.SimpleNamespace(__file__=str(outside))
                reject(lambda:p0_sources(root));sys.modules[name]=original
                checks.append('cached_p0_module_escape_rejected:'+name)
    finally:
        for n,value in previous.items():
            if value is None:sys.modules.pop(n,None)
            else:sys.modules[n]=value
    return dict(passed=True,checks=len(checks),main=main,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        limitations=['Static partition and fake-device wrapper only; no compiler/device/stream scheduler or cross-core visibility evidence',
                      'Concurrent fixture verifies allocation ownership, not physical concurrency correctness'])

if __name__=='__main__':
    r=run();Path(__file__).with_name('cpu_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
