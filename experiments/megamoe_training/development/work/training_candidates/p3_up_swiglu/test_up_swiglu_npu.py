"""Accuracy-only preparation. Default plan imports no Torch. No timing or training qualification."""
import argparse,gc,hashlib,json,sys,traceback
from pathlib import Path
from contract import plan,DEFAULT_CAP

LIMITS=dict(atol=.003,rtol=.03,rms_relative=.02)
ROWS=(0,1,7,8,9,17,63,64,65,127,128,129)

def source_inventory():
    parent=Path(__file__).resolve().parent
    return {name:hashlib.sha256((parent/name).read_bytes()).hexdigest() for name in
        (Path(__file__).name,'contract.py','up_swiglu.py','mapped_identity.py','CMakeLists.txt','native/up_swiglu.cpp')}

def p0_sources(root):
    result={}
    for name,relative in (('megamoe_training','megamoe_training/__init__.py'),
                          ('megamoe_training.gemm','megamoe_training/gemm.py')):
        module=sys.modules.get(name)
        if module is None or not getattr(module,'__file__',None):raise ValueError('Missing actual imported module: '+name)
        actual=Path(module.__file__).resolve(strict=True)
        if actual!=(root/relative).resolve(strict=True):raise ValueError('Cached P0 import escaped explicit root: '+name)
        result[name]=dict(path=str(actual),sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
    return result

def finish_identity(report,collector,maps_collector):
    """Never mask the original exception; caller rejects success if final evidence fails."""
    try:report['identity_final_maps']=maps_collector()
    except BaseException:report['identity_final_maps_error']=traceback.format_exc()
    try:
        final=collector();report['identity_final']=final
        def libs(value):
            return {v['path']:v['sha256'] for v in value.get('mapped',{}).get('libraries',[])}
        before=libs(report.get('identity_initial',{}));after=libs(final)
        report['identity_difference']=dict(added={k:v for k,v in after.items() if k not in before},
            removed={k:v for k,v in before.items() if k not in after},
            changed={k:dict(before=before[k],after=after[k]) for k in before.keys() & after.keys() if before[k]!=after[k]})
        if report['identity_difference']['changed'] or report['identity_difference']['removed']:
            raise ValueError('Initially mapped library content changed or mapping disappeared during run')
        if report.get('identity_final_maps_error') or report['identity_final_maps'].get('errors'):
            raise ValueError('Final best-effort actual maps contains errors')
        return True
    except BaseException:
        report['identity_final_error']=traceback.format_exc();report['all_passed']=False
        return False

def stats(actual,reference):
    assert actual.shape==reference.shape and actual.dtype==reference.dtype==torch.bfloat16
    finite=True;bad=0;mx=rel=err2=ref2=0.;n=actual.numel()
    for start in range(0,n,1<<20):
        a=actual.reshape(-1)[start:start+(1<<20)].float()
        r=reference.reshape(-1)[start:start+(1<<20)].float();d=(a-r).abs()
        finite=finite and bool((torch.isfinite(a).all() & torch.isfinite(r).all()).item())
        bad+=int((d>LIMITS['atol']+LIMITS['rtol']*r.abs()).sum().item())
        mx=max(mx,float(d.max().item()));rel=max(rel,float((d/r.abs().clamp_min(1e-8)).max().item()))
        err2+=float(d.square().sum().item());ref2+=float(r.square().sum().item())
    rms=(err2/max(n,1))**.5;rrms=(ref2/max(n,1))**.5;ratio=rms/max(rrms,1e-12)
    return dict(elements=n,shape=list(actual.shape),dtype=str(actual.dtype),finite=finite,badcount=bad,
        max_abs_error=mx,max_relative_error_floor_1e_8=rel,RMS_error=rms,reference_RMS=rrms,RMS_relative_error=ratio,
        passed=finite and bad==0 and (ratio<=LIMITS['rms_relative'] if rrms else rms<=LIMITS['atol']))

def activation(gu):
    g,u=gu.float().chunk(2,-1)
    return (torch.nn.functional.silu(g)*u).bfloat16()

def check_result(result,refgu,refa):
    metrics=dict(GU=stats(result.gu,refgu),A=stats(result.a,refa))
    # Each published job has one128-byte slot; unused words remain zero.
    flags_ok=bool((result.flags[:,0]==1).all().item()) and bool((result.flags[:,1:]==0).all().item())
    return dict(metrics=metrics,flags_exact=flags_ok,passed=flags_ok and all(v['passed'] for v in metrics.values()))

def run_case(ops,gemm,counts,h,i,device,concurrent=False,pattern='random'):
    m=sum(counts);p=plan(m,h,i,counts,args.cap_bytes)
    # Conservative own inputs+immutable copies+reference/candidate saves+FP32 activation temporaries.
    peak=p['total_bytes']*3+4*m*i*4
    if peak>args.cap_bytes:raise ValueError(f'Conservative test live-byte estimate {peak} exceeds cap')
    x=torch.randn((m,h),dtype=torch.bfloat16,device=device,generator=rng)*.1
    w=torch.randn((4,2*i,h),dtype=torch.bfloat16,device=device,generator=rng)*(h**-.5)
    if pattern=='zeros':x.zero_()
    elif pattern=='sentinels':
        x.zero_();x[:,0]=1.;w.zero_()
        w[:,:8,0]=torch.tensor([-20.,-4.,-1.28,-1.,0.,1.,4.,20.],device=device,dtype=torch.bfloat16)
        w[:,i:i+8,0]=1.
    xb=x.clone();wb=w.clone()
    refgu=gemm.grouped(x,w,counts);refa=activation(refgu)
    if concurrent:
        base=torch.npu.current_stream();streams=[torch.npu.Stream(),torch.npu.Stream()];results=[];events=[]
        for s in streams:
            s.wait_stream(base)
            with torch.npu.stream(s):
                results.append(ops(x,w,counts));event=torch.npu.Event();event.record(s);events.append(event)
        # Drop native input handles; record_stream must retain storage through both launches.
        del x,w;gc.collect()
        churn=[torch.empty_like(xb).fill_(123.) for _ in range(4)]
        for event in events:base.wait_event(event)
        torch.npu.synchronize();del churn
        immutable=None # Lifetime case intentionally releases original handles.
    else:
        results=[ops(x,w,counts),ops(x,w,counts)] # repeated fresh flags, no reset/reuse
        torch.npu.synchronize();immutable=bool(torch.equal(x,xb) and torch.equal(w,wb))
    tensors=[t for r in results for t in (r.gu,r.a,r.flags) if t.numel()]
    independent=len({t.untyped_storage().data_ptr() for t in tensors})==len(tensors)
    checks=[check_result(r,refgu,refa) for r in results]
    return dict(counts=counts,M=m,H=h,I=i,pattern=pattern,concurrent_streams=concurrent,conservative_peak_bytes=peak,
        zero_rows_scope=('host_empty_no_native_or_hardware_check' if not m else 'native_launch'),
        inputs_unchanged=immutable,independent_outputs_and_flags=independent,invocations=checks,
        passed=independent and immutable is not False and all(v['passed'] for v in checks))

def identity(ops,gemm,root,torch_npu):
    from mapped_identity import mapped_libraries
    mapped=mapped_libraries()
    if mapped['errors']:raise ValueError(mapped['errors'])
    # Field names from the independent dev/inode-bound helper are retained verbatim.
    serialized=json.dumps(mapped)
    for path in (ops.path,gemm.path):
        if str(Path(path).resolve()) not in serialized:raise ValueError('Actual mapped native SO missing: '+path)
    for token in ('libascendcl.so','libruntime.so','libtorch_npu.so'):
        if token not in serialized:raise ValueError('Missing CANN/PTA actual mapping: '+token)
    internal=torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
    if isinstance(internal,bytes):internal=internal.decode()
    if internal!='disable' or torch.npu.matmul.allow_hf32:raise ValueError('Precision contract differs')
    return dict(mapped=mapped,torch=str(torch.__version__),torch_npu=str(torch_npu.__version__),
        torch_npu_git_version=getattr(getattr(torch_npu,'version',None),'git_version',None),
        precision=dict(allow_hf32=False,internal_format_raw=internal),
        hardware=dict(name=torch.npu.get_device_name(args.device),properties=str(torch.npu.get_device_properties(args.device)),
                      native_required_physical_AIC=32,native_required_physical_AIV=64),
        catlass_source_root=str(root),p0_imported_sources=p0_sources(root),candidate_backend=ops.backend)

def main():
    global args,torch,rng
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-npu',action='store_true');ap.add_argument('--library',type=Path)
    ap.add_argument('--p0-source-root',type=Path);ap.add_argument('--catlass-library',type=Path)
    ap.add_argument('--output',type=Path);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--seed',type=int,default=4317);ap.add_argument('--cap-bytes',type=int,default=DEFAULT_CAP)
    ap.add_argument('--small-only',action='store_true');args=ap.parse_args()
    specification=dict(scope='compute-only GU/A accuracy; no communication/full-training/performance qualification',
        limits=LIMITS,experts=4,AIC=32,AIV=64,rows=ROWS,native_H_I=[128,256,384,2048,6144],
        main=dict(M=65536,H=6144,I=2048,counts=[16384]*4),concurrent='one small two-stream allocator control',
        default_action='plan only; no Torch import')
    if not args.run_npu:print(json.dumps(specification));return
    if any(v is None for v in (args.library,args.catlass_library,args.p0_source_root,args.output)):
        ap.error('Explicit run requires library, catlass-library, p0-source-root and output')
    root=args.p0_source_root.resolve(strict=True)
    if not (root/'megamoe_training/gemm.py').is_file():raise ValueError('P0 source root must be experiments/megamoe_training')
    report=dict(specification=specification,argv=sys.argv,all_passed=False,cases=[],source_sha256=source_inventory())
    from mapped_identity import mapped_libraries
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    def append(item):
        report['cases'].append(item);save()
        if not item['passed']:raise AssertionError('Strict GU/A check failed; later cases skipped')
    try:
        import torch,torch_npu
        from up_swiglu import GroupedUpSwiGLU
        sys.path.insert(0,str(root))
        from megamoe_training.gemm import CatlassGemm
        report['p0_imported_sources']=p0_sources(root)
        torch.npu.set_device(args.device);torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
        device=torch.device(f'npu:{args.device}');rng=torch.Generator(device=device).manual_seed(args.seed)
        ops=GroupedUpSwiGLU(args.library,args.cap_bytes);gemm=CatlassGemm(args.catlass_library,cores=32)
        report['identity_initial']=identity(ops,gemm,root,torch_npu);save()
        with torch.no_grad():
            for n in ROWS:
                for h,i in ((128,128),(256,256),(384,384),(6144,128),(128,2048)):
                    append(run_case(ops,gemm,[n,0,1 if n else 0,n//3],h,i,device))
            for counts in ([0,0,0,0],[32]*4,[0,0,0,257],[65,1,129,0]):
                append(run_case(ops,gemm,counts,128,384,device))
            for pattern in ('zeros','sentinels'):
                append(run_case(ops,gemm,[9,0,1,65],128,128,device,pattern=pattern))
            append(run_case(ops,gemm,[17,0,65,1],128,384,device,concurrent=True))
            if not args.small_only:append(run_case(ops,gemm,[16384]*4,6144,2048,device))
        report['all_passed']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        final_ok=finish_identity(report,lambda:identity(ops,gemm,root,torch_npu),mapped_libraries)
        try:save()
        except BaseException:
            if 'error' not in report:raise
            print('Final report write also failed: '+traceback.format_exc(),file=sys.stderr)
        print(json.dumps(dict(all_passed=report['all_passed'],cases=len(report['cases']),output=str(args.output))))
        if not final_ok and 'error' not in report:raise RuntimeError('Final standalone identity incomplete; see report')

if __name__=='__main__':main()
