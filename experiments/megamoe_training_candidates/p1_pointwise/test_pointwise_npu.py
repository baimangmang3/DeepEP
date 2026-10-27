"""Standalone candidate correctness only. Default prints plan, no Torch import.

Explicit device run: python test_pointwise_npu.py --run-npu --library
/ABS/build/libmegamoe_pointwise.so --output /ABS/NEW_RUN/results.json
Environment agent alone builds/launches on reserved devices. No fallback/benchmark.
"""
import argparse,gc,hashlib,json
from pathlib import Path

LIMIT={'atol':.003,'rtol':.03,'rms_relative':.02}
ROWS=(0,1,7,8,9,17)
WIDTHS=(128,256,384,2048,6144)


def metrics(actual,reference):
    if actual.shape!=reference.shape or actual.dtype!=torch.bfloat16 or reference.dtype!=torch.bfloat16:
        raise AssertionError('Output shape/dtype mismatch')
    n=actual.numel();bad=0;finite=True;maximum=0.;relative=0.;error2=0.;ref2=0.
    for start in range(0,n,1<<20):
        a=actual.reshape(-1)[start:start+(1<<20)].float()
        r=reference.reshape(-1)[start:start+(1<<20)].float();d=(a-r).abs()
        finite=finite and bool((torch.isfinite(a).all() & torch.isfinite(r).all()).item())
        bad+=int((d>LIMIT['atol']+LIMIT['rtol']*r.abs()).sum().item())
        maximum=max(maximum,float(d.max().item()))
        relative=max(relative,float((d/r.abs().clamp_min(1e-8)).max().item()))
        error2+=float(d.square().sum().item());ref2+=float(r.square().sum().item())
    rms=(error2/max(n,1))**.5;reference_rms=(ref2/max(n,1))**.5
    rms_relative=rms/max(reference_rms,1e-12)
    good=finite and bad==0 and (rms_relative<=LIMIT['rms_relative'] if reference_rms else rms<=LIMIT['atol'])
    return dict(shape=list(actual.shape),dtype=str(actual.dtype),elements=n,finite=finite,
        badcount=bad,max_abs_error=maximum,max_relative_error_floor_1e_8=relative,RMS_error=rms,
        reference_RMS=reference_rms,RMS_relative_error=rms_relative,limits=LIMIT,passed=good)


def check_unchanged(tensor,before):
    if not torch.equal(tensor,before):raise AssertionError('Candidate modified an input')


def activation_inputs(m,width,device):
    g=torch.randn((m,width),device=device,dtype=torch.bfloat16,generator=rng)*2.
    sentinels=torch.tensor([-20.,-4.,-1.,0.,1.,4.,20.,-1.28],device=device,dtype=torch.bfloat16)
    if m:g[:,:len(sentinels)]=sentinels
    u=torch.randn((m,width),device=device,dtype=torch.bfloat16,generator=rng)*.5
    gu=torch.cat((g,u),dim=-1).contiguous()
    da=(torch.randn((m,width),device=device,dtype=torch.bfloat16,generator=rng)*.2).contiguous()
    return gu,da


def activation_reference(gu):
    g,u=gu.float().chunk(2,dim=-1)
    return (torch.nn.functional.silu(g)*u).bfloat16()


def grad_reference(da,gu):
    g,u=gu.float().chunk(2,dim=-1);s=torch.sigmoid(g);d=da.float()
    return torch.cat((d*u*s*(1+g*(1-s)),d*g*s),dim=-1).bfloat16()


def test_activation(ops,m,width,device):
    gu,da=activation_inputs(m,width,device);gu_before=gu.clone();da_before=da.clone()
    expected=activation_reference(gu);actual=ops.activation(gu)
    torch.npu.synchronize();forward=metrics(actual,expected)
    del actual,expected
    expected=grad_reference(da,gu);actual=ops.activation_grad(da,gu)
    torch.npu.synchronize();backward=metrics(actual,expected)
    check_unchanged(gu,gu_before);check_unchanged(da,da_before)
    return dict(operation='activation_and_grad',M=m,I=width,forward=forward,backward=backward,
                inputs_unchanged=True,passed=forward['passed'] and backward['passed'])


def test_gate(ops,m,width,device):
    z=torch.randn((m,width),device=device,dtype=torch.bfloat16,generator=rng)
    values=torch.tensor([0.,.25,-.5,1.5,-2.,0.,.125],device=device,dtype=torch.float32)
    p=values[torch.arange(m,device=device)%len(values)].contiguous()
    z_before=z.clone();p_before=p.clone()
    expected=(z.float()*p[:,None]).bfloat16();actual=ops.gate(z,p)
    torch.npu.synchronize();result=metrics(actual,expected)
    check_unchanged(z,z_before);check_unchanged(p,p_before)
    if m and not torch.equal(actual[p==0],torch.zeros_like(actual[p==0])):
        raise AssertionError('Zero gates did not produce exact zero')
    return dict(operation='row_gate',M=m,H=width,metrics=result,inputs_unchanged=True,
                unweighted_Z_retained=True,passed=result['passed'])


def lifetime_pressure(ops,device):
    # One small independent stream/control comparison. No claimed stress coverage.
    gu,_=activation_inputs(17,384,device);expected=activation_reference(gu)
    default=torch.npu.current_stream();stream=torch.npu.Stream()
    stream.wait_stream(default)
    with torch.npu.stream(stream):
        actual=ops.activation(gu)
        done=torch.npu.Event();done.record(stream)
    del gu;gc.collect()
    churn=[torch.empty((17,768),device=device,dtype=torch.bfloat16).fill_(123.) for _ in range(8)]
    default.wait_event(done);torch.npu.synchronize()
    result=metrics(actual,expected);del churn
    return dict(operation='one_small_stream_allocator_lifetime_control',metrics=result,passed=result['passed'])


def runtime_identity(ops,args,torch_npu):
    """Actual mappings, not guessed SDK paths. This is not training qualification."""
    selected=Path(ops.path).resolve(strict=True);mapped=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)<6 or not fields[5].startswith('/'):continue
        raw=fields[5].replace('\\040',' ')
        if '.so' not in raw:continue
        name=Path(raw.removesuffix(' (deleted)')).name
        relevant=name.startswith(('libascendcl.so','libruntime.so','libopapi','libtorch_npu.so')) or str(selected)==raw
        if relevant:
            if raw.endswith(' (deleted)'):raise RuntimeError('Tracked loaded library was deleted: '+raw)
            mapped.add(Path(raw).resolve(strict=True))
    if selected not in mapped:raise RuntimeError('Selected pointwise SO is not actually mapped')
    for prefix in ('libascendcl.so','libruntime.so','libtorch_npu.so'):
        if not any(path.name.startswith(prefix) for path in mapped):
            raise RuntimeError('Missing actual CANN/PTA mapping: '+prefix)
    internal=torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
    if isinstance(internal,bytes):internal=internal.decode()
    if internal!='disable' or torch.npu.matmul.allow_hf32:
        raise RuntimeError('Actual native precision settings differ from the fixed test contract')
    return dict(pointwise_mapped_path=str(selected),libraries=[dict(path=str(path),bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(mapped,key=str)],
        torch=str(torch.__version__),torch_npu=str(torch_npu.__version__),
        torch_npu_git_version=getattr(getattr(torch_npu,'version',None),'git_version',None),
        precision=dict(allow_hf32=torch.npu.matmul.allow_hf32,
                       allow_internal_format=(internal=='enable')),
        device=dict(index=args.device,name=torch.npu.get_device_name(args.device),
                    properties_repr=str(torch.npu.get_device_properties(args.device))))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-npu',action='store_true');ap.add_argument('--library',type=Path)
    ap.add_argument('--output',type=Path);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--vector-cores',type=int,default=64);ap.add_argument('--seed',type=int,default=4317)
    ap.add_argument('--small-only',action='store_true',help='Diagnostic subset; not complete main-shape qualification')
    args=ap.parse_args()
    plan=dict(rows=list(ROWS),widths=list(WIDTHS),main_activation=[65536,2048],main_gate=[65536,6144],
              limits=LIMIT,scope='pointwise only; no communication/GEMM fusion',default_action='plan only, no Torch/NPU import')
    if not args.run_npu:
        print(json.dumps(plan));return
    if args.library is None or args.output is None:raise ValueError('--run-npu requires --library and --output')
    global torch,rng
    import torch,torch_npu
    from pointwise import TrainingPointwise
    torch.npu.set_device(args.device)
    torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
    rng=torch.Generator(device=f'npu:{args.device}').manual_seed(args.seed)
    ops=TrainingPointwise(args.library,vector_cores=args.vector_cores)
    report=dict(plan,seed=args.seed,vector_cores=args.vector_cores,torch=str(torch.__version__),
                torch_npu=str(torch_npu.__version__),library=str(args.library.resolve()),
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__),Path(__file__).with_name('pointwise.py'),Path(__file__).with_name('CMakeLists.txt'),Path(__file__).parent/'native/pointwise.cpp')},
                qualification_scope='pointwise accuracy only, not full training qualification; M0 uses wrapper short circuit',
                small_only=args.small_only,main_cases_executed=False,checks=[],passed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def record(check):
        report['checks'].append(check)
        if not check['passed']:
            raise AssertionError('Pointwise fixed gate failed; stopping before later cases: '+check['operation'])
    try:
        report['loaded_before_tests']=runtime_identity(ops,args,torch_npu)
        with torch.no_grad():
            for m in ROWS:
                for width in WIDTHS:
                    record(test_activation(ops,m,width,f'npu:{args.device}'))
                    record(test_gate(ops,m,width,f'npu:{args.device}'))
                    gc.collect();torch.npu.empty_cache()
            record(lifetime_pressure(ops,f'npu:{args.device}'))
            if not args.small_only:
                record(test_activation(ops,65536,2048,f'npu:{args.device}'))
                gc.collect();torch.npu.empty_cache()
                record(test_gate(ops,65536,6144,f'npu:{args.device}'))
                report['main_cases_executed']=True
            report['passed']=all(item['passed'] for item in report['checks'])
        if not report['passed']:raise AssertionError('Fixed pointwise numerical gate failed; do not loosen limits')
    except Exception as exc:
        report['error']=repr(exc);raise
    finally:
        try:report['loaded_after_tests']=runtime_identity(ops,args,torch_npu)
        except Exception as exc:report['identity_error']=repr(exc);report['passed']=False
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(passed=report['passed'],checks=len(report['checks']),main_cases_executed=report['main_cases_executed'],output=str(args.output.resolve()),scope=report['qualification_scope'])))
    if not report['passed']:raise RuntimeError('Pointwise report failed; see saved evidence')


if __name__=='__main__':main()
