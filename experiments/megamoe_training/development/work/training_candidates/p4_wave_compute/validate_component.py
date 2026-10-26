"""P4 accuracy-only small GU/A/Z smoke. Default plan imports no Torch.

No communication, PG, full-training, timing or automatic target successor.
Run requires separate exact-SO/device/own-PID bounded-controller authorization.
"""
import argparse,hashlib,json,math,os,sys,traceback,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from p4_wave_compute.contract import plan,DEFAULT_CAP
from p4_wave_compute.component_identity import collect,compare,observe_final,inventory
LIMITS=dict(atol=.003,rtol=.03,rms_relative=.02)
CASES=[
 dict(name='empty',counts=[0,0,0,0],H=128,I=128),
 dict(name='K128to384_I_and_H_Ntails_AIVrows128plus1',counts=[129,0,0,0],H=128,I=384),
 dict(name='K384to128_H_Ntail_empty_expert',counts=[0,1,128,17],H=384,I=128),
 dict(name='wave511_512_513',counts=[511,512,513,0],H=384,I=384),
 dict(name='skew1025_threewaves',counts=[0,0,0,1025],H=256,I=128),
 dict(name='balanced_fullN256',counts=[128]*4,H=256,I=256),
 dict(name='signed_activation_sentinels',counts=[129,0,17,1],H=128,I=384,pattern='sentinels'),
 dict(name='independent_two_stream_invocations',counts=[129,0,1,17],H=384,I=128,concurrent=True),
]
def summary(n,bad,finite,mx,rel,err2,ref2,shape,dtype):
    if not finite or not all(math.isfinite(v) and v>=0 for v in (mx,rel,err2,ref2)):
        return dict(shape=shape,dtype=dtype,elements=n,finite=False,badcount=bad,passed=False,limits=LIMITS)
    rms=(err2/max(n,1))**.5;reference=(ref2/max(n,1))**.5;ratio=rms/max(reference,1e-12)
    return dict(shape=shape,dtype=dtype,elements=n,finite=True,badcount=bad,max_abs_error=mx,
        max_relative_error_floor_1e_8=rel,RMS_error=rms,reference_RMS=reference,RMS_relative_error=ratio,limits=LIMITS,
        passed=bad==0 and (ratio<=LIMITS['rms_relative'] if reference else rms<=LIMITS['atol']))
def statistics(torch,actual,reference):
    if actual.shape!=reference.shape or actual.dtype!=reference.dtype or actual.dtype!=torch.bfloat16:raise ValueError('GU/A/Z shape or BF16 dtype mismatch')
    finite=True;bad=0;mx=rel=err2=ref2=0.;n=actual.numel()
    for start in range(0,n,1<<20):
        a=actual.reshape(-1)[start:start+(1<<20)].float();r=reference.reshape(-1)[start:start+(1<<20)].float();d=(a-r).abs()
        finite=finite and bool((torch.isfinite(a).all() & torch.isfinite(r).all()).item())
        bad+=int((d>LIMITS['atol']+LIMITS['rtol']*r.abs()).sum().item())
        mx=max(mx,float(d.max().item()));rel=max(rel,float((d/r.abs().clamp_min(1e-8)).max().item()))
        err2+=float(d.square().sum().item());ref2+=float(r.square().sum().item())
    return summary(n,bad,finite,mx,rel,err2,ref2,list(actual.shape),str(actual.dtype))
def staged_reference(torch,x,w1,w2,counts):
    """Independent per-expert FP32 mm operands, explicit BF16 storage boundaries."""
    m,h=x.shape;i=w2.shape[2]
    gu=torch.empty((m,2*i),device=x.device,dtype=torch.bfloat16)
    a=torch.empty((m,i),device=x.device,dtype=torch.bfloat16)
    z=torch.empty((m,h),device=x.device,dtype=torch.bfloat16);begin=0
    for expert,count in enumerate(counts):
        end=begin+count
        if count:
            q=torch.mm(x[begin:end].float(),w1[expert].float().T).bfloat16();gu[begin:end]=q
            g,u=q.float().chunk(2,-1);v=(torch.nn.functional.silu(g)*u).bfloat16();a[begin:end]=v
            z[begin:end]=torch.mm(v.float(),w2[expert].float().T).bfloat16()
        begin=end
    if begin!=m:raise ValueError('Reference counts must exactly cover X')
    return gu,a,z
def metadata(t):return dict(shape=list(t.shape),dtype=str(t.dtype),stride=list(t.stride()),device=str(t.device),data_ptr=t.data_ptr())
def run_case(torch,ops,spec,index,args,collector,item,save):
    counts=spec['counts'];m=sum(counts);h,i=spec['H'],spec['I'];seed=args.seed+101*index
    p=plan(m,h,i,counts,args.cap_bytes);reservation=3*p['total_bytes']+32*m*(h+i)
    if reservation>args.cap_bytes:raise ValueError('Conservative test tensor reservation exceeds explicit cap')
    item.update(spec,seed=seed,M=m,contract=p,conservative_tensor_reservation_bytes=reservation,passed=False,invocations=[])
    initial=None;prior=None
    try:
        item['phase']='identity_initial';save()
        initial=collector();item['identity_initial']=initial;save()
        gen=torch.Generator(device=args.device).manual_seed(seed)
        x=torch.randn((m,h),device=args.device,generator=gen,dtype=torch.float32).mul_(.1).bfloat16()
        w1=torch.randn((4,2*i,h),device=args.device,generator=gen,dtype=torch.float32).mul_(h**-.5).bfloat16()
        w2=torch.randn((4,h,i),device=args.device,generator=gen,dtype=torch.float32).mul_(i**-.5).bfloat16()
        if spec.get('pattern')=='sentinels':
            x.zero_();x[:,0]=1.;w1.zero_()
            w1[:,:8,0]=torch.tensor([-20.,-4.,-1.28,-1.,0.,1.,4.,20.],device=args.device,dtype=torch.bfloat16)
            w1[:,i:i+8,0]=1.
        immutable=[t.clone() for t in (x,w1,w2)];item['phase']='independent_reference';save()
        ref=staged_reference(torch,x,w1,w2,counts)
        item['inputs']=[metadata(t) for t in (x,w1,w2)]
        base=torch.npu.current_stream();results=[];item['streams']=[]
        if spec.get('concurrent'):
            streams=[torch.npu.Stream(),torch.npu.Stream()];events=[]
            for stream in streams:
                stream.wait_stream(base)
                with torch.npu.stream(stream):
                    item['streams'].append(int(stream.npu_stream));item['phase']='native_submit/'+str(len(results));save()
                    results.append(ops(x,w1,w2,counts));item['phase']='native_return/'+str(len(results)-1);save()
                    event=torch.npu.Event();event.record(stream);events.append(event)
            for event in events:base.wait_event(event)
        else:
            for invocation in range(2):
                item['streams'].append(int(base.npu_stream));item['phase']='native_submit/'+str(invocation);save()
                results.append(ops(x,w1,w2,counts));item['phase']='native_return/'+str(invocation);save()
        item['phase']='device_synchronize';save();torch.npu.synchronize();item['phase']='metrics_after_sync';save()
        all_tensors=[t for out in results for t in (out.gu,out.a,out.z,out.flags) if t.numel()]
        independent=len({t.untyped_storage().data_ptr() for t in all_tensors})==len(all_tensors)
        streams_distinct=not spec.get('concurrent') or len(set(item['streams']))==2
        unchanged=all(torch.equal(t,old) for t,old in zip((x,w1,w2),immutable))
        for out in results:
            metrics={name:statistics(torch,t,r) for name,t,r in zip(('GU','A','Z'),(out.gu,out.a,out.z),ref)}
            flag_ok=(list(out.flags.shape)==[2*p['jobs'],32] and out.flags.dtype==torch.int32 and
                bool((out.flags[:,0]==1).all().item()) and bool((out.flags[:,1:]==0).all().item()))
            item['invocations'].append(dict(metrics=metrics,flags_exact=flag_ok,outputs=[metadata(t) for t in (out.gu,out.a,out.z,out.flags)],
                passed=flag_ok and all(v['passed'] for v in metrics.values())))
        item.update(independent_outputs_and_flags=independent,inputs_unchanged=unchanged,streams_distinct=streams_distinct,
                    zero_rows_scope='host_empty_no_native_or_hardware_check' if not m else 'native_launch')
        item['passed']=independent and unchanged and streams_distinct and all(v['passed'] for v in item['invocations'])
        if not item['passed']:raise AssertionError('Fixed all-element GU/A/Z/privateflag/immutability gate failed')
    except BaseException as exc:
        prior=exc;item.update(error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        try:torch.npu.synchronize()
        except BaseException as exc:
            item.update(passed=False,synchronize_error=repr(exc),synchronize_traceback=traceback.format_exc());prior=prior or exc
        final_error=observe_final(item,collector,initial,prior)
        if final_error is not None and sys.exc_info()[1] is None:raise final_error
def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-npu',action='store_true')
    p.add_argument('--library',type=Path);p.add_argument('--expected-so-sha256');p.add_argument('--output',type=Path)
    p.add_argument('--device',type=int,default=0);p.add_argument('--seed',type=int,default=4317);p.add_argument('--cap-bytes',type=int,default=DEFAULT_CAP)
    a=p.parse_args()
    if a.device<0 or not 0<=a.seed<1<<63 or not 0<a.cap_bytes<1<<63:p.error('Invalid device/seed/cap')
    if a.run_npu and (a.library is None or a.output is None or not isinstance(a.expected_so_sha256,str) or len(a.expected_so_sha256)!=64 or any(c not in '0123456789abcdef' for c in a.expected_so_sha256)):p.error('Run requires exact library, SHA256 and new output')
    return a
def main():
    args=parse();config=dict(candidate='P4_compute_wave512_GU_A_Z_v1_UNQUALIFIED',limits=LIMITS,cases=CASES,seed=args.seed,cap_bytes=args.cap_bytes,
        mode='accuracy_only_single_worker',native_required_AIC=32,native_required_AIV=64,shared=False,recompute=False)
    if not args.run_npu:
        print(json.dumps(dict(action='plan only, no Torch import',configuration=config,scope='small component only; no target/training/performance successor')));return
    report=dict(configuration=config,argv=sys.argv,all_passed=False,cases=[],source_sha256=inventory(),
        worker=dict(pid=os.getpid(),cwd=os.getcwd(),executable=sys.executable,start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        environment={k:os.getenv(k) for k in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_HOME_PATH','LD_LIBRARY_PATH')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('Refuse reused report/output')
    def save():
        report['saved_wall_ns']=time.time_ns();report['saved_monotonic_ns']=time.monotonic_ns()
        args.output.write_bytes((json.dumps(report,indent=2,allow_nan=False)+'\n').encode('utf8'))
    ops=None;initial=None;prior=None;collector=None
    try:
        save();report['phase']='torch_import';save()
        import torch,torch_npu
        from p4_wave_compute import WaveCompute
        report['phase']='set_device';save();torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
        hardware=dict(device=args.device,name=torch.npu.get_device_name(args.device),properties=str(torch.npu.get_device_properties(args.device)))
        args.device=torch.device('npu:'+str(args.device));ops=WaveCompute(args.library,args.cap_bytes)
        collector=lambda:collect(ops,torch,torch_npu,config,hardware,args.expected_so_sha256)
        initial=collector();report['identity_initial']=initial;save()
        with torch.no_grad():
            for index,spec in enumerate(CASES):
                item={};report['cases'].append(item);report['phase']='case/'+spec['name'];save()
                try:run_case(torch,ops,spec,index,args,collector,item,save)
                finally:
                    first=sys.exc_info()[1]
                    try:save()
                    except BaseException as exc:
                        item['report_write_error']=repr(exc)
                        if first is None:raise
                        print('Case report write also failed: '+traceback.format_exc(),file=sys.stderr)
        report['all_passed']=True
    except BaseException as exc:
        prior=exc;report.update(all_passed=False,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        try:
            from p4_wave_compute.mapped_identity import mapped_libraries
            report['final_best_effort_maps']=mapped_libraries()
        except BaseException as exc:report['final_maps_error']=repr(exc)
        final_error=observe_final(report,collector,initial,prior) if collector is not None else prior or RuntimeError('No initialized collector')
        if final_error is not None or report.get('final_maps_error') or report.get('final_best_effort_maps',{}).get('errors'):report['all_passed']=False
        try:save()
        except BaseException:
            if prior is None:raise
            print('Final report write also failed: '+traceback.format_exc(),file=sys.stderr)
        print(json.dumps(dict(all_passed=report['all_passed'],cases=len(report['cases']),output=str(args.output))))
        if prior is None and not report['all_passed']:raise final_error or RuntimeError('Final P4 identity incomplete')
if __name__=='__main__':main()
