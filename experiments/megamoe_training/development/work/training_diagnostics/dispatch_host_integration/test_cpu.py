"""Actual patched routed methods + complete HT dispatch, with symbolic CPU stubs.

Notify device planning is not simulated: execute only its actual patched host
handle-construction region from explicit host rows/count_list inputs. No numeric,
autograd-engine, device/transport/visibility or native qualification claim.
"""
import ast,concurrent.futures,contextlib,hashlib,json,subprocess,sys,tempfile,threading,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from dispatch_host_integration import hooks
from dispatch_host_integration.recorder import DispatchHostTrace
from build_patch import build,INPUTS

class T:
    _version=0;requires_grad=False
    def __init__(self,shape,dtype='bfloat16'):
        self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.device=types.SimpleNamespace(type='npu',index=0)
    def data_ptr(self):return id(self) if all(self.shape) else 0
    def is_contiguous(self):return True
    def detach(self):return self
    def to(self,dtype):return T(self.shape,dtype)
    def float(self):return self.to('float32')
    def contiguous(self):return self
    def record_stream(self,stream):pass
    def sum(self,dim):return T(self.shape[:-1],'float32')
    def __invert__(self):return T(self.shape,self.dtype)
    def binary(self,*args):return T(self.shape,self.dtype)
    __mul__=__floordiv__=__mod__=__add__=__lt__=__eq__=binary
    def __bool__(self):raise AssertionError('Forbidden tensor scalar read')
    def cpu(self):raise AssertionError('No extra CPU tensor read')
    def item(self):raise AssertionError('No scalar read')
    def tolist(self):raise AssertionError('No tensor list read')

class Handle:
    def __init__(self,*a):
        self.is_low_latency=False;self.num_experts=a[1];self.topk_idx=a[5];self.num_expanded_tokens=a[7]
        self.num_recv_tokens_per_expert_list=a[8];self.metadata_only=False;self.metadata_version=7
        self._routing_versions={};self._dispatch_layout_cache={};self.dispatch_dst=None

class Event:
    def __init__(self,*args):pass
    def current_stream_wait(self):pass

def noop(*a,**k):return contextlib.nullcontext()
def layout(world,tokens,topk,experts,rows,fp8,weights,width):
    return dict(workspace_bytes=4096,address_stride=1<<(rows-1).bit_length(),sync=512,peer_book=1024,done=2048,
        weight_inbox=3072,weight_inbox_stride=512,aggregate_protocol=1)

def methods():
    s=(ROOT/'candidate/deep_ep/buffers/elastic.py').read_text();tree=ast.parse(s)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ElasticBuffer')
    dispatch=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch')
    # Existing imports resolved to stubs; function body is otherwise complete.
    dispatch.body=[n for n in dispatch.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    for a in (*dispatch.args.args,*dispatch.args.kwonlyargs):a.annotation=None
    dispatch.returns=None
    notify=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_notify')
    nodes=[n for n in ast.walk(notify) if isinstance(n,ast.Assign) and any(ast.unparse(t)=='result' for t in n.targets)]
    construct=next(n for n in nodes if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='EPHandle')
    # Ordered original host statements after existing count_list/rows are ready.
    container=next(n.body for n in ast.walk(notify) if isinstance(n,(ast.With,ast.Try)) and construct in getattr(n,'body',[]))
    region=container[container.index(construct):]
    stop=next(i for i,n in enumerate(region) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='result.trace')
    region=region[:stop+1]
    host=compile(ast.fix_missing_locations(ast.Module(body=region,type_ignores=[])),'patched_notify_host_region','exec')
    torch=types.SimpleNamespace(Tensor=T,bfloat16='bfloat16',float8_e4m3fn='float8',float32='float32',int64='int64',int32='int32',
        is_inference=lambda t:False,is_tensor=lambda t:isinstance(t,T),inference_mode=noop,no_grad=noop,where=lambda cond,a,b:a if isinstance(a,T) else b,
        empty=lambda shape,**k:T(shape,k.get('dtype','bfloat16')))
    scope=dict(torch=torch,EPHandle=Handle,EventHandle=Event,EventOverlap=Event,_PERSISTENT=('dst','forward_list','forward_counts'),
        native_label=hooks.native_label,retain_notify=hooks.retain_notify,runtime_of=lambda b:b._runtime,
        workspace_layout=lambda *a:None,validate_hidden=lambda *a:None,integer=lambda *a:None,dispatch_layout=layout,
        _payload_empty=lambda shape,**k:T(shape,k['dtype']),_collective_check=lambda r,e,*a,**k:(_ for _ in ()).throw(e) if e else None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[dispatch],type_ignores=[])),'actual_patched_dispatch','exec'),scope)
    p1=ast.parse((ROOT/'candidate/p1_pointwise/training_adapter.py').read_text())
    routed=next(n for n in p1.body if isinstance(n,ast.ClassDef) and n.name=='_RoutedP1')
    def once(fn):fn.once_differentiable_preserved=True;return fn
    shared=dict(torch=types.SimpleNamespace(autograd=types.SimpleNamespace(Function=object),bfloat16='bfloat16'),
        once_differentiable=once,routed_phase=hooks.routed_phase)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[routed],type_ignores=[])),'actual_patched_routed','exec'),shared)
    return scope,scope['dispatch'],host,shared['_RoutedP1']

def setup(rank=0,empty=False,enabled=True,path=None):
    scope,dispatch,hostregion,routed=methods();world=4 if empty else 2;rr=[8,8,0,0] if empty else [4,4]
    counts=[0]*4 if empty else [1]*4;local=sum(counts);b=2;k=4 if empty else 2;e=4*world
    native=types.ModuleType('native_cpu_stub');native.calls=[];native.error=None;native.result=object()
    def call(*args):
        native.calls.append(args)
        if native.error:raise native.error
        return native.result
    native.dispatch=call;native.notify_check_mapping=lambda s:None
    native.dispatch_layout=lambda rank,*a:layout(*a)
    native.megamoe_workspace_address=lambda session:8192
    session=types.SimpleNamespace(capacity=16384,peer_table=True)
    runtime=types.SimpleNamespace(native=session,rank=rank,world_size=world,device=0,capacity=16384,debug_collectives=False,
        group=None,notify_failed=False,notify_lock=threading.RLock(),stream=types.SimpleNamespace(npu_stream=17),ensure_alive=lambda:None)
    buffer=types.SimpleNamespace(_runtime=runtime,_handle_token=object(),num_max_tokens_per_rank=b,num_topk=k,hidden=128)
    def notify(self,*a):
        tensors=dict(dst=T((b,k),'int32'),forward_list=T((world,b*k,6),'int32'),forward_counts=T((world,),'int32'),_peer_plan=T((512,),'uint8'))
        env=dict(scope,self=self,runtime=runtime,num_experts=e,capacity=b,notify_cores=64,indices=a[1],
            rows=list(rr),gathered=list(rr),count_list=list(counts),counts=T((4,),'int32'),prefix=T((4,),'int32'),
            tensors=tensors,tokens=b,topk=k,hidden=128,chunk_tokens=256,rank_dedup=True)
        exec(hostregion,env);return None,None,None,env['result'],None
    buffer._notify=types.MethodType(notify,buffer);buffer.dispatch=types.MethodType(dispatch,buffer)
    scope['require_native']=lambda *a:native;scope['npu_api']=lambda:types.SimpleNamespace(stream=noop)
    class Gemm:
        def grouped(self,x,w,c,transpose_weights=True):return T((len_x(x),w.shape[1] if transpose_weights else w.shape[2]))
        def weight_grad(self,dy,x,c):return T((len(c),dy.shape[1],x.shape[1]),'float32')
    def len_x(x):return x.shape[0]
    def combine(x,handle,**kw):return T((b,128)),T((b,k),'float32') if kw else None,None
    buffer.combine=combine
    engine=types.SimpleNamespace(buffer=buffer,closed=False,num_experts=e,gemm=Gemm(),_validate=lambda *a:None,
        pointwise=types.SimpleNamespace(activation=lambda gu:T((local,128)),gate=lambda z,p:T(z.shape),activation_grad=lambda da,gu:T(gu.shape)))
    trace=None
    if enabled:
        trace=DispatchHostTrace(native,path,enabled=True,run_id='cpu-run',rank=rank,process_start_time='fixture-start');trace.__enter__()
        hooks.bind_case(engine,trace,session=session,session_identity='session-case0',case_index=0,case_name='empty' if empty else 'balanced')
    ctx=types.SimpleNamespace();ctx.save_for_backward=lambda *v:setattr(ctx,'saved_tensors',v)
    data=(T((b,128)),T((4,256,128)),T((4,128,128)),T((b,k),'int64'),T((b,k),'float32'))
    return types.SimpleNamespace(scope=scope,native=native,session=session,buffer=buffer,engine=engine,routed=routed,trace=trace,ctx=ctx,data=data)

def run(output):
    source_manifest=build();checks=[]
    def reject(fn):
        try:fn()
        except (ValueError,RuntimeError):return
        raise AssertionError('Expected rejection')
    with tempfile.TemporaryDirectory() as tmp:
        for empty in (False,True):
            f=setup(rank=2 if empty else 0,empty=empty,path=Path(tmp)/f'{empty}.jsonl')
            try:
                y=f.routed.forward(f.ctx,*f.data,f.engine);assert y.shape==(2,128)
                saved=f.ctx.handle._dispatch_host_snapshot;assert type(saved[1]) is tuple and type(saved[2]) is tuple
                assert saved[0] is f.session
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    gradients=pool.submit(f.routed.backward,f.ctx,T((2,128))).result()
                assert len(gradients)==6 and f.routed.backward.once_differentiable_preserved
                records=[json.loads(v) for v in f.trace.path.read_bytes().splitlines()];before=[v for v in records if v['event']=='before']
                assert [v['phase'] for v in before]==['forward','backward']
                assert [v['cached_handle'] for v in before]==[False,True]
                assert before[0]['metadata']['local_rows']==(0 if empty else 4)
                assert before[0]['metadata']['rows']==(8 if empty else 4)
                a,z=f.native.calls;assert a[0] is z[0] is f.session and a[2:]==z[2:]
                assert a[1]['input'] is f.data[0] and z[1]['weights'] is None
                assert hooks._PHASE.get() is None
                checks.extend([f'actual_forward_threaded_backward:{empty}',f'fresh_cached_actual_host_snapshot:{empty}',f'native_args_and_shapes:{empty}'])
            finally:f.trace.__exit__(None,None,None)
        f=setup(enabled=False);f.routed.forward(f.ctx,*f.data,f.engine);f.routed.backward(f.ctx,T((2,128)))
        assert len(f.native.calls)==2 and not hasattr(f.ctx.handle,'_dispatch_host_snapshot')
        checks.append('disabled_no_recorder_or_snapshot_and_original_calls_run')
        f=setup(path=Path(tmp)/'cache-is-not-phase.jsonl')
        try:
            f.routed.forward(f.ctx,*f.data,f.engine)
            with hooks.routed_phase(f.engine,'forward'):
                f.buffer.dispatch(T((2,128)),handle=f.ctx.handle,num_compute_units=32,num_qps=2,do_cpu_sync=False)
            with hooks.routed_phase(f.engine,'backward'):
                f.buffer.dispatch(*[f.data[0]],topk_idx=f.data[3],topk_weights=f.data[4],num_experts=8,
                    expert_alignment=1,num_compute_units=32,num_qps=2,do_cpu_sync=True)
            before=[json.loads(v) for v in f.trace.path.read_bytes().splitlines() if json.loads(v)['event']=='before']
            assert [(r['phase'],r['cached_handle']) for r in before]==[('forward',False),('forward',True),('backward',False)]
            checks.append('cached_flag_from_dispatch_entry_independent_of_explicit_phase')
        finally:f.trace.__exit__(None,None,None)
        for failure in ('missing_snapshot','stale_snapshot','stale_binding','missing_phase','native_error'):
            f=setup(path=Path(tmp)/(failure+'.jsonl'))
            try:
                f.routed.forward(f.ctx,*f.data,f.engine);n=len(f.native.calls)
                if failure=='missing_snapshot':del f.ctx.handle._dispatch_host_snapshot
                elif failure=='stale_snapshot':f.ctx.handle._dispatch_host_snapshot=(object(),(4,4),(1,1,1,1))
                elif failure=='stale_binding':f.buffer._runtime.native=object()
                elif failure=='native_error':f.native.error=RuntimeError('native-original')
                if failure=='missing_phase':reject(lambda:f.buffer.dispatch(T((2,128)),handle=f.ctx.handle,num_compute_units=32,num_qps=2,do_cpu_sync=False))
                elif failure=='native_error':
                    original=f.native.error
                    try:f.routed.backward(f.ctx,T((2,128)))
                    except RuntimeError as exc:assert exc is original
                    else:raise AssertionError('Native original exception lost')
                    checks.append('native_exception_identity_preserved_by_full_methods')
                else:reject(lambda:f.routed.backward(f.ctx,T((2,128))))
                assert len(f.native.calls)==n+int(failure=='native_error')
                assert f.trace._failed
                f.native.error=None;reject(lambda:f.routed.forward(f.ctx,*f.data,f.engine))
                assert len(f.native.calls)==n+int(failure=='native_error');checks.append('latched_stop:'+failure)
            finally:f.trace.__exit__(None,None,None)
        f=setup(enabled=False);trace=types.SimpleNamespace(_failed=False)
        reject(lambda:hooks.bind_case(f.engine,trace,session=object(),session_identity='bad',case_index=0,case_name='bad'))
        assert trace._failed;checks.append('validator_bind_old_session_failure_latches_before_native')
        calls=[];args=types.SimpleNamespace(dispatch_host_trace_dir=None)
        sentinel=object()
        assert hooks.collect_with_trace(lambda:calls.append('existing_identity') or sentinel,
            lambda:(_ for _ in ()).throw(AssertionError('disabled native loader called')),args,0) is sentinel
        assert calls==['existing_identity'];checks.append('disabled_validator_identity_result_and_no_native_install')
    # Inverse diagnostic-only wrappers: actual original math and dispatch calls unchanged.
    class Remove(ast.NodeTransformer):
        def visit_ImportFrom(self,n):return None if n.module=='dispatch_host_integration.hooks' else n
        def visit_Assign(self,n):return None if any(ast.unparse(t)=='_trace_cached_handle' for t in n.targets) else self.generic_visit(n)
        def visit_Expr(self,n):return None if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='retain_notify' else self.generic_visit(n)
        def visit_With(self,n):
            n=self.generic_visit(n)
            return n.body if any(isinstance(x.context_expr,ast.Call) and ast.unparse(x.context_expr.func) in ('native_label','routed_phase') for x in n.items) else n
    for name,path in INPUTS.items():
        original=ast.parse(path.read_bytes().decode('utf-8'));candidate=ast.parse((ROOT/'candidate'/name).read_bytes().decode('utf-8'))
        forbidden=('cpu','item','tolist','synchronize','EventHandle','EventOverlap','all_gather','all_reduce','collective_checked')
        def calls(tree):return sorted(ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call) and (ast.unparse(n.func).split('.')[-1] in forbidden))
        assert calls(original)==calls(candidate);checks.append('no_new_value_reads_events_sync_collectives:'+name)
        if not name.endswith('validate_training_p1.py'):
            assert ast.dump(original,include_attributes=False)==ast.dump(Remove().visit(candidate),include_attributes=False)
            checks.append('inverse_patch_AST_exact_original:'+name)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source_manifest[name]['original_sha256']
    with tempfile.TemporaryDirectory(dir=ROOT) as stage:
        stage=Path(stage)
        stat=stage/'stat';boot=stage/'boot'
        stat.write_text('42 (comm with ) and spaces) '+' '.join(['S']+['0']*18+['12345']+['0']*4),encoding='utf-8')
        boot.write_text('cpu-fixture-boot\n',encoding='utf-8')
        assert hooks.process_start_identity(stat,boot)=='proc_start_ticks=12345;boot_id=cpu-fixture-boot'
        checks.append('actual_proc_field22_parser_no_context_or_device_access')
        for name,path in INPUTS.items():
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        for flags in (['--check'],[]):
            completed=subprocess.run(['git','-c','core.autocrlf=false','-c','core.eol=lf','apply',*flags,str(ROOT/'integration.patch')],cwd=stage,capture_output=True,text=True)
            if completed.returncode:raise AssertionError(completed.stderr)
        assert all((stage/name).read_bytes()==(ROOT/'candidate'/name).read_bytes() for name in INPUTS)
    checks.append('unified_patch_check_apply_on_temporary_original_copies_exact_candidates')
    assert 'torch' not in sys.modules
    result=dict(passed=True,checks=len(checks),details=checks,source_manifest=source_manifest,
        scope='Offline symbolic CPU stubs only; actual patched full routed forward/backward and full HT dispatch executed; Notify host-handle region only, no Notify device plan',
        limitations=['No numerical/autograd-engine/NPU/SHMEM/transport/visibility proof','Diagnostic integration not deployed or authorized for devices',
            'Existing Notify tensor-to-host summary and existing events/synchronization remain in original positions; no additional such operations',
            'Validator full distributed execution not performed; its unchanged collective call sites checked statically'])
    output.write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'));return result

if __name__=='__main__':print(json.dumps(run(ROOT.parents[2]/'outputs/dispatch_host_integration_cpu_review.json')))
