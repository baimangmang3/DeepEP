"""No Torch imports. Native/tensor stubs check observation-only host behavior."""
import importlib.util,json,sys,tempfile,types
from pathlib import Path
from recorder import DispatchHostTrace

class Tensor:
    def __init__(self,shape,dtype='torch.bfloat16',ptr=100):self.shape=shape;self.dtype=dtype;self.ptr=ptr;self.pointer_reads=0
    def data_ptr(self):self.pointer_reads+=1;return self.ptr
    def __getattr__(self,name):
        if name in ('cpu','item','synchronize','Event','numpy','tolist'):
            raise AssertionError('Forbidden device/value method '+name)
        raise AttributeError(name)

def fixture():
    n=types.ModuleType('stub_native');n.calls=[];n.layouts=[];n.result=object();n.error=None
    def dispatch(*args,**kwargs):
        n.calls.append((args,kwargs))
        if n.error is not None:raise n.error
        return n.result
    def layout(*args):
        n.layouts.append(args)
        return dict(workspace_bytes=4096,address_stride=4,sync=512,peer_book=1024,done=2048,weight_inbox=3072,weight_inbox_stride=512)
    n.dispatch=dispatch;n.dispatch_layout=layout;n.megamoe_workspace_address=lambda session:8192
    s=types.SimpleNamespace(capacity=16384,peer_table=True)
    tensors=dict(input=Tensor((2,128)),output=Tensor((4,128)),dst=Tensor((2,2),'torch.int64'),
        forward_list=Tensor((2,4,6),'torch.int32'),forward_counts=Tensor((2,),'torch.int32'),
        weights=Tensor((2,2),'torch.float32'),output_weights=Tensor((4,),'torch.float32'),scales=None,output_scales=None,
        plan=Tensor((512,),'torch.uint8'))
    args=(s,tensors,2,2,8,4,4,False,17)
    labels=dict(session=s,session_identity='session0',case_index=0,case_name='skew',phase='forward',world=2,
        rank_rows=[4,4],counts=[1,1,1,1],cached_handle=False,rank_rows_source='existing Notify host summary')
    return n,s,tensors,args,labels

def run(output):
    checks=[];counter=0
    def record(name):checks.append(name)
    def reject(action,kind=(ValueError,RuntimeError,OSError,AttributeError)):
        try:action()
        except kind:return
        raise AssertionError('Expected failure')
    with tempfile.TemporaryDirectory() as tmp:
        def logger(n,**kw):
            nonlocal counter
            counter+=1
            return DispatchHostTrace(n,Path(tmp)/f'{counter}.jsonl',enabled=True,run_id='run0',rank=0,process_start_time='proc_start_ticks=123',**kw)
        n,s,t,args,labels=fixture();original=n.dispatch
        with logger(n) as log:
            with log.label(**labels):assert n.dispatch(*args) is n.result
            labels.update(phase='backward',cached_handle=True)
            with log.label(**labels):assert n.dispatch(*args[:8],stream_address=args[8]) is n.result
        assert n.dispatch is original and n.calls[0][0]==args
        assert n.calls[1][0]==args[:8] and n.calls[1][1]==dict(stream_address=17)
        lines=[json.loads(v) for v in log.path.read_bytes().decode('utf-8',errors='strict').splitlines()]
        assert [r['event'] for r in lines]==['before','native_return']*2
        assert [r['call_sequence'] for r in lines]==[1,1,2,2]
        assert lines[0]['phase']=='forward' and lines[2]['phase']=='backward'
        assert all(r.get('native_return_is_device_completion') is False for r in lines if r['event']!='before')
        assert lines[0]['metadata']['native_layout']['peer_book']==1024
        assert lines[0]['metadata']['workspace']['address']==8192 and n.layouts[0]==(0,2,2,2,8,4,False,True,128)
        record('args_kwargs_result_identity_and_phases_sequences');record('actual_layout_capacity_optional_workspace_API_metadata')
        record('native_return_explicitly_not_device_completion');record('exit_restores_original')
        assert all(v.pointer_reads==2 for v in t.values() if v is not None);record('metadata_only_no_device_value_methods')
        reject(lambda:log.__enter__());record('same_recorder_once_only')
        reject(lambda:logger(n).__enter__());record('same_native_once_only')
        n,s,t,args,labels=fixture();error=RuntimeError('native failure');n.error=error
        with logger(n) as log:
            with log.label(**labels):
                try:n.dispatch(*args)
                except RuntimeError as caught:assert caught is error
                else:raise AssertionError('Native exception swallowed')
                assert log._failed and len(n.calls)==1
                reject(lambda:n.dispatch(*args));assert len(n.calls)==1
                record('native_error_latches_failed_blocks_existing_label_call')
            reject(lambda:log.label(**labels).__enter__());assert len(n.calls)==1
            record('native_error_latches_failed_blocks_new_label')
        events=[json.loads(v) for v in log.path.read_bytes().splitlines()]
        assert events[-1]['event']=='native_error' and events[-1]['exception_message']=='native failure';record('native_exception_original_identity_and_error_record')
        n,s,t,args,labels=fixture();del n.megamoe_workspace_address
        with logger(n) as log:
            with log.label(**labels):n.dispatch(*args)
        first=json.loads(log.path.read_bytes().splitlines()[0]);assert first['metadata']['workspace']['available'] is False
        record('absent_workspace_API_explicit_gap_no_guess')
        n,s,t,args,labels=fixture();s.peer_table=False;t['plan']=None
        with logger(n) as log:
            with log.label(**labels):assert n.dispatch(*args) is n.result
        first=json.loads(log.path.read_bytes().splitlines()[0]);assert first['metadata']['session_peer_table'] is False
        assert first['metadata']['tensors']['plan'] is None and n.calls[0][0] is not None and t['plan'] is None
        record('nonpeer_table_None_plan_passes_through')
        n,s,t,args,labels=fixture();t['plan']=None
        with logger(n) as log:
            with log.label(**labels):reject(lambda:n.dispatch(*args))
            assert not n.calls
        record('peer_table_None_plan_blocks_native')
        n,s,t,args,labels=fixture()
        t.update(dedup_group_size=1,kernel_timing=True,benchmark_barriers=False,rotation_wqes=4,detail_mode=2,
                 profile=Tensor((64,32),'torch.int64'),detail_profile=Tensor((64,32),'torch.int64'),weight_profile=None)
        original_items=dict(t)
        with logger(n) as log:
            with log.label(**labels):n.dispatch(*args)
        meta=json.loads(log.path.read_bytes().splitlines()[0])['metadata']
        assert meta['incoming_controls']['dedup_group_size']==dict(provided=True,value=1)
        assert meta['incoming_controls']['kernel_timing']==dict(provided=True,value=True)
        assert meta['incoming_controls']['benchmark_barriers']==dict(provided=True,value=False)
        assert meta['incoming_controls']['detail_mode']['value']==2
        assert meta['incoming_profile_buffers']['detail_profile']['tensor']['data_ptr']==t['detail_profile'].ptr
        assert t==original_items and n.calls[0][0][1] is t
        record('incoming_dedup_diagnostics_profile_snapshot_never_enabled_or_mutated')
        for kind in ('no_label','missing_tensor','missing_layout','wrong_rows','wrong_session','bad_workspace','bad_capacity','missing_peer_table'):
            n,s,t,args,labels=fixture();before=n.dispatch
            if kind=='missing_tensor':del t['plan']
            if kind=='missing_layout':n.dispatch_layout=lambda *a:{}
            if kind=='wrong_rows':labels['rank_rows']=[3,4]
            if kind=='wrong_session':labels['session']=object()
            if kind=='bad_workspace':n.megamoe_workspace_address=lambda s:0
            if kind=='bad_capacity':s.capacity=1024
            if kind=='missing_peer_table':del s.peer_table
            with logger(n) as log:
                if kind=='no_label':reject(lambda:n.dispatch(*args))
                else:
                    with log.label(**labels):reject(lambda:n.dispatch(*args))
                assert not n.calls and log._failed
            assert n.dispatch is before;record('precall_blocks_native:'+kind)
        for kind in ('rank_rows_tensor','phase','counts_tensor'):
            n,s,t,args,labels=fixture()
            if kind=='rank_rows_tensor':labels['rank_rows']=Tensor((2,))
            elif kind=='counts_tensor':labels['counts']=Tensor((4,))
            else:labels['phase']='guess-from-stack'
            with logger(n) as log:reject(lambda:log.label(**labels).__enter__())
            assert not n.calls;record('explicit_host_labels_required:'+kind)
        for kind in ('record_budget','json_bytes','write_failure'):
            n,s,t,args,labels=fixture();kw={'max_records':2} if kind=='record_budget' else {}
            if kind=='json_bytes':t['extra_ignored']=None;labels['rank_rows_source']='x'*128;kw['max_record_bytes']=1024
            with logger(n,**kw) as log:
                if kind=='write_failure':
                    def fail_write(raw):raise OSError('before disk failure')
                    log._write=fail_write
                if kind=='record_budget':
                    with log.label(**labels):n.dispatch(*args)
                    assert len(n.calls)==1
                    with log.label(**labels):reject(lambda:n.dispatch(*args))
                    assert len(n.calls)==1
                elif kind=='write_failure':
                    with log.label(**labels):reject(lambda:n.dispatch(*args))
                    assert not n.calls and log._failed
                else:
                    with log.label(**labels):reject(lambda:n.dispatch(*args))
                    assert not n.calls
            record('fail_closed:'+kind)
        n,s,t,args,labels=fixture()
        with logger(n,max_records=100,max_total_bytes=65536) as log:
            for attempt in range(100):
                before=len(n.calls)
                try:
                    with log.label(**labels):n.dispatch(*args)
                except ValueError:assert len(n.calls)==before;break
            else:raise AssertionError('Total-byte limit did not close')
            assert log._failed and log._bytes<=65536
        record('total_file_byte_budget_blocks_next_native')
        n,s,t,args,labels=fixture();external=lambda *a,**k:None
        log=logger(n);log.__enter__();n.dispatch=external
        reject(lambda:log.__exit__(None,None,None));assert n.dispatch is external
        record('external_new_hook_not_overwritten')
        n,s,t,args,labels=fixture();error=RuntimeError('original native');n.error=error
        with logger(n) as log:
            original_write=log._write
            def fail_terminal(raw):
                if b'native_error' in raw:raise OSError('terminal disk error')
                original_write(raw)
            log._write=fail_terminal
            try:
                with log.label(**labels):n.dispatch(*args)
            except RuntimeError as caught:assert caught is error and log.last_logging_error=='terminal disk error'
            else:raise AssertionError('Original exception lost')
        record('terminal_log_failure_preserves_original_native_exception')
        n,s,t,args,labels=fixture();log=DispatchHostTrace(n,Path(tmp)/'off',run_id='r',rank=0,process_start_time='123')
        reject(lambda:log.__enter__());assert not n.calls;record('explicit_opt_in_required')
    assert 'torch' not in sys.modules;record('no_Torch_import')
    result=dict(passed=True,checks=len(checks),details=checks,scope='CPU stubs only; not integrated, deployed or authorized for NPU',
        limitations=['Caller must supply actual existing host Notify rank_rows/counts; logger cannot retrieve device values',
                    'Native return is not device completion; no visibility/transport/root-cause proof',
                    'Existing workspace accessor optional and context-checked; no fallback or guessed addresses'])
    output.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));return result

if __name__=='__main__':
    # Script lives under work/training_diagnostics/dispatch_host_trace.
    output=Path(__file__).resolve().parents[3]/'outputs/dispatch_host_trace_cpu_review.json'
    print(json.dumps(run(output)))
