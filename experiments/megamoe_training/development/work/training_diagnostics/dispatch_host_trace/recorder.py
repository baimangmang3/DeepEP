"""Opt-in host metadata only. No Torch import, transport, event or device completion claim."""
import contextvars
from contextlib import contextmanager
import hashlib,json,os,threading,time,weakref
from pathlib import Path

_LOCK=threading.Lock()
_INSTALLED=weakref.WeakKeyDictionary()
_ARGS=('session','tensors','tokens','topk','experts','rows','local_rows','fp8','stream_address')
_TENSORS=('input','output','dst','forward_list','forward_counts','weights','output_weights','scales','output_scales','plan')

def integer(value,name,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError('Missing/invalid integer '+name)
    return value

def text(value,name,limit=128):
    if not isinstance(value,str) or not value or len(value)>limit:raise ValueError('Missing/invalid '+name)
    return value

class DispatchHostTrace:
    def __init__(self,native,path,*,enabled=False,run_id,rank,process_start_time,
                 max_records=256,max_record_bytes=32768,max_total_bytes=2<<20):
        self.native=native;self.path=Path(path);self.enabled=enabled
        self.run_id=text(run_id,'run_id');self.rank=integer(rank,'rank')
        self.process_start_time=text(process_start_time,'process_start_time')
        self.max_records=integer(max_records,'max_records',2)
        self.max_record_bytes=integer(max_record_bytes,'max_record_bytes',1024)
        self.max_total_bytes=integer(max_total_bytes,'max_total_bytes',2*self.max_record_bytes)
        self._label=contextvars.ContextVar('dispatch_host_trace_label',default=None)
        self._io_lock=threading.Lock();self._file=None;self._used=False;self._failed=False
        self._records=self._bytes=self._sequence=self._pending=0
        self.recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.last_logging_error=None

    def __enter__(self):
        if self.enabled is not True:raise ValueError('Explicit enabled=True required')
        with _LOCK:
            if self._used or self.native in _INSTALLED:raise RuntimeError('Recorder/native installation is once-only')
            for name in ('dispatch','dispatch_layout'):
                if not callable(getattr(self.native,name,None)):raise ValueError('Missing native host metadata API '+name)
            self._original=self.native.dispatch;self._used=True
            self.path.parent.mkdir(parents=True,exist_ok=True);self._file=self.path.open('xb')
            self._wrapper=self._call
            try:self.native.dispatch=self._wrapper
            except BaseException:self._file.close();raise
            _INSTALLED[self.native]=True
        return self

    def __exit__(self,typ,value,tb):
        error=None
        if self._pending:error=RuntimeError('Recorder exited with native calls still in flight')
        with _LOCK:
            if self.native.dispatch is self._wrapper:self.native.dispatch=self._original
            else:error=RuntimeError('External dispatch hook changed; refusing to overwrite it')
        try:self._file.close()
        except BaseException as exc:error=error or exc
        if error:
            if value is not None:
                if hasattr(value,'add_note'):value.add_note(str(error))
            else:raise error
        return False

    @contextmanager
    def label(self,*,session,session_identity,case_index,case_name,phase,world,rank_rows,counts,
              cached_handle,rank_rows_source):
        if self._file is None or self._file.closed or self._failed:raise RuntimeError('Recorder not active/healthy')
        if self._label.get() is not None:raise ValueError('Nested labels forbidden')
        if phase not in ('forward','backward'):raise ValueError('Explicit forward/backward phase required')
        world=integer(world,'world',1)
        if self.rank>=world or world>512 or type(cached_handle) is not bool:raise ValueError('Invalid rank/world/cache metadata')
        if not isinstance(rank_rows,(list,tuple)) or len(rank_rows)!=world:raise ValueError('Host rank_rows required; no tensor reads permitted')
        if not isinstance(counts,(list,tuple)) or not counts or len(counts)>1024:raise ValueError('Bounded host expert counts required')
        rr=[integer(v,'rank_rows') for v in rank_rows];cc=[integer(v,'counts') for v in counts]
        data=dict(session=session,session_identity=text(session_identity,'session_identity'),case_index=integer(case_index,'case_index'),
            case_name=text(case_name,'case_name'),phase=phase,world=world,rank_rows=rr,counts=cc,
            cached_handle=cached_handle,rank_rows_source=text(rank_rows_source,'rank_rows_source'))
        token=self._label.set(data)
        try:yield
        finally:self._label.reset(token)

    def _bind(self,args,kwargs):
        if len(args)>len(_ARGS):raise ValueError('Unexpected native.dispatch positional ABI')
        values=dict(zip(_ARGS,args))
        for key,value in kwargs.items():
            # Raw pybind lambda also exposes arg0..arg8 rather than named arguments.
            name=_ARGS[int(key[3:])] if key.startswith('arg') and key[3:].isdigit() and int(key[3:])<len(_ARGS) else key
            if name not in _ARGS or name in values:raise ValueError('Unexpected/duplicate dispatch keyword ABI')
            values[name]=value
        if set(values)!=set(_ARGS):raise ValueError('Missing native.dispatch argument metadata')
        return values

    def _metadata(self,args,kwargs,label):
        v=self._bind(args,kwargs)
        if v['session'] is not label['session']:raise ValueError('Call session does not match explicit label')
        for name in ('tokens','topk','experts','rows','stream_address'):integer(v[name],name,1)
        integer(v['local_rows'],'local_rows')
        if type(v['fp8']) is not bool or not isinstance(v['tensors'],dict):raise ValueError('Invalid fp8/tensors metadata')
        if (v['local_rows']!=label['rank_rows'][self.rank] or sum(label['counts'])!=v['local_rows'] or
            v['rows']!=max(1,max(label['rank_rows']))):raise ValueError('Provided actual Notify host rows/counts disagree with dispatch')
        t=v['tensors'];metadata={}
        peer_table=v['session'].peer_table
        if type(peer_table) is not bool:raise ValueError('Missing/invalid session.peer_table metadata')
        if peer_table and t.get('plan') is None:raise ValueError('Peer-table session requires its plan')
        def tensor_metadata(value):
            if len(value.shape)>8:raise ValueError('Unexpected tensor metadata rank')
            shape=[integer(x,'tensor dimension') for x in value.shape]
            return dict(shape=shape,dtype=text(str(value.dtype),'dtype'),data_ptr=integer(value.data_ptr(),'data_ptr'))
        for name in _TENSORS:
            if name not in t and name!='plan':raise ValueError('Missing tensor metadata '+name)
            value=t.get(name)
            if value is None:
                if name in ('input','output','dst','forward_list','forward_counts'):raise ValueError('Required tensor is None: '+name)
                metadata[name]=None;continue
            metadata[name]=tensor_metadata(value)
        # Snapshot incoming controls only. Never enable/create diagnostics or edit tensors.
        controls={}
        for name,default in (('dedup_group_size',8),('kernel_timing',False),('benchmark_barriers',False),
                             ('rotation_wqes',4),('detail_mode',None)):
            value=t.get(name,default)
            if value is not None and type(value) not in (int,bool):raise ValueError('Non-scalar host control '+name)
            controls[name]=dict(provided=name in t,value=value)
        profiles={name:dict(provided=name in t,tensor=None if t.get(name) is None else tensor_metadata(t[name]))
                  for name in ('profile','detail_profile','weight_profile')}
        if len(metadata['input']['shape'])!=2:raise ValueError('Input matrix metadata required')
        hidden=metadata['input']['shape'][1];integer(hidden,'H',1)
        if metadata['input']['shape'][0]!=v['tokens'] or metadata['output']['shape']!=[v['local_rows'],hidden]:raise ValueError('Input/output shape metadata mismatch')
        weights=t['weights'] is not None
        layout=self.native.dispatch_layout(self.rank,label['world'],v['tokens'],v['topk'],v['experts'],v['rows'],v['fp8'],weights,hidden)
        required=('workspace_bytes','address_stride','sync','peer_book','done','weight_inbox','weight_inbox_stride')
        if not isinstance(layout,dict) or len(layout)>32 or any(k not in layout for k in required):raise ValueError('Missing/unbounded native layout offsets')
        layout={text(k,'layout key'):integer(val,'native layout value') for k,val in layout.items()}
        capacity=integer(v['session'].capacity,'session.capacity',1)
        if layout['workspace_bytes']>capacity//2:raise ValueError('Native layout exceeds session dispatch half')
        api=getattr(self.native,'megamoe_workspace_address',None)
        workspace=dict(available=False,api=None,gap='Existing megamoe_workspace_address binding absent; no guessed pointer')
        if api is not None:
            if not callable(api):raise ValueError('Workspace API exists but is not callable')
            workspace=dict(available=True,api='native.megamoe_workspace_address(session)',address=integer(api(v['session']),'workspace base',1),
                boundary='Existing accessor invokes Session::check_context; no tensor value read or synchronization added')
        return dict(B=v['tokens'],K=v['topk'],E=v['experts'],H=hidden,rows=v['rows'],local_rows=v['local_rows'],fp8=v['fp8'],
            weights_present=weights,stream_handle=v['stream_address'],native_layout=layout,workspace_capacity=capacity,
            workspace=workspace,tensors=metadata,session_peer_table=peer_table,plan_key_present='plan' in t,
            incoming_controls=controls,incoming_profile_buffers=profiles,native_session_python_identity=id(v['session']),
            caller_host_metadata_boundary='rank_rows/counts explicitly supplied; logger never fetches tensor values or infers from stack')

    def _encode(self,record):
        raw=(json.dumps(record,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8',errors='strict')
        if len(raw)>self.max_record_bytes:raise ValueError('JSON record exceeds configured byte limit')
        return raw

    def _write(self,raw):
        if self._file.write(raw)!=len(raw):raise OSError('Short JSONL write')
        self._file.flush();self._records+=1;self._bytes+=len(raw)

    def _call(self,*args,**kwargs):
        label=self._label.get()
        if label is None:
            self._failed=True;raise ValueError('Explicit case/phase/session label missing; native not called')
        if self._failed or self._file.closed:raise RuntimeError('Recorder closed/failed; native not called')
        # Metadata failures, byte/record limits and before-write failures block native execution.
        try:metadata=self._metadata(args,kwargs,label)
        except BaseException:self._failed=True;raise
        public={k:v for k,v in label.items() if k!='session'}
        with self._io_lock:
            self._sequence+=1;sequence=self._sequence
            base=dict(run_id=self.run_id,rank=self.rank,PID=os.getpid(),process_start_time=self.process_start_time,
                call_sequence=sequence,operation='native.dispatch',recorder_sha256=self.recorder_sha256,**public)
            before=dict(base,event='before',host_time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),metadata=metadata)
            try:raw=self._encode(before)
            except BaseException:self._failed=True;raise
            if self._records+self._pending+2>self.max_records or self._bytes+self._pending*self.max_record_bytes+len(raw)+self.max_record_bytes>self.max_total_bytes:
                self._failed=True;raise ValueError('Record/file budget exhausted; native not called')
            try:self._write(raw)
            except BaseException:self._failed=True;raise
            self._pending+=1
        def terminal(event,**fields):
            with self._io_lock:
                try:
                    self._write(self._encode(dict(base,event=event,host_time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),
                        native_return_is_device_completion=False,**fields)))
                except BaseException as exc:self._failed=True;self.last_logging_error=str(exc);raise
                finally:self._pending-=1
        try:result=self._original(*args,**kwargs)
        except BaseException as exc:
            self._failed=True
            try:terminal('native_error',exception_type=type(exc).__module__+'.'+type(exc).__qualname__,exception_message=str(exc))
            except BaseException as log_error:
                if hasattr(exc,'add_note'):exc.add_note('Host trace terminal logging also failed: '+str(log_error))
            raise
        terminal('native_return',return_type=type(result).__module__+'.'+type(result).__qualname__)
        return result

if __name__=='__main__':
    print(json.dumps(dict(action='plan only; no Torch import or native installation',opt_in='DispatchHostTrace(...,enabled=True) context + explicit label()',
        scope='Host metadata; before/native_return/native_error never assert device completion',device_launch_authorized=False)))
