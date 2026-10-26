"""Explicit labels in actual routed methods; host metadata only, diagnostic identity only."""
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
from pathlib import Path

_PHASE=contextvars.ContextVar('dispatch_explicit_phase',default=None)
_ACTIVE=None

def process_start_identity(stat_path=Path('/proc/self/stat'),boot_path=Path('/proc/sys/kernel/random/boot_id')):
    # starttime is proc field22; use final ')' because comm may contain spaces/parentheses.
    body=stat_path.read_text(encoding='utf-8').rsplit(')',1)[1].split()
    ticks=int(body[19]);boot=boot_path.read_text(encoding='utf-8').strip()
    if ticks<0 or not boot:raise ValueError('Missing actual process-start identity')
    return f'proc_start_ticks={ticks};boot_id={boot}'

@dataclass
class Binding:
    trace:object
    session:object
    session_identity:str
    case_index:int
    case_name:str
    failed:bool=False

def install_trace(native,path,**kwargs):
    global _ACTIVE
    if _ACTIVE is not None:raise RuntimeError('Diagnostic trace already active')
    from .recorder import DispatchHostTrace
    trace=DispatchHostTrace(native,path,enabled=True,**kwargs)
    trace.__enter__();_ACTIVE=trace
    return trace

def close_trace():
    global _ACTIVE
    trace=_ACTIVE
    if trace is not None:
        _ACTIVE=None;trace.__exit__(None,None,None)

def collect_with_trace(action,native_provider,args,rank):
    value=action()
    if args.dispatch_host_trace_dir is not None:
        if any(args.output.glob('rank*.json')):raise ValueError('Diagnostic output must not contain existing qualification reports')
        install_trace(native_provider(),args.dispatch_host_trace_dir/f'rank{rank}.jsonl',
            run_id=args.dispatch_trace_run_id,rank=rank,
            process_start_time=args.dispatch_trace_process_start or process_start_identity())
    return value

def get_trace():return _ACTIVE

def bind_case(engine,trace,*,session,session_identity,case_index,case_name):
    try:
        if trace._failed:raise RuntimeError('Trace already failed; no new session admitted')
        if engine.buffer._runtime.native is not session:raise ValueError('Explicit validator session is not engine session')
        if getattr(engine.buffer,'_dispatch_host_trace_binding',None) is not None:raise RuntimeError('Case already bound')
        binding=Binding(trace,session,session_identity,case_index,case_name)
        engine._dispatch_host_trace_binding=binding;engine.buffer._dispatch_host_trace_binding=binding
    except BaseException:trace._failed=True;raise

def fail(binding):
    if binding is not None:binding.failed=True;binding.trace._failed=True

@contextmanager
def routed_phase(engine,phase):
    binding=getattr(engine,'_dispatch_host_trace_binding',None)
    if binding is None:yield;return
    try:
        if phase not in ('forward','backward') or binding.failed or binding.trace._failed:raise RuntimeError('Diagnostic phase/trace failed')
        if engine.buffer._runtime.native is not binding.session:raise ValueError('Old diagnostic session binding')
        if _PHASE.get() is not None:raise RuntimeError('Nested routed phase forbidden')
        token=_PHASE.set((binding,phase))
        try:yield
        finally:_PHASE.reset(token)
    except BaseException:fail(binding);raise

def retain_notify(buffer,handle,session,rows,count_list):
    binding=getattr(buffer,'_dispatch_host_trace_binding',None)
    if binding is None:return
    try:
        if binding.failed or binding.trace._failed or session is not binding.session:raise RuntimeError('Notify diagnostic session failed/stale')
        if any(type(v) is not int or v<0 for v in (*rows,*count_list)):raise ValueError('Notify host summary malformed')
        handle._dispatch_host_snapshot=(session,tuple(rows),tuple(count_list))
    except BaseException:fail(binding);raise

@contextmanager
def native_label(buffer,session,handle,*,cached_handle):
    binding=getattr(buffer,'_dispatch_host_trace_binding',None)
    if binding is None:yield;return
    try:
        active=_PHASE.get()
        if binding.failed or binding.trace._failed or active is None or active[0] is not binding:raise RuntimeError('Missing explicit method phase or failed binding')
        if session is not binding.session:raise ValueError('Native call session stale')
        saved=getattr(handle,'_dispatch_host_snapshot',None)
        if saved is None or saved[0] is not session:raise ValueError('Missing/stale actual Notify host snapshot')
        with binding.trace.label(session=session,session_identity=binding.session_identity,case_index=binding.case_index,
            case_name=binding.case_name,phase=active[1],world=buffer._runtime.world_size,rank_rows=saved[1],counts=saved[2],
            cached_handle=cached_handle,rank_rows_source='existing Notify summary host rows/count_list; immutable handle snapshot'):
            yield
    except BaseException:fail(binding);raise
