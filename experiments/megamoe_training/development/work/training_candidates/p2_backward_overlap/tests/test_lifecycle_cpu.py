"""Execute actual P2 backward AST with symbolic tensors; no Torch/device."""
import ast
import hashlib
import json
from pathlib import Path
import types

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]


class Tensor:
    def detach(self):return self
    def to(self,*args):return self
    def contiguous(self):return self
    def float(self):return self
    def __mul__(self,other):return self
    def sum(self,**kw):return self


def main():
    source=ROOT/'training_adapter.py';tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_RoutedP2')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='backward')
    method.decorator_list=[];scope={'torch':types.SimpleNamespace(bfloat16='bf16')}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
    results=[]
    for schedule in ('serial','overlap'):
        for failure in (None,'dW1','dW2','dp_missing','event_missing'):
            log=[];retained=[Tensor(),Tensor()]
            class Event:
                released=False
                def current_stream_wait(self,release_handle=False):
                    log.append('wait_release' if release_handle else 'wait')
                    if release_handle:self.released=True;retained.clear()
            event=Event()
            class Buffer:
                def dispatch(self,*a,**k):log.append('dispatch');return Tensor(),None,None,None,None
                def combine(self,*a,**k):
                    assert k['async_with_compute_stream'] is True
                    log.append('combine_async')
                    return Tensor(),None if failure=='dp_missing' else Tensor(),None if failure=='event_missing' else event
            class Gemm:
                group_calls=0;weight_calls=0
                def grouped(self,*a,**k):
                    self.group_calls+=1;log.append('dA' if self.group_calls==1 else 'dXe');return Tensor()
                def weight_grad(self,*a,**k):
                    self.weight_calls+=1;name='dW'+str(self.weight_calls);log.append(name)
                    assert retained and not event.released,'inputs released before dW submission'
                    if failure==name:raise RuntimeError('injected '+name)
                    return Tensor()
            def gate(*a):log.append('gate');return Tensor()
            def activation(*a):log.append('dGU');return Tensor()
            engine=types.SimpleNamespace(closed=False,backward_schedule=schedule,buffer=Buffer(),gemm=Gemm(),
                pointwise=types.SimpleNamespace(gate=gate,activation_grad=activation))
            ctx=types.SimpleNamespace(engine=engine,handle=object(),counts=(1,),weight_dtypes=('bf16','bf16'),saved_tensors=(Tensor(),)*7)
            error=None
            try:scope['backward'](ctx,Tensor())
            except RuntimeError as exc:error=str(exc)
            assert bool(error)==(failure is not None)
            if failure=='event_missing':assert 'dW1' not in log and 'wait_release' not in log
            else:
                assert event.released and log[-1]=='wait_release'
                if failure not in ('dp_missing',):
                    if schedule=='serial':assert log.index('wait')<log.index('dW1')
                    else:assert 'wait' not in log
            results.append(dict(schedule=schedule,failure=failure,trace=log,passed=True,error=error))
    # Exercise actual retained-tensor/consumer-stream release implementation.
    events=WORK/'DeepEP/deep_ep/utils/event.py';event_tree=ast.parse(events.read_text())
    klass=next(n for n in event_tree.body if isinstance(n,ast.ClassDef) and n.name=='EventOverlap')
    wait=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='current_stream_wait');wait.decorator_list=[]
    log=[];env={'require_native':lambda *a:None,'npu_api':lambda:types.SimpleNamespace(current_stream=lambda device:'consumer')}
    exec(compile(ast.Module(body=[wait],type_ignores=[]),str(events),'exec'),env)
    tensor=types.SimpleNamespace(record_stream=lambda stream:log.append(('record',stream)))
    handle=types.SimpleNamespace(_device=0,current_stream_wait=lambda:log.append(('event_wait',)))
    overlap=types.SimpleNamespace(event=handle,extra_tensors=(tensor,),_hook=None)
    env['current_stream_wait'](overlap,release_handle=True)
    assert log==[('event_wait',),('record','consumer')] and overlap.event is None and overlap.extra_tensors is None
    results.append({'name':'actual EventOverlap waits/registers consumer before dropping refs','passed':True})
    result=dict(scope='CPU symbolic control/lifetime model only; no real transport, streams or overlap execution',checks=results,all_passed=True,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'__init__.py',source)},
        event_source_sha256=hashlib.sha256(events.read_bytes()).hexdigest(),
        missing_event_boundary='Rejected before dW; no completion handle exists to drain in that error path. Native actual combine must always supply a valid EventOverlap; model does not establish that hardware contract.')
    (ROOT/'tests/lifecycle_cpu_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_passed':True,'checks':len(results),'source_sha256':result['source_sha256']}))


if __name__=='__main__':main()
