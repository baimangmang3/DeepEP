"""Run real StoreGroup protocol on its existing in-memory CPU test Store."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time
import types

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]/'DeepEP'


def main():
    store_source=REPO/'deep_ep/_store.py'
    spec=importlib.util.spec_from_file_location('actual_host_store',store_source)
    store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)
    source=REPO/'experiments/megamoe/tests/test_store.py'
    tree=ast.parse(source.read_text(encoding='utf-8'));memory=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MemoryStore')
    ns={'threading':threading};exec(compile(ast.Module(body=[memory],type_ignores=[]),'existing_MemoryStore','exec'),ns)
    tree=ast.parse((ROOT/'host_control.py').read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('close_case_host','finish_host_report')]
    helper={'phase':lambda *args,**kwargs:None}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'actual_candidate_helpers','exec'),helper)
    checks=[]
    for scenario in ('success','rank12_failed_close','rank12_slow_close','rank12_stage_mismatch','rank12_exchange_failure','rank7_numerical_failure','rank5_report_IO_failure'):
        memory=ns['MemoryStore']();groups=[store.StoreGroup(memory,r,16,prefix=scenario,timeout=1) for r in range(16)]
        closed=[False]*16;calls=[0]*16;lock=threading.Lock()
        def worker(rank):
            group=groups[rank];report={'passed':True};error=None
            if scenario=='rank12_exchange_failure' and rank==12:
                group.all_gather=lambda *args:(_ for _ in ()).throw(OSError('injected TCPStore exchange failure'))
            class Engine:
                def close(self):
                    calls[rank]+=1
                    if scenario=='rank12_slow_close' and rank==12:time.sleep(.05)
                    with lock:closed[rank]=True
                    if scenario=='rank12_failed_close' and rank==12:raise RuntimeError('original SHMEM finalize status -3 (synthetic)')
            prior=None
            if scenario=='rank7_numerical_failure' and rank==7:prior=AssertionError('original numerical failure')
            if scenario=='rank5_report_IO_failure' and rank==5:prior=OSError('original report IO failure')
            case='zero_gate' if scenario=='rank12_stage_mismatch' and rank==12 else 'balanced'
            try:helper['close_case_host'](Engine(),group,report,case,prior)
            except Exception as exc:error=repr(exc)
            next_case=error is None
            if next_case:assert all(closed),'next case entered before every close returned'
            # Complete the actual StoreGroup close protocol on every scenario,
            # including already-aborted groups; this is a cleanup attempt, never
            # a way to clear the original case failure.
            teardown=[]
            fake_dist=types.SimpleNamespace(is_initialized=lambda:True,
                destroy_process_group=lambda:teardown.append('PG.destroy'))
            try:
                helper['finish_host_report'](group,report,fake_dist,
                    lambda:teardown.append('save'),None if error is None else RuntimeError(error))
            except Exception as exc:
                if error is None:error=repr(exc)
            assert teardown==['PG.destroy','save']
            if scenario in ('success','rank12_slow_close','rank12_failed_close','rank7_numerical_failure','rank5_report_IO_failure'):
                assert group._store is None,'normal host close/ack must finish even after case failure'
            return {'rank':rank,'next_case':next_case,'error':error,'report':report}
        with ThreadPoolExecutor(max_workers=16) as pool:results=list(pool.map(worker,range(16)))
        expected_success=scenario in ('success','rank12_slow_close')
        assert all(v['next_case']==expected_success for v in results)
        assert calls==[1]*16,'Even numerical/IO-failed ranks must attempt close'
        if not expected_success:assert all(v['report']['passed'] is False for v in results)
        if scenario=='rank12_failed_close':assert 'status -3' in results[12]['report']['case_cleanup'][0]['close_error']
        if scenario=='rank5_report_IO_failure':assert 'report IO failure' in results[5]['report']['case_cleanup'][0]['body_error']
        if scenario in ('rank12_stage_mismatch','rank12_exchange_failure'):
            first=memory.values[scenario+'/abort']
            groups[0].abort('later diagnostic must not replace first failure')
            assert memory.values[scenario+'/abort']==first,'StoreGroup.abort must preserve first cause'
        checks.append({'scenario':scenario,'ranks':16,'passed':True,'next_case_ranks':sum(v['next_case'] for v in results),
                       'close_called_once_each':True,'errors_sample':results[12]['error']})
    # Exit helper: required host phases/close precede final report I/O.
    for scenario in ('success','PG_cleanup_failure','remote_only_prior_error','remote_only_PG_failure','host_exit_failure','final_report_IO_failure'):
        events=[];report={'passed':True}
        class Host:
            def rank(self):return 0
            def all_gather(self,value,stage):
                events.append(stage)
                if scenario=='remote_only_prior_error' and stage=='comparison/before-PG-teardown':
                    return [value,{'rank':12,'error':'original peer prior error, synthetic'}]
                if scenario=='remote_only_PG_failure' and stage=='comparison/after-PG-teardown':
                    return [value,{'rank':1,'errors':['remote-only PG teardown failed']}]
                return [value]
            def close(self):
                events.append('host.close')
                if scenario=='host_exit_failure':raise RuntimeError('host exit failed')
        class Dist:
            @staticmethod
            def is_initialized():return True
            @staticmethod
            def destroy_process_group():
                events.append('PG.destroy')
                if scenario=='PG_cleanup_failure':raise RuntimeError('PG cleanup failed')
        def save():
            events.append('save')
            if scenario=='final_report_IO_failure':raise OSError('final IO failed')
        error=None
        try:helper['finish_host_report'](Host(),report,Dist,save)
        except Exception as exc:error=repr(exc)
        assert events==['comparison/before-PG-teardown','PG.destroy','comparison/after-PG-teardown','host.close','save']
        assert bool(error)==(scenario!='success')
        if scenario!='success':assert report['passed'] is False
        if scenario=='remote_only_prior_error':
            assert report['peer_before_PG_teardown'][1]['error']=='original peer prior error, synthetic'
        checks.append({'scenario':scenario,'passed':True,'events':events,'error':error})
    # Exercise actual logger flush behavior without Torch/NPU import.
    phase_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='phase')
    logged=[];phase_scope={'print':lambda value,**kw:logged.append((json.loads(value),kw))}
    exec(compile(ast.Module(body=[phase_fn],type_ignores=[]),'actual_phase_logger','exec'),phase_scope)
    phase_scope['phase']('CPU_phase',12)
    assert logged[0][0]['rank']==12 and logged[0][0]['time_ns']>0 and logged[0][0]['monotonic_ns']>0 and logged[0][1]=={'flush':True}
    phase_scope['print']=lambda *a,**k:(_ for _ in ()).throw(BrokenPipeError('CPU closed log pipe'))
    phase_scope['phase']('CPU_logging_failure',12)  # Cannot skip close/status convergence.
    checks.append({'scenario':'phase_JSON_flush_and_broken_pipe_safe','passed':True})
    result={'scope':'CPU-only actual StoreGroup logic with MemoryStore; no real TCP/NPU/HCCL; thread delays are test controls, not clocks from failing run',
            'checks':checks,'all_passed':True,'StoreGroup_sha256':hashlib.sha256(store_source.read_bytes()).hexdigest(),
            'candidate_sha256':hashlib.sha256((ROOT/'host_control.py').read_bytes()).hexdigest()}
    (ROOT/'host_control_cpu_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_passed':True,'scenarios':len(checks),'rank_scenarios':7,'candidate_sha256':result['candidate_sha256']}))


if __name__=='__main__':main()
