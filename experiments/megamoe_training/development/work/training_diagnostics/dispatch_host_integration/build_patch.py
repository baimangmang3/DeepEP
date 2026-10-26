"""Generate isolated source copies/unified patch; never writes frozen inputs."""
import ast,difflib,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
INPUTS={'deep_ep/buffers/elastic.py':WORK/'DeepEP/deep_ep/buffers/elastic.py',
 'p1_pointwise/training_adapter.py':WORK/'training_candidates/p1_host_v2/p1_pointwise/training_adapter.py',
 'p1_pointwise/validate_training_p1.py':WORK/'training_candidates/p1_host_v2/p1_pointwise/validate_training_p1.py'}
def replace(s,old,new):
    if s.count(old)!=1:raise ValueError('Expected one exact source anchor: '+old[:60])
    return s.replace(old,new)
def wrap_node(source,node,statement):
    lines=source.splitlines(keepends=True);a=node.lineno-1;b=node.end_lineno
    indent=' '*(node.col_offset)
    lines[a:b]=[indent+statement+'\n']+['    '+s for s in lines[a:b]]
    return ''.join(lines)
def build():
    design=json.loads((WORK.parent/'outputs/training_dispatch_host_integration_design.json').read_bytes().decode('utf-8'))
    patch=[];manifest={}
    for name,path in INPUTS.items():
        original=path.read_bytes().decode('utf-8');s=original
        key=str(path.relative_to(WORK.parent)).replace('\\','/')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=design['reviewed_source_sha256'][key]:raise ValueError('Frozen source drift '+key)
        if name.startswith('deep_ep/'):
            s=replace(s,'from .._native import _unavailable, npu_api, require_native',
                'from dispatch_host_integration.hooks import retain_notify, native_label\nfrom .._native import _unavailable, npu_api, require_native')
            s=replace(s,'        require_native("ElasticBuffer.dispatch")\n','        _trace_cached_handle = handle is not None\n        require_native("ElasticBuffer.dispatch")\n')
            s=replace(s,'                    result._rank_return_plan = rank_dedup\n                    result._buffer_token = self._handle_token\n',
                '                    result._rank_return_plan = rank_dedup\n                    result._buffer_token = self._handle_token\n                    retain_notify(self, result, runtime.native, rows, count_list)\n')
            tree=ast.parse(s);node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(c,ast.Call) and ast.unparse(c.func)=='native.dispatch' for c in ast.walk(n)))
            s=wrap_node(s,node,'with native_label(self, runtime.native, handle, cached_handle=_trace_cached_handle):')
        elif name.endswith('training_adapter.py'):
            s=replace(s,'import torch\n','import torch\nfrom dispatch_host_integration.hooks import routed_phase\n')
            tree=ast.parse(s);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_RoutedP1')
            for fn in sorted((n for n in cls.body if isinstance(n,ast.FunctionDef)),key=lambda n:n.lineno,reverse=True):
                # Backward needs its existing engine assignment before entering the scope.
                first=1 if fn.name=='backward' else 0
                if fn.name=='backward':assert ast.unparse(fn.body[0].targets[0])=='(engine, handle, counts)'
                lines=s.splitlines(keepends=True);a=fn.body[first].lineno-1;b=fn.body[-1].end_lineno
                lines[a:b]=[' '*8+f'with routed_phase(engine, "{fn.name}"):\n']+['    '+line for line in lines[a:b]]
                s=''.join(lines)
        else:
            s=replace(s,'import sys\n','import sys\nfrom dispatch_host_integration.hooks import collect_with_trace, get_trace, close_trace, bind_case, fail\n')
            s=replace(s,"    ap.add_argument('--run-npu',action='store_true')", "    ap.add_argument('--run-npu',action='store_true')\n    ap.add_argument('--dispatch-host-trace-dir',type=Path)\n    ap.add_argument('--dispatch-trace-run-id')\n    ap.add_argument('--dispatch-trace-process-start')")
            s=replace(s,"        identity,loaded_pointwise=collective_checked(torch,dist,\n            lambda:collect_identity(root,args.pointwise_library,execution,base),'candidate identity/load')",
                "        from deep_ep._native import require_native\n        identity,loaded_pointwise=collective_checked(torch,dist,\n            lambda:collect_with_trace(lambda:collect_identity(root,args.pointwise_library,execution,base),\n                lambda:require_native('explicit diagnostic host trace'),args,rank),'candidate identity/load')")
            s=replace(s,"        args.weight_dtype={'bf16':torch.bfloat16,'fp32':torch.float32}[args.weight_dtype]", "        host_trace=get_trace()\n        report['diagnostic_only']=host_trace is not None\n        args.weight_dtype={'bf16':torch.bfloat16,'fp32':torch.float32}[args.weight_dtype]")
            s=replace(s,"        for case in execution['cases']:","        for case_index,case in enumerate(execution['cases']):")
            s=replace(s,"            try:\n                assert torch.is_grad_enabled()", "            try:\n                if host_trace is not None:\n                    bind_case(engine,host_trace,session=engine.buffer._runtime.native,\n                        session_identity=f'{args.dispatch_trace_run_id}/case{case_index}',case_index=case_index,case_name=case)\n                assert torch.is_grad_enabled()")
            s=replace(s,'            finally:\n                close_case_host(engine,host,report,case,sys.exc_info()[1])',
                "            except BaseException:\n                fail(getattr(engine,'_dispatch_host_trace_binding',None));raise\n            finally:\n                close_case_host(engine,host,report,case,sys.exc_info()[1])")
            s=replace(s,"        report['all_passed']=True\n", "        report['diagnostic_all_cases_passed']=True\n        report['all_passed']=host_trace is None  # Diagnostic reports cannot grant qualification.\n")
            s=replace(s,'    finally:\n        finish_host_report(host,report,dist,',
                "    finally:\n        trace_cleanup_error=None\n        try:close_trace()\n        except Exception as exc:\n            trace_cleanup_error=exc;report['diagnostic_trace_cleanup_error']=repr(exc);report['all_passed']=False\n        finish_host_report(host,report,dist,")
            s=replace(s,"write_text(json.dumps(report,indent=2)),sys.exc_info()[1])", "write_text(json.dumps(report,indent=2)),sys.exc_info()[1] or trace_cleanup_error)\n        if trace_cleanup_error is not None:raise trace_cleanup_error")
        ast.parse(s)
        target=ROOT/'candidate'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(s.encode('utf-8'))
        patch.extend(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        manifest[name]=dict(original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (ROOT/'integration.patch').write_bytes(''.join(patch).encode('utf-8'))
    return manifest
if __name__=='__main__':print(json.dumps(build()))
