"""Opt-in CPU import/CDLL/maps preflight only; never an engine/session/tensor/PG.

Default plan imports only stdlib. Run mode intentionally exercises prepare's
precision setters and actual native loaders, while asserting NPU stays uninitialized.
All roots and the original pointwise SO are explicit; no historical report is rewritten.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types


def plans():
    result=[]
    for world,b,h,i,e,k in ((2,32,128,128,8,4),(16,32,128,128,64,8),(16,8192,6144,2048,64,8)):
        for shared,dtype in ((False,'bf16'),(True,'bf16'))+(((True,'fp32'),) if b==32 else ()):
            result.append(dict(world_size=world,B=b,H=h,I=i,E=e,K=k,shared=shared,weight_dtype=dtype,
                cases='balanced' if b==8192 else 'balanced,zero_gate,empty,skew',seed=4317,
                oracle_chunk=1024,max_reference_gib=12.))
    return result


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-cpu',action='store_true')
    for name in ('training-source-root','candidate-source-root','pointwise-library','output'):
        p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.run_cpu and any(getattr(a,k) is None for k in vars(a) if k!='run_cpu'):
        p.error('--run-cpu requires all explicit source/library/artifact/output paths')
    return a


def snapshot(torch,dist,runtime):
    result=dict(npu_initialized=torch.npu.is_initialized(),PG_initialized=dist.is_initialized(),
                ElasticBuffer_owner_exists=getattr(runtime,'_OWNER',None) is not None)
    if any(result.values()):raise RuntimeError('CPU-only state contract violated: '+repr(result))
    return result


def inspect_identity(identity,provenance,base):
    provenance.validate_identity(identity,base)
    maps=identity['content']['mapped_libraries'];actual={v['path']:v for v in identity['actual_maps']}
    dual={pkg:[role for role in maps if role.startswith('package/'+pkg+'/') and Path(role).name.startswith('_C.')]
          for pkg in ('deep_ep','torch_npu')}
    if any(len(roles)!=1 for roles in dual.values()):raise ValueError('Expected both actual imported package _C FD objects')
    runtime_base=base.collect_runtime_identity()
    selected=Path(identity['diagnostic_paths']['native.pointwise']).resolve(strict=True)
    provenance.bind_selected_maps(identity,runtime_base,selected,actual)
    # Repeat the exact-path binding in this probe using actual current loader paths.
    selected_FD={role:actual[path] for role,path in runtime_base['diagnostic_paths'].items()
                 if role.startswith(('native.','loaded_dependency/'))}
    selected_FD['native.pointwise']=actual[str(selected)]
    return dict(dual_C_roles=dual,selected_native_and_dependency_FD=selected_FD,
                base_loader_paths=runtime_base['diagnostic_paths'])



def finish_state(report,check,body_error):
    """Keep the first prepare/collector error; report teardown observation separately."""
    try:report['after']=check()
    except Exception as exc:
        import traceback
        report.update(passed=False,after_state_error=repr(exc),after_state_traceback=traceback.format_exc())
        if body_error is None:raise


def run(a,report):
    paths={k:Path(v).resolve(strict=True) for k,v in vars(a).items() if k not in ('run_cpu','output')}
    a=types.SimpleNamespace(**paths)
    root=a.candidate_source_root
    if root.name!='p1_pointwise':raise ValueError('Candidate must be an importable p1_pointwise package')
    sys.path[:0]=[str(root),str(a.training_source_root/'tests'),str(a.training_source_root),
                 str(root.parent)]
    import torch
    import torch_npu
    import torch.distributed as dist
    runtime=importlib.import_module('deep_ep._runtime')
    report['before']=snapshot(torch,dist,runtime)
    body_error=None
    try:
        spec=importlib.util.spec_from_file_location('p1_cpu_validator',root/'validate_training_p1.py')
        v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
        provenance=importlib.import_module('candidate_provenance')
        if Path(provenance.__file__).resolve()!=root/'candidate_provenance.py':raise ValueError('Provenance escaped explicit root')
        report['plans']=[];report['identities']=[]
        for index,plan in enumerate(plans()):
            args=types.SimpleNamespace(**vars(a),**plan)
            base,ref,candidate,prepared_root,reservation,execution=v.prepare(args,plan['world_size'],torch,torch_npu)
            identity,wrappers=v.collect_identity(prepared_root,args.pointwise_library,execution,base)
            evidence=inspect_identity(identity,provenance,base)
            if identity.get('schema')!='p1-training-2':raise ValueError('Require reviewed P1 FD namespace schema2')
            repeated,_=v.collect_identity(prepared_root,args.pointwise_library,execution,base)
            difference=provenance.final_identity(identity,repeated,base)
            if repeated['content_digest']!=identity['content_digest']:raise ValueError('Repeated collector content identity unstable')
            report['repeat_collector_stable']=True
            report['plans'].append(dict(plan,execution=execution,reservation=reservation,evidence=evidence,identity_difference=difference,repeat_final_digest=repeated['content_digest'],passed=True))
            report['identities'].append(dict(initial=identity,repeat_final=repeated))
            snapshot(torch,dist,runtime)
        trace=[k for k in sys.modules if k.startswith('dispatch_host_integration')]
        if trace:raise ValueError('Unexpected diagnostic integration import')
        report['host_trace_modules']=trace;report['passed']=True
    except BaseException as exc:
        body_error=exc
        raise
    finally:
        finish_state(report,lambda:snapshot(torch,dist,runtime),body_error)


def main():
    a=parse()
    report=dict(passed=False,scope='CPU import/prepare/CDLL/actual FD maps only; not full-training/device/performance qualification',
        probe_sha256=sha(__file__),argv=sys.argv,plans=plans(),suggested_deployment='candidate-p1-qualification-schema2-20261026/{p1_pointwise,comparison/gate.py}',
        frozen_P0='/home/zzy/deploy/megamoe-20261024/training/qualified-p0-20261025/experiments/megamoe_training')
    if not a.run_cpu:
        report['mode']='plan-only; Torch not imported';print(json.dumps(report,indent=2));return
    error=None
    try:run(a,report)
    except Exception as exc:
        import traceback
        error=exc;report.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
    finally:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_bytes((json.dumps(report,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    if error is not None:raise SystemExit(1)
    print(json.dumps(dict(passed=True,plans=len(report['plans']),output=str(a.output),probe_sha256=report['probe_sha256'])))


if __name__=='__main__':main()
