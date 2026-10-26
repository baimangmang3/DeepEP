"""Stdlib gates only; actual archived component evidence plus negative fixtures."""
import copy,importlib.util,json,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
import candidate_provenance as P
import validate_training_p3 as V

def run():
    checks=[]
    def ok(name,fn):fn();checks.append(name)
    def rejects(fn):
        try:fn()
        except (ValueError,FileNotFoundError):return
        raise AssertionError('Expected rejection')
    base=types.SimpleNamespace(validate_identity=lambda value: value['content_digest'] if value.get('schema')==1 else (_ for _ in ()).throw(ValueError('base schema')))
    execution=dict(P.FIXED,world=16,shape=dict(B=8192,H=6144,I=2048,E=64,K=8),seed=4317,
        shared=False,weight_dtype='torch.bfloat16',cases=['balanced'],oracle_chunk=1024,max_reference_gib=12.,limits=P.LIMITS)
    roles={f'{prefix}/{f}' for prefix,files in (('p3',P.P3),('p1',P.P1),('up',P.UP_PACKAGE)) for f in files}|{'native.catlass','native.pointwise','native.up_swiglu'}
    b=dict(schema=1,content=dict(files=[dict(role='source.training',sha256='a'*64,bytes=1)],versions={'torch':'x','torch_npu':'x'},precision={'hf32':False}),content_digest='b'*64)
    content=dict(base_identity=b,execution=execution,files=[dict(role=r,sha256='a'*64,bytes=1) for r in sorted(roles)],
        standalone=dict(report_sha256='a'*64),mapped_libraries={'libx.so':dict(sha256='a'*64,bytes=1)})
    def identity(c):return dict(schema=P.SCHEMA,content=c,content_digest=P.digest(c))
    current=identity(content);ok('independent_P3_identity_valid',lambda:P.validate_identity(current,base))
    for schema in (1,'p1-training-1',None):
        x=copy.deepcopy(current);x['schema']=schema;ok('old_schema_rejected',lambda x=x:rejects(lambda:P.validate_identity(x,base)))
    for key,value in (('world',8),('schedule','overlap'),('shared',1),('local_experts',8),('gemm_cores',32.0)):
        x=copy.deepcopy(content);x['execution'][key]=value
        ok('execution_'+key,lambda x=x:rejects(lambda:P.validate_identity(identity(x),base)))
    for change in ('limits','E','missing_role','digest'):
        x=copy.deepcopy(content)
        if change=='limits':x['execution']['limits']['Y']['atol']=.01
        elif change=='E':x['execution']['shape']['E']=256
        elif change=='missing_role':x['files'].pop()
        z=identity(x)
        if change=='digest':z['content_digest']='0'*64
        ok(change,lambda z=z:rejects(lambda:P.validate_identity(z,base)))
    x=copy.deepcopy(content);x['mapped_libraries']['lazy.so']=dict(sha256='c'*64,bytes=1)
    x['base_identity']['content']['files'].append(dict(role='loaded_dependency/lazy.so',sha256='c'*64,bytes=1))
    ok('final_only_lazy_added',lambda:P.final_identity(current,identity(x),base))
    for change in ('changed','removed','source'):
        x=copy.deepcopy(content)
        if change=='changed':x['mapped_libraries']['libx.so']['sha256']='b'*64
        elif change=='removed':x['mapped_libraries']={}
        else:x['files'][0]['sha256']='b'*64
        ok('final_'+change,lambda x=x:rejects(lambda:P.final_identity(current,identity(x),base)))
    ok('final_missing_identity',lambda:rejects(lambda:P.final_identity(None,current,base)))
    actual=WORK.parent/'outputs/training_results_20261025_1413/p3-up-swiglu-unit/rack05os7/up-swiglu-results.json'
    report=json.loads(actual.read_text());up=ROOT.parent/'p3_up_swiglu'
    maps=P.checked_maps(report['identity_final']['mapped'])
    native=next(v['sha256'] for p,v in maps.items() if Path(p).name=='libmegamoe_up_swiglu.so')
    catlass=next(v['sha256'] for p,v in maps.items() if Path(p).name=='libmegamoe_training.so')
    ok('real_archived_68_component_source_SO_metrics',lambda:P.require_standalone(report,up,native,catlass,execution))
    for change in ('missing','source','so','final','metric','invocations','capacity','release_missing'):
        r=copy.deepcopy(report);n=native;e=copy.deepcopy(execution)
        if change=='missing':r={}
        elif change=='source':r['source_sha256']['up_swiglu.py']='0'*64
        elif change=='so':n='0'*64
        elif change=='final':r.pop('identity_final')
        elif change=='metric':r['cases'][0]['invocations'][0]['metrics']['GU']['badcount']=1
        elif change=='invocations':r['cases'][0]['invocations']=[]
        elif change=='capacity':e['shape']['H']=256;e['shape']['I']=6144
        if change=='release_missing':fn=lambda:P.require_standalone_audit({},actual,native)
        else:fn=lambda r=r,n=n,e=e:P.require_standalone(r,up,n,catlass,e)
        ok('component_'+change,lambda fn=fn:rejects(fn))
    audit=json.loads((actual.parent.parent/'unit-audit.json').read_text())
    ok('real_audit_rc_release_SO_and_report',lambda:P.require_standalone_audit(audit,actual,native))
    for key,value in (('node_rc',1),('node_rc',False),('all_own_released',False),('timeout',True),('reaped',False)):
        a=copy.deepcopy(audit);a['process'][key]=value
        ok('audit_reject_'+key,lambda a=a:rejects(lambda:P.require_standalone_audit(a,actual,native)))
    for key,value in (('report_sha256','0'*64),('release_evidence',{})):
        a=copy.deepcopy(audit);a[key]=value
        ok('audit_reject_'+key,lambda a=a:rejects(lambda:P.require_standalone_audit(a,actual,native)))
    for change in ('duplicate_case','metric_shape','metric_dtype','RMS','missing_metrics','main_missing'):
        r=copy.deepcopy(report)
        if change=='duplicate_case':r['cases'][1]=copy.deepcopy(r['cases'][0])
        elif change=='main_missing':r['cases'].pop()
        elif change=='metric_shape':r['cases'][1]['invocations'][0]['metrics']['A']['shape']=[999,1]
        elif change=='metric_dtype':r['cases'][1]['invocations'][0]['metrics']['A']['dtype']='torch.float32'
        elif change=='RMS':r['cases'][1]['invocations'][0]['metrics']['A']['RMS_error']=1.
        else:r['cases'][0]['invocations'][0]['metrics']={}
        ok('standalone_'+change,lambda r=r:rejects(lambda:P.require_standalone(r,up,native,catlass,execution)))
    small=copy.deepcopy(execution);small.update(world=2,shape=dict(B=32,H=128,I=128,E=8,K=8))
    ok('small_new_M256_allowed_with_unit_native_HI_coverage',lambda:P.require_standalone(report,up,native,catlass,small))
    # Actual imported child modules cannot escape declared roots via module cache.
    roots=(WORK/'DeepEP/experiments/megamoe_training',ROOT,ROOT.parent/'p1_pointwise',up)
    ok('real_source_package_layout_complete',lambda:P.check_source_layout(*roots))
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)/'candidate-p3-up-swiglu-20261025';folder.mkdir()
        ok('standalone_nonpackage_root_rejected',lambda:rejects(lambda:P.check_source_layout(*roots[:3],folder)))
        folder=Path(tmp)/'p3_up_swiglu';folder.mkdir()
        ok('incomplete_component_package_rejected',lambda:rejects(lambda:P.check_source_layout(*roots[:3],folder)))
    paths={}
    def module(name):
        if name=='provenance':path=roots[0]/'tests/provenance.py'
        elif name=='validate_training_npu':path=roots[0]/'tests/validate_training_npu.py'
        elif name.startswith('megamoe_training'):path=roots[0]/'megamoe_training'/('__init__.py' if name=='megamoe_training' else name.rsplit('.',1)[1]+'.py')
        else:
            package,*child=name.split('.');root={'p3_training':roots[1],'p1_pointwise':roots[2],'p3_up_swiglu':roots[3]}[package]
            path=root/(child[0]+'.py' if child else '__init__.py')
        paths[name]=path;return types.SimpleNamespace(__file__=str(path))
    with patch.object(P.importlib,'import_module',side_effect=module):ok('explicit_actual_import_paths',lambda:P.check_imports(*roots))
    for name in ('p3_training.training_adapter','p1_pointwise.pointwise','p3_up_swiglu.contract','megamoe_training.gemm'):
        def escaped(n):return types.SimpleNamespace(__file__='C:/escaped.py') if n==name else module(n)
        with patch.object(P.importlib,'import_module',side_effect=escaped):ok('module_escape_'+name,lambda:rejects(lambda:P.check_imports(*roots)))
    class Host:
        def rank(self):return 0
        def size(self):return 2
        def all_gather(self,value,stage):
            if stage=='P3/final-identity':return [value,dict(rank=1,error='remote identity error')]
            if stage.endswith('before-PG-teardown'):return [value,dict(rank=1,error='remote prior error')]
            return [value,dict(rank=1,errors=['remote PG error'])]
        def close(self):pass
    report={}
    with patch.object(P.importlib,'import_module',return_value=base):err=P.finish_identity_host(Host(),report,lambda:current,current)
    assert err is not None and report['all_passed'] is False;checks.append('remote_final_identity_failure_converges')
    report=dict(all_passed=True)
    try:V.finish_host_report(Host(),report,types.SimpleNamespace(is_initialized=lambda:False),lambda:None)
    except RuntimeError:pass
    else:raise AssertionError('Remote cleanup error accepted')
    assert not report['all_passed'];checks.append('remote_prior_PG_failure_nonzero')
    math_root=WORK.parent/'outputs/p3_training_math_cpu'
    math=json.loads((math_root/'cpu-results.json').read_text());isolation=json.loads((math_root/'isolation.json').read_text())
    execution_record=json.loads((math_root/'execution.json').read_text())
    assert math['all_passed'] is True and len(math['checks'])==16 and all(v['passed'] for v in math['checks'])
    assert execution_record['returncode']==0 and isolation['isolated'] is True
    for stage in ('before','after'):
        assert isolation[stage]['npu_initialized'] is False and not isolation[stage]['torch_npu_modules']
        assert isolation[stage]['process_group_initialized'] is False and not isolation[stage]['udma_modules']
    for name in ('training_adapter.py','__init__.py','tests/test_math_cpu.py'):
        assert execution_record['source_sha256']['p3_training/'+name]==P.sha(ROOT/name)
    checks.append('actual_CPU_Torch16_source_matched_and_isolated')
    return dict(passed=True,checks=len(checks),details=checks,actual_component_report_sha256=P.sha(actual),
        limitations=['Actual archived component report and actual audited rc/release checked offline',
        'No native collection/NPU/SHMEM here; actual CPU Torch16 checks independently collected by environment and source matched'])

if __name__=='__main__':
    result=run();Path(__file__).with_name('qualification_cpu_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
