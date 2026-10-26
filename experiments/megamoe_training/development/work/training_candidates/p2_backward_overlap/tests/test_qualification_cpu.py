"""Isolated stdlib-only P2 gates; no Torch/Torch_NPU/PG/native initialization."""
import argparse,ast,copy,importlib.util,json,subprocess,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
import candidate_provenance as P
import validate_training_p2 as V
spec=importlib.util.spec_from_file_location('actual_frozen_base_provenance',WORK/'DeepEP/experiments/megamoe_training/tests/provenance.py')
BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)

def execution(schedule='serial',world=2,shared=True):
    values=P.SMALL[world]
    return dict(P.FIXED,backward_schedule=schedule,world=world,shape=dict(zip(('B','H','I','E','K'),values)),seed=4317,
        weight_dtype='torch.float32',shared=shared,cases=['skew','balanced','zero_gate','empty'],oracle_chunk=1024,max_reference_gib=12.,limits=copy.deepcopy(P.LIMITS))

def identity(e):
    files=[dict(role=r,bytes=1,sha256=P.ELASTIC_SHA if r=='deepep_python/buffers/elastic.py' else 'a'*64) for r in sorted(BASE.REQUIRED_ROLES)]
    bc=dict(files=files,versions=dict(torch='CPU-fixture',torch_npu='not-imported-fixture'),precision=dict(matmul_allow_hf32=False,allow_internal_format=False))
    base=dict(schema=1,content=bc,content_digest=BASE.canonical_digest(bc));BASE.validate_identity(base)
    files=[dict(role=f'{prefix}/{name}',bytes=1,sha256=(P.ADAPTER_SHA if prefix=='p2' else P.P1_ADAPTER_SHA) if name=='training_adapter.py' else 'a'*64)
           for prefix,names in (('p2',P.SOURCES),('p1',P.P1_SOURCES)) for name in names]
    files.extend(dict(role='native.'+name,bytes=1,sha256='a'*64) for name in ('catlass','pointwise'))
    c=dict(base_identity=base,execution=e,files=files,mapped_libraries={'loaded/libfixture.so':dict(bytes=1,sha256='a'*64)})
    return seal(dict(schema=P.SCHEMA,content=c))

def seal(value):value['content_digest']=P.digest(value['content']);return value

def report(rank,e,current):
    b,h,i,_,k=(e['shape'][x] for x in ('B','H','I','E','K'));cases=[];cleanup=[];world=e['world']
    dtype=e['weight_dtype']
    for j,name in enumerate(e['cases']):
        counts=P.expected_counts(e,rank,name);rows=sum(counts)
        shapes=dict(x=[b,h],xe=[rows,h],w1=[4,2*i,h],gu=[rows,2*i],a=[rows,i],w2=[4,h,i],z=[rows,h],y=[b,h])
        dt=dict(Y='torch.bfloat16',dx='torch.bfloat16',dp='torch.float32',dW1='torch.float32',dW2='torch.float32',leaf_dW1=dtype,leaf_dW2=dtype)
        sizes=dict(Y=b*h,dx=b*h,dp=b*k,dW1=8*i*h,dW2=4*h*i,leaf_dW1=8*i*h,leaf_dW2=4*h*i)
        if e['shared']:
            dt.update(dSW1='torch.float32',dSW2='torch.float32',leaf_dSW1=dtype,leaf_dSW2=dtype)
            sizes.update(dSW1=2*i*h,dSW2=i*h,leaf_dSW1=2*i*h,leaf_dSW2=i*h)
        metrics={n:dict(passed=True,finite=True,out_of_tolerance_elements=0,elements=v,actual_dtype=dt[n],reference_dtype=dt[n],limits=P.LIMITS['Y'],
            RMS_error=0.,reference_RMS=0.,RMS_relative_error=0.,max_abs_error=0.,max_relative_error_floor_1e_8=0.,max_reference_abs=0.) for n,v in sizes.items()}
        cases.append(dict(case=name,all_ranks_numerical_pass=True,counts_match=True,shapes_match=True,dtypes_match=True,schedule_match=True,
            backward_schedule=e['backward_schedule'],candidate_counts=counts,oracle_counts=counts,candidate_shapes=shapes,dtypes=dt,metrics=metrics))
        stage=f'P2/case-close/{j}/{name}'
        peers=[dict(rank=r,case_index=j,case=name,stage=stage,close_error=None,body_error=None) for r in range(world)]
        cleanup.append(dict(case_index=j,case=name,stage=stage,close_error=None,body_error=None,all_ranks_closed=True,all_ranks_case_passed=True,peer_status=peers))
    return dict(rank=rank,world=world,backend=P.FIXED['backend'],all_passed=True,execution=e,shape=e['shape'],seed=e['seed'],weight_dtype=dtype,
        shared=e['shared'],limits=P.LIMITS,identity=current,identity_final=current,cases=cases,case_cleanup=cleanup,
        peer_before_PG_teardown=[dict(rank=r,error=None) for r in range(world)],peer_exit_status=[dict(rank=r,errors=[]) for r in range(world)],
        final_identity_peer_status=[dict(rank=r,error=None) for r in range(world)])

def run():
    checks=[]
    def ok(name,fn):fn();checks.append(name)
    def reject(fn):
        try:fn()
        except (ValueError,FileNotFoundError):return
        raise AssertionError('Expected gate rejection')
    e=execution();current=identity(e);ok('actual_frozen_provenance_plus_P2_schema',lambda:P.validate_identity(current,BASE))
    for old in (1,'p1-training-1','p3-training-1','p2-training-1'):
        v=copy.deepcopy(current);v['schema']=old
        ok('old_schema_rejected:'+str(old),lambda v=v:reject(lambda:P.validate_identity(v,BASE)))
    overlap=identity(execution('overlap'))
    ok('another_schedule_identity_rejected',lambda:reject(lambda:P.require_identity_match(current,overlap,BASE)))
    for world in (2,16):
        for schedule in ('serial','overlap'):ok(f'small_shape_config:{world}:{schedule}',lambda world=world,schedule=schedule:P.check_execution(execution(schedule,world)))
    for change in ('missing_schedule','K','E','cases','limits','diagnostic_P1','adapter','source','SO','dependency'):
        value=copy.deepcopy(current);c=value['content']
        if change=='missing_schedule':c['execution'].pop('backward_schedule')
        elif change in ('K','E'):c['execution']['shape'][change]=8 if change=='K' else 64
        elif change=='cases':c['execution']['cases']=['balanced']
        elif change=='limits':c['execution']['limits']['dp']['atol']=.01
        elif change in ('diagnostic_P1','adapter','source','SO'):
            role={'diagnostic_P1':'p1/training_adapter.py','adapter':'p2/training_adapter.py','source':'p2/validate_training_p2.py','SO':'native.pointwise'}[change]
            next(v for v in c['files'] if v['role']==role)['sha256']='b'*64
        else:c['mapped_libraries']['loaded/libfixture.so']['sha256']='b'*64
        seal(value)
        ok('identity_change:'+change,lambda value=value:reject(lambda:P.require_identity_match(value,current,BASE)))
    v=copy.deepcopy(current);v['diagnostic_import_paths']={'native':'/different/host/path'}
    ok('diagnostic_absolute_paths_excluded_from_digest',lambda:P.require_identity_match(v,current,BASE))
    v=copy.deepcopy(current);v['content']['mapped_libraries']['loaded/lazy.so']=dict(bytes=1,sha256='b'*64);seal(v)
    ok('final_only_lazy_added_allowed',lambda:P.final_identity(current,v,BASE))
    for change in ('missing_final','removed','changed'):
        v=copy.deepcopy(current)
        if change=='missing_final':v=None
        elif change=='removed':v['content']['mapped_libraries']={};seal(v)
        else:v['content']['mapped_libraries']['loaded/libfixture.so']['sha256']='b'*64;seal(v)
        ok('final_gate:'+change,lambda v=v:reject(lambda:P.final_identity(current,v,BASE)))
    with tempfile.TemporaryDirectory() as tmp:
        directory=Path(tmp)
        def write(reports):
            for path in directory.glob('rank*.json'):path.unlink()
            for r in reports:(directory/f"rank{r['rank']}.json").write_text(json.dumps(r),encoding='utf-8')
        originals=[report(r,e,current) for r in range(2)];write(originals)
        ok('complete_shared_11metric_reports',lambda:P.require_reports(directory,e,current,BASE))
        lazy=copy.deepcopy(current)
        lazy['content']['mapped_libraries']['loaded/liblazy_hccl.so']=dict(bytes=7,sha256='b'*64)
        base_content=lazy['content']['base_identity']['content']
        base_content['files'].append(dict(role='loaded_dependency/liblazy_hccl.so',bytes=7,sha256='b'*64))
        lazy['content']['base_identity']['content_digest']=BASE.canonical_digest(base_content);seal(lazy)
        rs=copy.deepcopy(originals)
        for item in rs:item['identity_final']=lazy
        write(rs);ok('report_initial_to_final_current_lazy_dependency_addition',lambda:P.require_reports(directory,e,lazy,BASE))
        for change in ('old_library_changed','old_library_removed','source_changed','schedule_changed','final_current_mismatch'):
            changed=copy.deepcopy(lazy);c=changed['content']
            if change=='old_library_changed':
                c['mapped_libraries']['loaded/libfixture.so']['sha256']='c'*64
                for item in c['files']:
                    if item['role'].startswith('native.'):item['sha256']='c'*64
                bc=c['base_identity']['content']
                next(item for item in bc['files'] if item['role']=='native.catlass')['sha256']='c'*64
                c['base_identity']['content_digest']=BASE.canonical_digest(bc)
            elif change=='old_library_removed':c['mapped_libraries'].pop('loaded/libfixture.so')
            elif change=='source_changed':next(item for item in c['files'] if item['role']=='p2/validate_training_p2.py')['sha256']='c'*64
            elif change=='schedule_changed':c['execution']['backward_schedule']='overlap'
            else:c['mapped_libraries']['loaded/liblazy_hccl.so']['sha256']='c'*64
            seal(changed)
            mutated=copy.deepcopy(rs)
            if change=='final_current_mismatch':expected=changed # Report's valid final stays unchanged.
            else:
                expected=changed
                for item in mutated:item['identity_final']=changed
            write(mutated)
            ok('lazy_report_reject:'+change,lambda expected=expected:reject(lambda:P.require_reports(directory,e,expected,BASE)))
        write(originals)
        e16=execution(world=16);i16=identity(e16);write([report(r,e16,i16) for r in range(16)])
        ok('complete16rank_fourcase_11metric_reports',lambda:P.require_reports(directory,e16,i16,BASE))
        (directory/'rank12.json').unlink()
        ok('rank12_missing_rejected',lambda:reject(lambda:P.require_reports(directory,e16,i16,BASE)))
        write(originals)
        for change in ('missing_rank','P1_report','P3_report','schedule','metric','shared_leaf','metric_finite','metric_RMS','metric_absmax','metric_limits','counts','shape','cleanup','peer_close','peer_prior','peer_PG','final_identity','final_peer','diagnostic_report'):
            rs=copy.deepcopy(originals)
            if change=='missing_rank':rs.pop()
            elif change in ('P1_report','P3_report'):rs[0]['identity']['schema']='p1-training-1' if change=='P1_report' else 'p3-training-1'
            elif change=='schedule':rs[0]['cases'][0]['backward_schedule']='overlap'
            elif change=='metric':rs[0]['cases'][0]['metrics'].pop('dp')
            elif change=='shared_leaf':rs[0]['cases'][0]['metrics'].pop('leaf_dSW2')
            elif change=='metric_finite':rs[0]['cases'][0]['metrics']['dp']['finite']=False
            elif change=='metric_RMS':rs[0]['cases'][0]['metrics']['dp']['RMS_error']=.1
            elif change=='metric_absmax':rs[0]['cases'][0]['metrics']['dp']['max_abs_error']=.1
            elif change=='metric_limits':rs[0]['cases'][0]['metrics']['dp']['limits']['atol']=.1
            elif change=='counts':rs[0]['cases'][0]['candidate_counts'][0]+=1
            elif change=='shape':rs[0]['cases'][0]['candidate_shapes']['xe'][0]+=1
            elif change=='cleanup':rs[0].pop('case_cleanup')
            elif change=='peer_close':rs[0]['case_cleanup'][0]['peer_status'][1]['close_error']='remote close'
            elif change=='peer_prior':rs[0]['peer_before_PG_teardown'][1]['error']='remote prior'
            elif change=='peer_PG':rs[0]['peer_exit_status'][1]['errors']=['remote PG']
            elif change=='final_identity':rs[0].pop('identity_final')
            elif change=='final_peer':rs[0]['final_identity_peer_status'][1]['error']='remote identity'
            else:rs[0]['diagnostic_only']=True
            write(rs);ok('report_reject:'+change,lambda:reject(lambda:P.require_reports(directory,e,current,BASE)))
        write(originals);ok('serial_reports_cannot_authorize_overlap',lambda:reject(lambda:P.require_reports(directory,overlap['content']['execution'],overlap,BASE)))
    root_args=types.SimpleNamespace(world_size=16,B=8192,H=6144,I=2048,E=64,K=8,cases=None,seed=4317,weight_dtype='bf16',
        shared=False,oracle_chunk=1024,max_reference_gib=12.,backward_schedule='serial')
    ok('main_without_explicit_balanced_reject',lambda:reject(lambda:P.execution_from_args(root_args,16)))
    root_args.cases='balanced';ok('main_explicit_balanced_accept',lambda:P.execution_from_args(root_args,16))
    root_args.cases='skew,balanced,zero_gate,empty';ok('main_fourcases_reject',lambda:reject(lambda:P.execution_from_args(root_args,16)))
    actual_p1=ROOT.parent/'p1_pointwise';frozen=WORK/'DeepEP/experiments/megamoe_training'
    ok('actual_original_package_layout_and_adapter_hash',lambda:P.check_source_layout(frozen,ROOT,actual_p1))
    ok('overlapping_roots_rejected',lambda:reject(lambda:P.check_imports(frozen,ROOT,ROOT)))
    def fake_import(name):
        if name=='provenance':path=frozen/'tests/provenance.py'
        elif name=='validate_training_npu':path=frozen/'tests/validate_training_npu.py'
        elif name=='candidate_provenance':path=ROOT/'candidate_provenance.py'
        else:
            package,*suffix=name.split('.');folder={'megamoe_training':frozen/'megamoe_training','p2_backward_overlap':ROOT,'p1_pointwise':actual_p1}[package]
            path=folder/(suffix[0]+'.py' if suffix else '__init__.py')
        return types.SimpleNamespace(__file__=str(path))
    with patch.object(P.importlib,'import_module',side_effect=fake_import):ok('actual_import_paths',lambda:P.check_imports(frozen,ROOT,actual_p1))
    for name in ('p2_backward_overlap.training_adapter','p1_pointwise.training_adapter','megamoe_training.gemm','candidate_provenance'):
        def bad(n):return types.SimpleNamespace(__file__='C:/escaped.py') if n==name else fake_import(n)
        with patch.object(P.importlib,'import_module',side_effect=bad):ok('module_cache_escape:'+name,lambda:reject(lambda:P.check_imports(frozen,ROOT,actual_p1)))
    class Host:
        def __init__(self,kind):self.kind=kind;self.closed=False
        def rank(self):return 0
        def size(self):return 2
        def all_gather(self,value,stage):
            peer=dict(value,rank=1)
            if 'case-close' in stage and self.kind=='close':peer['close_error']='peer close failure'
            if 'before-PG' in stage and self.kind=='prior':peer['error']='peer prior failure'
            if 'after-PG' in stage and self.kind=='PG':peer['errors']=['peer PG failure']
            if stage=='P2/final-identity' and self.kind=='identity':peer['error']='peer identity failure'
            return [value,peer]
        def abort(self,error):pass
        def close(self):self.closed=True
    r=dict(all_passed=True);host=Host('close')
    try:V.close_case_host(types.SimpleNamespace(close=lambda:None),host,r,'skew')
    except RuntimeError:pass
    else:raise AssertionError('Peer close failure accepted')
    assert not r['all_passed'];checks.append('real_close_helper_peer_failure_stops_case')
    for kind in ('prior','PG'):
        r=dict(all_passed=True);host=Host(kind)
        try:V.finish_host_report(host,r,types.SimpleNamespace(is_initialized=lambda:False),lambda:None)
        except RuntimeError:pass
        else:raise AssertionError('Peer exit failure accepted')
        assert not r['all_passed'] and host.closed;checks.append('real_exit_helper_peer_'+kind)
    r=dict(all_passed=True);host=Host('identity')
    with patch.object(P.importlib,'import_module',return_value=BASE):error=P.finish_identity_host(host,r,lambda:current,current)
    assert error is not None and not r['all_passed'];checks.append('final_identity_failure_host_converges')
    # Execute argparse/default plan in isolated interpreter; no native/Torch import.
    argv=[sys.executable,'-I',str(ROOT/'validate_training_p2.py'),'--world-size','2','--E','8','--p1-source-root','/draft/p1_pointwise',
          '--pointwise-library','/draft/p1.so','--output','/draft/output']
    for schedule in ('serial','overlap'):
        proc=subprocess.run(argv+['--backward-schedule',schedule],capture_output=True,text=True)
        assert proc.returncode==0 and json.loads(proc.stdout)['backward_schedule']==schedule;checks.append('isolated_default_plan_'+schedule)
    proc=subprocess.run(argv,capture_output=True,text=True);assert proc.returncode==2 and '--backward-schedule' in proc.stderr
    checks.append('CLI_schedule_required')
    assert P.sha(ROOT/'training_adapter.py')==P.ADAPTER_SHA
    assert not any(n=='torch' or n.startswith(('torch.','torch_npu','deep_ep')) for n in sys.modules)
    return dict(passed=True,checks=len(checks),details=checks,scope='Isolated stdlib only; no Torch/Torch_NPU/PG/UDMA/native init; synthetic report gates, not device qualification',
        adapter_sha256=P.sha(ROOT/'training_adapter.py'),fixed_shared_metric_count=11,
        isolation=dict(python_isolated=bool(sys.flags.isolated),torch_modules=[n for n in sys.modules if n=='torch' or n.startswith('torch.')],
            torch_npu_modules=[n for n in sys.modules if n.startswith('torch_npu')],deep_ep_modules=[n for n in sys.modules if n.startswith('deep_ep')]),
        remaining='Actual collect_identity/native mapping load and P2 all-rank precision/cleanup unexecuted. Existing CPU36 math evidence remains separate; not repeated.')

if __name__=='__main__':
    result=run();(ROOT/'tests/qualification_cpu_results.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'));print(json.dumps(result))
