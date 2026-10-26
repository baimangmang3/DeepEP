"""Contract/identity/lifecycle gates only, no Torch/native/device import."""
import copy
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
from unittest.mock import patch
import zipfile
import qualification as q
from prepare_build_contract import make_contract,digest,EXPECTED_EP16_SOURCES,REV,PATCH
from validate import cleanup_owner,host_stage
import validate as validator
from host_store import StoreGroup
from host_exit import finish_host_report

ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1]


def main():
    checks=[]
    def passed(name):checks.append(dict(name=name,passed=True))
    def reject(name,action):
        try:action()
        except (ValueError,RuntimeError,KeyError):passed(name);return
        raise AssertionError('Accepted invalid '+name)
    state16=json.loads((WS/'outputs/ep_gmm_adapted_ep16_build/remote-final-build-state.json').read_text())
    state8=json.loads((WS/'outputs/ep_gmm_pinned_ep8_build/remote-final-build-state.json').read_text())
    assert state16['source_sha256']==EXPECTED_EP16_SOURCES
    with tempfile.TemporaryDirectory(dir=ROOT) as td:
        tmp=Path(td);wheel=tmp/'fixture.whl';sdksha=hashlib.sha256(b'fake SHMEM fixture only').hexdigest()
        # Synthetic wheel exercises real ZIP/SHA logic; not an installable artifact.
        with zipfile.ZipFile(wheel,'w') as z:
            for name in ('deep_ep/_C.abi3.so','deep_ep_experimental/ep_gmm_fused/_C.abi3.so','libdeepep_kernels.so'):
                z.writestr(name,('not an ELF: '+name).encode())
            for name in ('deep_ep/__init__.py','deep_ep_experimental/__init__.py','deep_ep_experimental/ep_gmm_fused/__init__.py'):
                z.writestr(name,b'# CPU contract fixture only\n')
            for name,data in [('deep_ep/_build_info.json',{'shmem':{'sha256':{'lib/libshmem.so':sdksha}}}),
                              ('deep_ep_experimental/ep_gmm_fused/_build_info.json',{'fixture':True}),
                              ('deep_ep_experimental/ep_gmm_fused/upstream.json',{'fixture':True})]:z.writestr(name,json.dumps(data))
        wheel_info=dict(relative='wheels/fixture.whl',sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),bytes=wheel.stat().st_size)
        s16=copy.deepcopy(state16);s16['files']=[wheel_info];s8=copy.deepcopy(state8);s8['files']=[wheel_info]
        c16=make_contract('ep16_adapted',s16,wheel);c8=make_contract('ep8_original',s8,wheel)
        q.check_contract(c16,'ep16_adapted');q.check_contract(c8,'ep8_original');passed('real ZIP/native SHA extraction, variant contracts')
        reject('EP8 contract cannot qualify EP16',lambda:q.check_contract(c8,'ep16_adapted'))
        reject('EP16 contract cannot qualify EP8',lambda:q.check_contract(c16,'ep8_original'))
        wrong=copy.deepcopy(s16);wrong['patch_sha256']='0'*64;reject('unreviewed patch',lambda:make_contract('ep16_adapted',wrong,wheel))
        wrong=copy.deepcopy(s16);wrong['source_sha256'][next(iter(wrong['source_sha256']))]='0'*64;reject('changed coordinated source',lambda:make_contract('ep16_adapted',wrong,wheel))
        wrong=copy.deepcopy(s16);wrong['files'][0]['sha256']='0'*64;reject('wheel SHA mismatch',lambda:make_contract('ep16_adapted',wrong,wheel))
        limits={n:dict(atol=.003,rtol=.03,rms_relative=.02) for n in ('Y','dx','dp','dW1','dW2')}
        config=dict(variant='ep16_adapted',world=16,shape=dict(B=32,H=128,I=128,E=64,K=8),
            gemm_cores=32,owner='one_unique_id_udma',schedule='serial',recompute=False,seed=4317,
            heap_size=1<<30,oracle_chunk=1024,max_reference_gib=12.,limits=limits,weight_dtype='torch.bfloat16',
            shared=False,cases=['balanced'],actual_hardware=dict(aic_cores=32,ub_bytes=256*1024))
        q.check_config(config,True);passed('valid exact16 execution')
        for key,value in [('world',8),('owner','ElasticBuffer'),('schedule','overlap'),('heap_size',1),('seed',True),('actual_hardware',dict(aic_cores=64,ub_bytes=256*1024))]:
            reject('invalid config '+key,lambda k=key,v=value:q.check_config(dict(config,**{k:v}),True))
        basepath=WS/'work/DeepEP/experiments/megamoe_training/tests/provenance.py'
        spec=importlib.util.spec_from_file_location('frozen_base_identity',basepath);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
        actual=json.loads(next((WS/'outputs/training_results_20261025_0658').rglob('rank0.json')).read_text())['identity']
        rows=[dict(role=k,sha256=v['sha256'],bytes=v.get('bytes',len(b'fake SHMEM fixture only'))) for k,v in c16['content']['native'].items()]
        rows.append(dict(role='native.catlass',sha256='1'*64,bytes=10))
        for name in q.LOCAL:rows.append(dict(role='harness/'+name,sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),bytes=(ROOT/name).stat().st_size))
        for name in ('ep_gmm_training','ep_gmm_training_ep16'):
            for file in ('__init__.py','training_adapter.py'):
                p=WS/'work/training_candidates'/name/file;rows.append(dict(role='wrapper/'+name+'/'+file,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
        content=dict(execution=config,build_contract=c16,base_identity={k:actual[k] for k in ('schema','content','content_digest')},files=rows)
        identity=dict(schema='ep-gmm-training-1',content=content,content_digest=digest(content));q.validate_identity(identity,base.validate_identity);passed('synthetic full identity accepted')
        reject('P0 identity refused',lambda:q.validate_identity(actual,base.validate_identity))
        changed=copy.deepcopy(identity);changed['content']['files'][0]['sha256']='0'*64;changed['content_digest']=digest(changed['content'])
        reject('actual native SO SHA differs from wheel',lambda:q.validate_identity(changed,base.validate_identity))
        changed=copy.deepcopy(identity);changed['content']['files']=[v for v in rows if v['role']!='wrapper/ep_gmm_training_ep16/training_adapter.py'];changed['content_digest']=digest(changed['content'])
        reject('missing adapter source',lambda:q.validate_identity(changed,base.validate_identity))
        moved=copy.deepcopy(identity);moved['diagnostic_paths']={'foo':'/other-host/prefix'};q.require_identity_match(moved,identity,base.validate_identity);passed('host diagnostic path differences allowed')
    for scenario in ('success','adapter_close_failure','destroy_failure','remote_destroy_failure','partial_init','never_owned'):
        events=[];report={'all_passed':True}
        class Engine:
            def close(self):
                events.append('adapter.close')
                if scenario=='adapter_close_failure':raise RuntimeError('original adapter close error')
        class Runtime:
            def destroy_udma_group(self):
                events.append('UDMA.destroy')
                if scenario=='destroy_failure':raise RuntimeError('original UDMA destroy error')
        class Host:
            def rank(self):return 0
            def all_gather(self,item,stage):
                if scenario=='remote_destroy_failure' and item['stage']=='UDMA.destroy':return [item,dict(item,rank=1,error='remote destroy failed')]
                return [item]
        failed=False
        try:cleanup_owner(None if scenario in ('partial_init','never_owned') else Engine(),Runtime(),scenario!='never_owned',Host(),report)
        except RuntimeError:failed=True
        assert failed==(scenario in ('adapter_close_failure','destroy_failure','remote_destroy_failure'))
        if scenario=='never_owned':assert events==[]
        elif scenario=='partial_init':assert events==['UDMA.destroy']
        else:assert events==['adapter.close','UDMA.destroy']
        passed('ordered/exclusive cleanup '+scenario)
    # Actual local StoreGroup protocol: failure before/inside init never allows
    # constructor/next case; partial-owned cleanup is still attempted all-rank.
    tree=ast.parse((WS/'work/DeepEP/experiments/megamoe/tests/test_store.py').read_text(encoding='utf-8'))
    memory=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MemoryStore');scope={'threading':threading}
    exec(compile(ast.Module(body=[memory],type_ignores=[]),'existing_MemoryStore','exec'),scope)
    validator.phase=lambda *a,**k:None
    for scenario in ('config_failure','partial_init_failure','numerical_failure','remote_destroy_failure'):
        store=scope['MemoryStore']();groups=[StoreGroup(store,r,16,prefix=scenario,timeout=1) for r in range(16)]
        def worker(rank):
            host=groups[rank];events=[];report={'rank':rank,'stages':[],'all_passed':False};owned=False;engine=None;prior=None
            class Engine:
                def close(self):events.append('close')
            class Runtime:
                def destroy_udma_group(self):
                    events.append('destroy')
                    if scenario=='remote_destroy_failure' and rank==12:raise RuntimeError('original destroy error')
            def check():
                if scenario=='config_failure' and rank==7:raise ValueError('configuration failed before native init')
            def initialize():
                nonlocal owned
                owned=True;events.append('init')
                if scenario=='partial_init_failure' and rank==5:raise RuntimeError('partial init failed')
            def numbers():
                if scenario=='numerical_failure' and rank==3:raise AssertionError('fixed numerical gate failed')
            try:
                host_stage('config',check,host,report,lambda:None)
                host_stage('init',initialize,host,report,lambda:None)
                engine=Engine();events.append('construct')
                host_stage('numerical',numbers,host,report,lambda:None);report['all_passed']=True
            except Exception as exc:prior=exc
            try:cleanup_owner(engine,Runtime(),owned,host,report)
            except Exception as exc:prior=exc
            try:finish_host_report(host,report,types.SimpleNamespace(is_initialized=lambda:True,destroy_process_group=lambda:events.append('PG')),lambda:None,prior)
            except Exception:pass
            assert report['all_passed'] is False and host._store is None
            if scenario=='config_failure':assert events==['PG']
            elif scenario=='partial_init_failure':assert events==['init','destroy','PG']
            else:assert events==['init','construct','close','destroy','PG']
            return True
        with ThreadPoolExecutor(max_workers=16) as pool:assert all(pool.map(worker,range(16)))
        passed('16rank all-rank init/cleanup fail-closed '+scenario)
    expected=q.required_module_paths(WS/'work/DeepEP/experiments/megamoe_training',WS/'work/training_candidates','ep16_adapted')
    fake={}
    for name,path in expected.items():fake[name]=types.ModuleType(name);fake[name].__file__=str(path)
    with patch.dict(sys.modules,fake):q.require_module_paths(expected)
    passed('cached modules from all explicit expected paths accepted')
    for name in ('ep_gmm_training.training_adapter','ep_gmm_training_ep16.training_adapter',
                 'megamoe_training','megamoe_training.training','megamoe_training.gemm'):
        escaped=dict(fake);module=types.ModuleType(name);module.__file__='/wrong-cached-root/'+name.replace('.','/')+'.py';escaped[name]=module
        with patch.dict(sys.modules,escaped):reject('cached module root escape '+name,lambda:q.require_module_paths(expected))
    result=subprocess.run([sys.executable,'-c',"import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=['validate.py'];runpy.run_path(sys.path[0]+'/validate.py',run_name='__main__');assert 'torch' not in sys.modules",str(ROOT)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr;passed('default plan imports no Torch/native and needs no local wheel')
    report={'all_passed':True,'checks':checks,'scope':'Offline contract/identity/lifecycle only; synthetic wheel is not a real ELF or qualification',
            'source_unchanged':'No original EP8/EP16 adapter edit; no repeated math32/36 tests','files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
    (ROOT/'cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'all_passed':True,'checks':len(checks)}))


if __name__=='__main__':main()
