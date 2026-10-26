"""Stdlib actual constructor/runtime-guard tests. No Torch or native import."""
import ast
import hashlib
import json
from pathlib import Path
import types

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'ep_gmm_training/training_adapter.py'


def main():
    state={'world':16,'rank':0,'config':None,'gemm_calls':0}
    def config():
        if state['config'] is None:raise RuntimeError('UDMA not initialized')
        return state['config']
    def new_config(**kw):
        return types.SimpleNamespace(**dict(dict(rank=0,world_size=16,mode='unique_id',members=tuple(range(16)),device='npu:0'),**kw))
    state['config']=new_config()
    dist=types.SimpleNamespace(get_world_size=lambda group:state['world'],get_rank=lambda group:state['rank'],
                               get_process_group_ranks=lambda group:list(range(state['world'])))
    def gemm(*args):state['gemm_calls']+=1;return object()
    torch=types.SimpleNamespace(npu=types.SimpleNamespace(current_device=lambda:0,synchronize=lambda *a:None),device=lambda d:d)
    scope=dict(torch=torch,dist=dist,get_udma_config=config,CatlassGemm=gemm)
    parent_tree=ast.parse(PARENT.read_text());parent=next(n for n in parent_tree.body if isinstance(n,ast.ClassDef) and n.name=='EpGmmTrainingMoE')
    exec(compile(ast.Module(body=[parent],type_ignores=[]),str(PARENT),'exec'),scope)
    source=ROOT/'training_adapter.py';tree=ast.parse(source.read_text());body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    require=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name=='require_ep16_metadata')
    expected=ast.literal_eval(next(n.value for n in require.body if isinstance(n,ast.Assign)))
    scope['ops']=types.SimpleNamespace(**expected);exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),scope)
    cls=scope['Ep16GmmTrainingMoE'];base=scope['EpGmmTrainingMoE'];checks=[]
    def accept(name,**kwargs):
        args=dict(group=object(),max_tokens=32,hidden=128,topk=8,num_experts=64,cores=32);args.update(kwargs)
        obj=cls(**args);checks.append(dict(name=name,passed=True));return obj
    def reject(name,action):
        before=state['gemm_calls']
        try:action()
        except (ValueError,RuntimeError):pass
        else:raise AssertionError(name+' accepted')
        assert state['gemm_calls']==before,'Invalid construction must not create GEMM loader'
        checks.append(dict(name=name,passed=True))
    owner=accept('target configuration')
    accept('last source/local-expert/topk boundary',max_tokens=16384,num_experts=512,topk=16,hidden=6144)
    accept('minimum configuration',max_tokens=1,num_experts=16,topk=1)
    for key,value in [('max_tokens',0),('max_tokens',16385),('hidden',0),('hidden',129),('topk',0),('topk',17),('num_experts',63),('num_experts',528),('cores',31),('cores',64)]:
        kwargs={key:value};reject('invalid '+key+'='+str(value),lambda kw=kwargs:cls(**dict(dict(group=None,max_tokens=32,hidden=128,topk=8,num_experts=64),**kw)))
    for world in (8,17):
        state['world']=world;reject('world'+str(world),lambda:cls(group=None,max_tokens=32,hidden=128,topk=8,num_experts=64));state['world']=16
    for key,value in expected.items():
        bad=dict(expected);bad[key]=value+1;scope['ops']=types.SimpleNamespace(**bad)
        reject('metadata '+key,lambda:cls(group=None,max_tokens=32,hidden=128,topk=8,num_experts=64))
    scope['ops']=types.SimpleNamespace(**expected)
    reject('missing metadata fields',lambda:scope['require_ep16_metadata'](types.SimpleNamespace()))
    for key,value in [('rank',1),('world_size',8),('mode','default'),('members',tuple(range(15))),('device','npu:1')]:
        state['config']=new_config(**{key:value})
        reject('runtime '+key,lambda:cls(group=None,max_tokens=32,hidden=128,topk=8,num_experts=64))
    state['config']=None;reject('missing caller UDMA',lambda:cls(group=None,max_tokens=32,hidden=128,topk=8,num_experts=64))
    # Existing owner must reject a different, even value-equal, domain instance.
    state['config']=new_config()
    try:owner._check_runtime()
    except RuntimeError:checks.append(dict(name='old owner rejects replacement UDMA config identity',passed=True))
    else:raise AssertionError('owner replacement accepted')
    state['config']=owner._config;owner.close();owner.close()
    assert owner.closed and state['config'] is owner._config
    checks.append(dict(name='inherited close retains caller UDMA and is idempotent',passed=True))
    inherited={name:getattr(cls,name) is getattr(base,name) for name in ('_validate','_check_runtime','__call__','close')}
    assert all(inherited.values())
    original8=ast.unparse(parent);assert 'self.world_size <= 8' in original8
    # Record contract limits honestly; this tests configured cores, not silicon.
    output=dict(scope='CPU constructor/metadata/runtime guards only; no native ABI, device AIC-count or EP16 precision proof',
        checks=checks,all_passed=True,inherited_same_function_objects=inherited,
        module_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'__init__.py',source)},
        parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),expected_metadata=expected,
        limitations=['cores==32 checks configured CATLASS core argument; actual hardware AIC count is not queried by this wrapper.',
                     'Python metadata values do not prove loaded native header/layout matches; full coordinated11-file build and mapped library identity required.',
                     'The equality-only metadata guard accepts numeric values such as16.0 equal to16; production constants are source-bound integers. Constructor also inherits permissive numeric typing from EP8. No algorithm correction applied.'])
    (ROOT/'tests/cpu_constructor_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'all_passed':True,'checks':len(checks),'inherited':inherited,'module_sha256':output['module_sha256']}))


if __name__=='__main__':main()
