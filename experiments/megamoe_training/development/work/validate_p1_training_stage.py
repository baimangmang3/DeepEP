"""CPU-only strict audit of P1 full-gradient reports, including nested identity."""
import hashlib,json,math
from pathlib import Path
import sys

tag,stage=sys.argv[1:3];root=Path('outputs','training_results_'+tag)
control=json.loads(Path('outputs','training_round_'+tag,stage+'-controller','summary.json').read_text());args=control['script_arguments']
def option(key,default=None):return args[args.index(key)+1] if key in args else default
world=int(option('--world-size'));shape={k:int(option('--'+k)) for k in ('B','H','I','E','K')};shared='--shared' in args
dtype='torch.'+{'bf16':'bfloat16','fp32':'float32'}[option('--weight-dtype')];cases=option('--cases').split(',')
manifest=json.loads(Path('outputs/p1_training_deployment/manifest.json').read_text())['sources_sha256'];expected={'source/'+p:s for p,s in manifest.items()};expected['native.pointwise']='e189f39a8448df078e082ba90b3147e593962bbcc374aa388d69344cfc125ca1'
limit=dict(atol=.003,rtol=.03,rms_relative=.02);reports=[];ranks=[];identities=set()
for rank in range(world):
    host='rack05os7' if rank<8 or world==2 else 'rack05os8';p=root/stage/host/f'rank{rank}.json'
    if not p.exists():print(json.dumps(dict(complete=False,pending_rank=rank)));raise SystemExit(0)
    r=json.loads(p.read_text());reports.append(r)
    assert r['rank']==rank and r['world']==world and r['all_passed'] is True and not r.get('error') and not r.get('cleanup_error')
    assert r['backend']=='shmem_catlass_pointwise_p1' and r['shape']==shape and r['shared']==shared and r['weight_dtype']==dtype and r['seed']==4317
    identity=r['identity'];content=identity['content'];assert identity['schema']=='p1-training-1'
    assert hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==identity['content_digest']
    assert content['base_identity']['content_digest']=='15399c4d3624b1e6fd5f5789e81e80b4d4c62bc73d98fa3be9874fb1613c6015'
    assert {e['role']:e['sha256'] for e in content['candidate_files']}==expected and len(content['candidate_files'])==9
    execution=content['execution'];assert execution==r['execution']
    for key,value in dict(candidate='P1_native_pointwise_v1',backend='shmem_catlass_pointwise_p1',vector_cores=64,gemm_cores=32,schedule='serial',recompute=False,world=world,shape=shape,seed=4317,weight_dtype=dtype,shared=shared,cases=cases,oracle_chunk=1024,max_reference_gib=12.).items():assert execution[key]==value,key
    assert [c['case'] for c in r['cases']]==cases
    names={'Y','dx','dp','dW1','dW2','leaf_dW1','leaf_dW2'}|({'dSW1','dSW2','leaf_dSW1','leaf_dSW2'} if shared else set());maxima={}
    for case in r['cases']:
        assert all(case[k] is True for k in ('counts_match','shapes_match','dtypes_match','all_ranks_numerical_pass'))
        assert set(case['metrics'])==names
        for name,metric in case['metrics'].items():
            assert metric['passed'] is True and metric['finite'] is True and metric['out_of_tolerance_elements']==0 and metric['limits']==limit
            assert all(math.isfinite(metric[k]) and metric[k]>=0 for k in ('RMS_error','reference_RMS','RMS_relative_error','max_abs_error'))
            assert metric['RMS_relative_error']<=.02 if metric['reference_RMS'] else metric['RMS_error']<=.003
            expected_dtype=dtype if name.startswith('leaf_') else ('torch.bfloat16' if name in ('Y','dx') else 'torch.float32');assert metric['actual_dtype']==expected_dtype
            maxima[name]=max(maxima.get(name,0),metric['RMS_relative_error'])
        if case['case']=='balanced' and shape==dict(B=8192,H=6144,I=2048,E=64,K=8):assert case['candidate_counts']==[16384]*4 and case['candidate_shapes']['xe']==[65536,6144]
    identities.add(identity['content_digest']);ranks.append(dict(rank=rank,max_RMS_relative=maxima))
assert len(identities)==1
rc={}
for host,node in ([('rack05os7',0)] if world==2 else [('rack05os7',0),('rack05os8',1)]):
    p=root/stage/host/'logs'/f'node{node}.rc';rc[host]=int(p.read_text()) if p.exists() else None
result=dict(run_tag=tag,stage=stage,world=world,shape=shape,shared=shared,weight_dtype=dtype,cases=cases,all_rank_accuracy_passed=True,identity=next(iter(identities)),base_identity=content['base_identity']['content_digest'],source_manifest=manifest,native_pointwise_sha256=expected['native.pointwise'],node_rc=rc,complete=all(v==0 for v in rc.values()),ranks=ranks)
(root/'p1-stage-accuracy-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('run_tag','complete','world','identity','node_rc')}))
