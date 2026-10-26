"""CPU audit for approved isolated full-training debug/timing/profile artifacts."""
import hashlib
import json
import math
from pathlib import Path
import sys

tag,stage,mode=sys.argv[1:4]
root=Path('outputs','training_results_'+tag);reports=[];traces=[]
missing=[rank for rank in range(16) if not (root/stage/('rack05os7' if rank<8 else 'rack05os8')/f'perf_rank{rank}.json').exists()]
if missing:
    print(json.dumps(dict(run_tag=tag,complete=False,status='awaiting_final_rank_reports',missing_ranks=missing)))
    raise SystemExit(0)
expected={'probe.py':'8cd2d92b7f1b085fe80218cec34d38df149aa353cbc912f236463ae95ad1cd2c',
          'gate.py':'4755516f347d78afdc5719e0c0000afa69d43e2222c8689616ca5f7a3c71fe8f'}
for rank in range(16):
    host='rack05os7' if rank<8 else 'rack05os8';base=root/stage/host
    r=json.loads((base/f'perf_rank{rank}.json').read_text());reports.append(r)
    assert r['rank']==rank and r['world']==16 and r['passed'] is True and r['qualified'] is True
    assert not r.get('error') and not r.get('cleanup_errors')
    assert r['mode']==mode and r['harness_sha256']==expected
    assert r['identity']['content_digest']=='15399c4d3624b1e6fd5f5789e81e80b4d4c62bc73d98fa3be9874fb1613c6015'
    execution=r['execution']
    assert execution['shape']==dict(B=8192,H=6144,I=2048,E=64,K=8)
    assert execution['seed']==4317 and execution['weight_dtype']=='torch.bfloat16' and execution['shared'] is False
    assert execution['backends']==['shmem_catlass','cann_ai_cpu'] and execution['route']=='balanced'
    assert r['environment']['TRAINING_SOURCE_ROOT']=='/home/zzy/deploy/megamoe-20261024/training/qualified-p0-20261025/experiments/megamoe_training'
    for backend,evidence in r['qualification'].items():
        assert len(evidence)==16 and {q['rank'] for q in evidence}==set(range(16))
        assert all(q['identity_digest']==r['identity']['content_digest'] for q in evidence)
    if mode=='debug':assert r['debug_steps']==1
    if mode=='profile':
        assert set(r['traces'])=={'shmem_catlass','cann_ai_cpu'}
        for name in r['traces']:
            p=base/f'{name}_rank{rank}_trace.json';data=p.read_bytes();trace=json.loads(data);events=trace['traceEvents'] if isinstance(trace,dict) else trace
            assert isinstance(events,list) and events
            windows=[e for e in events if e.get('ph')=='X' and e.get('name')=='training_perf_probe/'+name+'/forward_backward']
            assert len(windows)==2
            traces.append(dict(rank=rank,backend=name,file=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),event_count=len(events),step_windows=windows))
if mode=='benchmark':
    first=reports[0]['timing'];assert all(r['timing']==first for r in reports)
    assert first['iterations']==10 and first['warmup']==3
    assert set(first['paths'])=={'shmem_catlass','cann_ai_cpu'}
    for metrics in first['paths'].values():
        for metric,entry in metrics.items():
            matrix=entry['per_rank'];assert len(matrix)==16 and all(len(row)==10 for row in matrix)
            assert all(math.isfinite(v) and v>0 for row in matrix for v in row)
            assert all(abs(max(row[i] for row in matrix)-entry['max_rank'][i])<1e-4 for i in range(10))
node_rc={}
for host,node in [('rack05os7',0),('rack05os8',1)]:
    p=root/stage/host/'logs'/f'node{node}.rc';node_rc[host]=int(p.read_text()) if p.exists() else None
result=dict(run_tag=tag,stage=stage,mode=mode,all16_reports_passed=True,node_rc=node_rc,complete=all(v==0 for v in node_rc.values()),
            identity=reports[0]['identity']['content_digest'],harness=expected,execution=reports[0]['execution'],traces=traces)
if mode=='benchmark':result['timing']=reports[0]['timing']
(root/'perf-stage-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ('run_tag','mode','complete','all16_reports_passed','node_rc','identity')}))
