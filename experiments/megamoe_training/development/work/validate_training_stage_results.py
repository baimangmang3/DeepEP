"""CPU-only full-rank qualification audit; no torch import or device access."""
import json
from pathlib import Path
import sys

tag, stage = sys.argv[1:3]
controller = json.loads(Path('outputs', 'training_round_'+tag, stage+'-controller', 'summary.json').read_text())
argv = controller['script_arguments']
def option(key, default=None):
    return argv[argv.index(key)+1] if key in argv else default
shape = {key:int(option('--'+key)) for key in ('B','H','I','E','K')}
cases = option('--cases').split(',')
shared = '--shared' in argv
dtype = 'torch.'+{'bf16':'bfloat16','fp32':'float32'}[option('--weight-dtype','bf16')]
backend = option('--backend','shmem_catlass')
required = {'Y','dx','dp','dW1','dW2','leaf_dW1','leaf_dW2'}
if shared:
    required |= {'dSW1','dSW2','leaf_dSW1','leaf_dSW2'}
limit = dict(atol=.003, rtol=.03, rms_relative=.02)
metric_dtypes = {'Y':'torch.bfloat16','dx':'torch.bfloat16','dp':'torch.float32',
                 'dW1':'torch.float32','dW2':'torch.float32','leaf_dW1':dtype,'leaf_dW2':dtype,
                 'dSW1':'torch.float32','dSW2':'torch.float32','leaf_dSW1':dtype,'leaf_dSW2':dtype}
elements = {'Y':shape['B']*shape['H'],'dx':shape['B']*shape['H'],'dp':shape['B']*shape['K'],
            'dW1':(shape['E']//16)*2*shape['I']*shape['H'],
            'dW2':(shape['E']//16)*shape['H']*shape['I'],
            'dSW1':2*shape['I']*shape['H'],'dSW2':shape['H']*shape['I']}
elements.update({'leaf_'+key:value for key,value in list(elements.items()) if key.startswith('dW') or key.startswith('dSW')})
root = Path('outputs', 'training_results_'+tag)
result = dict(run_tag=tag, stage=stage, shape=shape, cases=cases, shared=shared,
              weight_dtype=dtype, backend=backend, node_rc={}, ranks=[], errors=[])
for host,node in [('rack05os7',0),('rack05os8',1)]:
    base = root/stage/host
    rc = base/'logs'/f'node{node}.rc'
    result['node_rc'][host] = int(rc.read_text()) if rc.is_file() else None
    for path in base.glob('rank*.json'):
        report = json.loads(path.read_text())
        errors = []
        for key,expected in dict(world=16,shape=shape,shared=shared,weight_dtype=dtype,backend=backend,all_passed=True).items():
            if report.get(key) != expected:
                errors.append('mismatched '+key)
        if report.get('error'):
            errors.append(report['error'])
        records = report.get('cases',[])
        if [x.get('case') for x in records] != cases:
            errors.append('missing/reordered cases')
        maxima = {}
        for case in records:
            if not (case.get('counts_match') is True and case.get('all_ranks_numerical_pass') is True):
                errors.append('case counts/global numerical gate: '+str(case.get('case')))
            metrics = case.get('metrics',{})
            if case.get('case')=='balanced' and shape==dict(B=8192,H=6144,I=2048,E=64,K=8):
                expected_shapes = dict(x=[8192,6144],xe=[65536,6144],w1=[4,4096,6144],
                                       gu=[65536,4096],a=[65536,2048],w2=[4,6144,2048],y=[8192,6144])
                expected_shapes['packed_z' if backend.startswith('cann_') else 'z']=[65536,6144]
                if case.get('candidate_counts') != [16384]*4 or case.get('oracle_counts') != [16384]*4:
                    errors.append('target65536 row counts mismatch')
                if any(case.get('candidate_shapes',{}).get(k)!=v for k,v in expected_shapes.items()):
                    errors.append('target65536 routed tensor shapes mismatch')
            if not required.issubset(metrics):
                errors.append('missing gradient metric: '+str(required-set(metrics)))
            for name,metric in metrics.items():
                if not (metric.get('passed') is True and metric.get('finite') is True and metric.get('limits') == limit):
                    errors.append('metric/threshold: '+str(case.get('case'))+'/'+name)
                if metric.get('actual_dtype') != metric_dtypes.get(name) or metric.get('elements') != elements.get(name):
                    errors.append('metric dtype/element count: '+str(case.get('case'))+'/'+name)
                maxima[name] = max(maxima.get(name,0.), metric.get('RMS_relative_error',float('inf')))
        result['ranks'].append(dict(rank=report.get('rank'), passed=not errors, errors=errors,
                                    identity=report.get('identity',{}).get('content_digest'), max_RMS_relative=maxima))
ranks = result['ranks']
result['identities'] = sorted({r['identity'] for r in ranks if r.get('identity')})
result['rank_set_complete'] = len(ranks)==16 and {r['rank'] for r in ranks}==set(range(16))
result['numerical_and_identity_passed'] = result['rank_set_complete'] and all(r['passed'] for r in ranks) and len(result['identities'])==1 and all(r['identity'] for r in ranks)
result['complete'] = result['numerical_and_identity_passed'] and all(rc==0 for rc in result['node_rc'].values())
(root/'stage-accuracy-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({key:result[key] for key in ('run_tag','complete','numerical_and_identity_passed','identities','node_rc')}))
