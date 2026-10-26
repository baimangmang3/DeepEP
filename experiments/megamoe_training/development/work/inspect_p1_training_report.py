import json,pathlib
r=json.loads(pathlib.Path('outputs/training_results_20261025_0820/p1-ep2-bf16/rack05os7/rank0.json').read_text());print('keys',list(r));print('identitykeys',list(r['identity']));print('contentkeys',list(r['identity']['content']));print('execution',r.get('execution'));print('casekeys',list(r['cases'][0]));print('base?',json.dumps(r['identity']['content'])[:1800])
