import pathlib,json
p=pathlib.Path('outputs/training_results_20261025_0814/p1-pointwise-unit/rack05os7/pointwise-results.json');r=json.loads(p.read_text());print('keys',list(r));print('before',json.dumps(r.get('loaded_before_tests'))[:2200]);print('checks',[(x.get('operation'),x.get('shape'),x.get('passed')) for x in r['checks'][-5:]])
