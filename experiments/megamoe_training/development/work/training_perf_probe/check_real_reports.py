"""CPU-only acceptance of recovered real reports; no Torch/NPU import."""
import ast,hashlib,importlib.util,json,shutil,tempfile
from pathlib import Path
from gate import qualify

BASE=Path('work/training_perf_probe')
FROZEN=Path('work/DeepEP/experiments/megamoe_training')
SCENARIOS={
    '0658_P0_routed':Path('outputs/training_results_20261025_0658/ep16-user-shape-bf16'),
    '0703_P0_shared':Path('outputs/training_results_20261025_0703/ep16-user-shape-shared-bf16'),
    '0714_CANN_routed':Path('outputs/training_results_20261025_0714/ep16-cann-ai-cpu-user-shape'),
}


def main():
    helper=FROZEN/'tests/provenance.py'
    spec=importlib.util.spec_from_file_location('frozen_provenance_for_CPU_gate',helper)
    prov=importlib.util.module_from_spec(spec);spec.loader.exec_module(prov)
    tree=ast.parse((FROZEN/'tests/validate_training_npu.py').read_text())
    limits=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='LIMITS' for t in n.targets)))
    outcome={'scope':'CPU gate acceptance of recovered real numerical reports, not a new NPU run',
             'passed':False,'positive':[],'negative':[]}
    with tempfile.TemporaryDirectory(dir=BASE) as tmp:
        directories={};rank0={}
        for tag,source in SCENARIOS.items():
            paths=list(source.rglob('rank*.json'));by_rank={}
            for path in paths:
                r=json.loads(path.read_text());rank=r['rank']
                if rank in by_rank:raise ValueError('Duplicate recovered rank '+str(rank))
                by_rank[rank]=path
            assert set(by_rank)==set(range(16)),(tag,sorted(by_rank))
            target=Path(tmp)/tag;target.mkdir()
            for rank,path in by_rank.items():shutil.copyfile(path,target/f'rank{rank}.json')
            r=json.loads((target/'rank0.json').read_text());rank0[tag]=r;directories[tag]=target
            identity=prov.validate_identity(r['identity'])
            roles={item['role']:item for item in r['identity']['content']['files']}
            # Prove the CPU validator used here is the frozen qualified helper.
            assert hashlib.sha256(helper.read_bytes()).hexdigest()==roles['source.provenance']['sha256']
            assert hashlib.sha256((FROZEN/'tests/validate_training_npu.py').read_bytes()).hexdigest()==roles['source.validator_oracle_limits']['sha256']
            evidence=qualify(target,r['backend'],16,r['shape'],r['seed'],r['shared'],
                             r['identity'],limits,prov.validate_identity)
            outcome['positive'].append({'tag':tag,'backend':r['backend'],'shared':r['shared'],
                'ranks_passed':len(evidence),'identity_digest':identity,'passed':True})
        for source_tag,requested_tag in [('0703_P0_shared','0658_P0_routed'),
                                         ('0658_P0_routed','0703_P0_shared'),
                                         ('0714_CANN_routed','0658_P0_routed'),
                                         ('0703_P0_shared','0714_CANN_routed')]:
            r=rank0[requested_tag]
            try:
                qualify(directories[source_tag],r['backend'],16,r['shape'],r['seed'],r['shared'],
                        r['identity'],limits,prov.validate_identity)
            except ValueError as exc:
                outcome['negative'].append({'source_reports':source_tag,'requested_contract':requested_tag,
                                            'rejected':True,'error':str(exc).split(':')[0]})
            else:raise AssertionError('Swapped qualifications accepted')
    outcome['frozen_helper_sha256']=hashlib.sha256(helper.read_bytes()).hexdigest()
    outcome['passed']=True
    (BASE/'real_report_results.json').write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':main()
