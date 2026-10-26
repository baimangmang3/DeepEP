"""Real P0 report acceptance plus synthetic P1 negative gates; stdlib only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from gate import qualify

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[2]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    base=load('base_cpu',WORKSPACE/'work/DeepEP/experiments/megamoe_training/tests/provenance.py')
    candidate=WORKSPACE/'work/training_candidates/p1_pointwise'
    cp=load('candidate_cpu',candidate/'candidate_provenance.py')
    report={'scope':'CPU real P0 qualification acceptance and synthetic P1 gates, not NPU qualification','checks':[]}
    with tempfile.TemporaryDirectory(dir=ROOT) as td:
        tmp=Path(td);p0=tmp/'p0';p0.mkdir()
        recovered=WORKSPACE/'outputs/training_results_20261025_0658/ep16-user-shape-bf16'
        for path in recovered.rglob('rank*.json'):
            rank=json.loads(path.read_text())['rank'];shutil.copyfile(path,p0/f'rank{rank}.json')
        seed=json.loads((p0/'rank0.json').read_text());limits=seed['limits'];shape=seed['shape']
        assert len(qualify(p0,'shmem_catlass',16,shape,seed['seed'],False,seed['identity'],limits,base.validate_identity))==16
        report['checks'].append({'name':'real 0658 P0 all16 accepted, same qualification used0742','passed':True})
        for tag,source in [('0703 P0 shared','outputs/training_results_20261025_0703/ep16-user-shape-shared-bf16'),
                           ('0714 CANN routed','outputs/training_results_20261025_0714/ep16-cann-ai-cpu-user-shape')]:
            target=tmp/tag;target.mkdir()
            for path in (WORKSPACE/source).rglob('rank*.json'):
                record=json.loads(path.read_text());shutil.copyfile(path,target/f"rank{record['rank']}.json")
            record=json.loads((target/'rank0.json').read_text())
            assert len(qualify(target,record['backend'],16,record['shape'],record['seed'],record['shared'],record['identity'],record['limits'],base.validate_identity))==16
            report['checks'].append({'name':'real '+tag+' all16 legacy accepted','passed':True})
        (tmp/'fake.so').write_bytes(b'CPU synthetic only, not ELF or runtime library')
        files={'source/'+name:candidate/name for name in cp.SOURCES};files['native.pointwise']=tmp/'fake.so'
        files['source/validate_training_p1.py']=WORKSPACE/'work/p1_case_lifecycle_review/validate_training_p1_host_candidate.py'
        execution=dict(cp.FIXED,world=16,shape=shape,seed=seed['seed'],weight_dtype='torch.bfloat16',shared=False,
            cases=['balanced'],oracle_chunk=1024,max_reference_gib=12.,limits=limits)
        identity=cp.build_identity(seed['identity'],files,execution,base)
        directory=tmp/'p1';directory.mkdir()
        for rank in range(16):
            record=json.loads((p0/f'rank{rank}.json').read_text());record.update(backend='shmem_catlass_pointwise_p1',identity=identity,execution=execution)
            peers=[dict(rank=peer,case='balanced',case_index=0,stage='P1/case-close/0/balanced',close_error=None,body_error=None) for peer in range(16)]
            record['case_cleanup']=[dict(peers[rank],peer_status=peers,all_ranks_closed=True,all_ranks_case_passed=True)]
            record['peer_before_PG_teardown']=[dict(rank=peer,error=None) for peer in range(16)]
            record['peer_exit_status']=[dict(rank=peer,errors=[]) for peer in range(16)]
            record['cases'][0].update(shapes_match=True,dtypes_match=True)
            (directory/f'rank{rank}.json').write_text(json.dumps(record))
        validate=lambda value:cp.validate_identity(value,base)
        def gate():return qualify(directory,'shmem_catlass_pointwise_p1',16,shape,seed['seed'],False,identity,limits,validate)
        assert len(gate())==16
        report['checks'].append({'name':'synthetic P1 schema/current config all16 accepted','passed':True})
        target=directory/'rank15.json';original=json.loads(target.read_text())
        def negative(name,mutate):
            record=copy.deepcopy(original);mutate(record);target.write_text(json.dumps(record))
            try:gate()
            except (ValueError,TypeError,KeyError):report['checks'].append({'name':name,'passed':True})
            else:raise AssertionError('Accepted invalid '+name)
            finally:target.write_text(json.dumps(original))
        negative('missing P1 cleanup',lambda r:r.pop('case_cleanup'))
        negative('failed peer close',lambda r:r['case_cleanup'][0]['peer_status'][12].update(close_error='synthetic original finalize error'))
        negative('missing peer close',lambda r:r['case_cleanup'][0]['peer_status'].pop())
        negative('missing host pre-exit',lambda r:r.pop('peer_before_PG_teardown'))
        negative('remote prior error',lambda r:r['peer_before_PG_teardown'][12].update(error='original peer error'))
        negative('remote PG error',lambda r:r['peer_exit_status'][12].update(errors=['PG failed']))
        negative('missing host exit',lambda r:r.pop('peer_exit_status'))
        negative('P1 shapes gate false',lambda r:r['cases'][0].update(shapes_match=False))
        negative('P1 dtype gate false',lambda r:r['cases'][0].update(dtypes_match=False))
        negative('different execution config',lambda r:r['execution'].update(shared=True))
        negative('old P0 identity',lambda r:r.update(identity=seed['identity']))
        negative('wrong backend',lambda r:r.update(backend='shmem_catlass'))
        negative('wrong shared',lambda r:r.update(shared=True))
        negative('wrong seed',lambda r:r.update(seed=0))
        negative('wrong world',lambda r:r.update(world=8))
        negative('wrong recv shape',lambda r:r['cases'][0]['candidate_shapes'].update(xe=[1,shape['H']]))
        negative('wrong counts',lambda r:r['cases'][0]['candidate_counts'].__setitem__(0,0))
        negative('missing gradient',lambda r:r['cases'][0]['metrics'].pop('dp'))
        negative('wrong gradient dtype',lambda r:r['cases'][0]['metrics']['dW1'].update(actual_dtype='torch.bfloat16'))
        negative('sampled element count',lambda r:r['cases'][0]['metrics']['dx'].update(elements=1))
        negative('nonfinite gradient',lambda r:r['cases'][0]['metrics']['dp'].update(finite=False))
        negative('nonzero badcount',lambda r:r['cases'][0]['metrics']['dp'].update(out_of_tolerance_elements=1))
        negative('relaxed threshold',lambda r:r['cases'][0]['metrics']['dp']['limits'].update(atol=1.))
        negative('contradictory RMS',lambda r:r['cases'][0]['metrics']['dp'].update(RMS_relative_error=1.))
        changed=copy.deepcopy(identity);changed['content']['execution']['shared']=True;changed['content_digest']=cp.digest(changed['content'])
        negative('serial config identity changed shared',lambda r:r.update(identity=changed))
        changed=copy.deepcopy(identity);changed['content']['candidate_files'][0]['sha256']='0'*64;changed['content_digest']=cp.digest(changed['content'])
        negative('candidate source identity changed',lambda r:r.update(identity=changed))
        report['identity_digest']=identity['content_digest']
    report.update(passed=True,probe_sha256=hashlib.sha256((ROOT/'probe.py').read_bytes()).hexdigest(),gate_sha256=hashlib.sha256((ROOT/'gate.py').read_bytes()).hexdigest(),
        static_review={'single_owner':'Exactly one chosen backend engine; identity collectors load only libraries, no SHMEM session',
        'execution':'P1 collector config matches main balanced BF16 validator: fixed vector64/gemm32/serial/no-recompute, exact shape/seed/shared, chunk1024/cap12 and frozen LIMITS',
        'math':'Frozen inputs/shared generator/clear_grads and timing reused; full backward, autograd required; no reference computed in timing',
        'scope':'Benchmark and profile separate, alignment outside profile label; cleanup forces false and raises; ABBA is caller-orchestrated separate processes'})
    (ROOT/'cpu_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':len(report['checks']),'probe_sha256':report['probe_sha256'],'gate_sha256':report['gate_sha256']}))


if __name__=='__main__':main()
