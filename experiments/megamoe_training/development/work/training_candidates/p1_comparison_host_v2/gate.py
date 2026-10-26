"""Stdlib-only, backend-specific full-gradient qualification gate."""
import hashlib
import json
import math
from pathlib import Path

FIXED_LIMIT={'atol':.003,'rtol':.03,'rms_relative':.02}
NAMES=('Y','dx','dp','dW1','dW2')


def require_p1_host_cleanup(report,world,rank,current_identity):
    """Only new P1 source/config identity plus complete successful host exit."""
    content=current_identity.get('content',{})
    if current_identity.get('schema')!='p1-training-1':raise ValueError('P1 expanded identity required')
    files={item['role']:item for item in content.get('candidate_files',[])}
    if 'source/validate_training_p1.py' not in files:raise ValueError('Current P1 validator identity missing')
    execution=content.get('execution',{})
    if report.get('execution')!=execution:raise ValueError('P1 qualification execution differs from current root/config')
    cases=execution.get('cases',[]);cleanup=report.get('case_cleanup')
    if not isinstance(cleanup,list) or len(cleanup)!=len(cases):raise ValueError('P1 complete case cleanup required')
    for index,(name,item) in enumerate(zip(cases,cleanup)):
        stage=f'P1/case-close/{index}/{name}'
        if (item.get('rank')!=rank or item.get('case')!=name or item.get('case_index')!=index or item.get('stage')!=stage
                or item.get('close_error','missing') is not None or item.get('body_error','missing') is not None
                or item.get('all_ranks_closed') is not True or item.get('all_ranks_case_passed') is not True
                or item.get('host_exchange_error') is not None):raise ValueError('P1 case cleanup failed/incomplete')
        peers=item.get('peer_status',[])
        if (len(peers)!=world or [peer.get('rank') for peer in peers]!=list(range(world))
                or any(peer.get('case')!=name or peer.get('case_index')!=index or peer.get('stage')!=stage
                       or peer.get('close_error','missing') is not None or peer.get('body_error','missing') is not None for peer in peers)):
            raise ValueError('P1 all-rank case close status incomplete')
    before=report.get('peer_before_PG_teardown',[]);after=report.get('peer_exit_status',[])
    if (len(before)!=world or [peer.get('rank') for peer in before]!=list(range(world))
            or any(peer.get('error','missing') is not None for peer in before)
            or len(after)!=world or [peer.get('rank') for peer in after]!=list(range(world))
            or any(peer.get('errors')!=[] for peer in after)
            or report.get('cleanup_errors') not in (None,[]) or report.get('report_io_error') is not None):
        raise ValueError('P1 successful all-rank host/PG exit required')


def expected_counts(shape,world):
    b,e,k=shape['B'],shape['E'],shape['K']
    counts=[0]*e
    if e%k:
        q,r=divmod(world*b*k,e)
        return [q+(j<r) for j in range(e)]
    q,r=divmod(world*b,e)
    for residue in range(e):
        for slot in range(k):counts[(residue+slot*(e//k))%e]+=q+(residue<r)
    return counts


def qualify(directory,backend,world,shape,seed,shared,identity,limits,validate_identity):
    if backend not in ('shmem_catlass','shmem_catlass_pointwise_p1','cann_ai_cpu') or (shared and backend=='cann_ai_cpu'):
        raise ValueError('Unsupported backend/shared combination')
    if limits!={name:FIXED_LIMIT for name in NAMES}:
        raise ValueError('Frozen precision LIMITS changed; this harness does not loosen thresholds')
    digest=validate_identity(identity)
    counts=expected_counts(shape,world);local=shape['E']//world
    b,h,i,k=shape['B'],shape['H'],shape['I'],shape['K']
    elements={'Y':b*h,'dx':b*h,'dp':b*k,'dW1':local*2*i*h,'dW2':local*h*i,
              'leaf_dW1':local*2*i*h,'leaf_dW2':local*h*i}
    if shared:elements.update(dSW1=2*i*h,dSW2=h*i,leaf_dSW1=2*i*h,leaf_dSW2=h*i)
    evidence=[]
    for rank in range(world):
        path=Path(directory)/f'rank{rank}.json'
        raw=path.read_bytes();report=json.loads(raw)
        expected={'rank':rank,'world':world,'shape':shape,'seed':seed,'shared':shared,
                  'backend':backend,'weight_dtype':'torch.bfloat16','all_passed':True,'limits':limits}
        if any(key not in report or report[key]!=value for key,value in expected.items()):
            raise ValueError(f'{backend} qualification contract mismatch: {path}')
        if validate_identity(report.get('identity'))!=digest:
            raise ValueError(f'{backend} stale implementation/runtime identity: {path}')
        if backend=='shmem_catlass_pointwise_p1':
            require_p1_host_cleanup(report,world,rank,identity)
        balanced=[case for case in report.get('cases',[]) if case.get('case')=='balanced']
        if len(balanced)!=1:raise ValueError(f'Expected exactly one balanced case: {path}')
        case=balanced[0];lc=counts[rank*local:(rank+1)*local];recv=sum(lc)
        if backend=='shmem_catlass_pointwise_p1' and (case.get('shapes_match') is not True or case.get('dtypes_match') is not True):
            raise ValueError('P1 shapes/dtypes gate missing or failed')
        shapes={'x':[b,h],'xe':[recv,h],'w1':[local,2*i,h],'gu':[recv,2*i],
                'a':[recv,i],'w2':[local,h,i],'y':[b,h]}
        shapes['z' if backend in ('shmem_catlass','shmem_catlass_pointwise_p1') else 'packed_z']=[recv,h] if backend in ('shmem_catlass','shmem_catlass_pointwise_p1') else [b*k,h]
        if (case.get('all_ranks_numerical_pass') is not True or case.get('counts_match') is not True or
                case.get('oracle_counts')!=lc or case.get('candidate_counts')!=lc or
                any(case.get('candidate_shapes',{}).get(key)!=value for key,value in shapes.items())):
            raise ValueError(f'{backend} balanced counts/shapes/pass mismatch: {path}')
        for name,size in elements.items():
            metric=case.get('metrics',{}).get(name,{})
            dtype='torch.float32' if name in ('dp','dW1','dW2','dSW1','dSW2') else 'torch.bfloat16'
            if (metric.get('passed') is not True or metric.get('finite') is not True or
                    metric.get('elements')!=size or metric.get('limits')!=FIXED_LIMIT or
                    metric.get('actual_dtype')!=dtype or metric.get('out_of_tolerance_elements')!=0):
                raise ValueError(f'{backend} missing/failed/incomplete full gradient {name}: {path}')
            stats=[metric.get(key) for key in ('RMS_error','reference_RMS','RMS_relative_error','max_abs_error','max_reference_abs')]
            if any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in stats):
                raise ValueError(f'{backend} invalid numerical summary {name}: {path}')
            rms,reference,rms_rel,maximum,ref_max=stats
            if ((rms_rel>FIXED_LIMIT['rms_relative'] if reference else rms>FIXED_LIMIT['atol']) or
                    maximum>FIXED_LIMIT['atol']+FIXED_LIMIT['rtol']*ref_max):
                raise ValueError(f'{backend} numerical summary contradicts passed flag {name}: {path}')
        evidence.append({'rank':rank,'file':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),
                         'identity_digest':digest})
    return evidence
