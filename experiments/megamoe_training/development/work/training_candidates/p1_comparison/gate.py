"""Stdlib-only, backend-specific full-gradient qualification gate."""
import hashlib
import json
import math
from pathlib import Path

FIXED_LIMIT={'atol':.003,'rtol':.03,'rms_relative':.02}
NAMES=('Y','dx','dp','dW1','dW2')


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
        balanced=[case for case in report.get('cases',[]) if case.get('case')=='balanced']
        if len(balanced)!=1:raise ValueError(f'Expected exactly one balanced case: {path}')
        case=balanced[0];lc=counts[rank*local:(rank+1)*local];recv=sum(lc)
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
