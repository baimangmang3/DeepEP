"""Content-bound qualification identity; importable with the stdlib alone.

Absolute paths are diagnostic only. The digest covers stable logical roles,
content SHA256/size, framework versions and read-back precision settings.
Runtime collection loads CATLASS/DeepEP native libraries, not ElasticBuffer or
a new communication domain. Missing/ambiguous loaded-library evidence fails.
"""
import hashlib
import json
import os
from pathlib import Path
import re

_EXPLICIT_LIBRARY_HANDLES=[]  # Keep verified formal CANN loads alive.

SCHEMA=1
SOURCE_FILES={
    'source.training_init':'megamoe_training/__init__.py',
    'source.training':'megamoe_training/training.py',
    'source.gemm':'megamoe_training/gemm.py',
    'source.native_gemm':'native/gemm.cpp',
    'source.cmake':'CMakeLists.txt',
    'source.validator_oracle_limits':'tests/validate_training_npu.py',
    'source.cann_forward_oracle':'tests/cann_baseline.py',
    'source.cann_training':'tests/cann_training.py',
    'source.comparison':'tests/benchmark_compare.py',
    'source.provenance':'tests/provenance.py',
}
REQUIRED_ROLES=set(SOURCE_FILES)|{
    'native.catlass','native.deepep_core','native.shmem',
    'native.cann_acl','native.cann_runtime','native.cann_opapi',
    'deepep_python/__init__.py','deepep_python/_native.py',
    'deepep_python/_runtime.py','deepep_python/buffers/elastic.py',
}


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(content):
    data=json.dumps(content,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def validate_identity(identity):
    if not isinstance(identity,dict) or identity.get('schema')!=SCHEMA:
        raise ValueError('Missing or obsolete content identity')
    content=identity.get('content')
    if not isinstance(content,dict) or identity.get('content_digest')!=canonical_digest(content):
        raise ValueError('Identity digest does not match its content manifest')
    files=content.get('files')
    if not isinstance(files,list):
        raise ValueError('Missing identity files')
    roles=set()
    for item in files:
        if (not isinstance(item,dict) or set(item)!= {'role','sha256','bytes'} or not isinstance(item['role'],str) or
                not re.fullmatch(r'[0-9a-f]{64}',str(item['sha256'])) or
                not isinstance(item['bytes'],int) or item['bytes']<=0 or item['role'] in roles):
            raise ValueError('Invalid or duplicate identity file role')
        roles.add(item['role'])
    if REQUIRED_ROLES-roles:
        raise ValueError('Missing required identity roles: '+repr(sorted(REQUIRED_ROLES-roles)))
    versions=content.get('versions',{})
    if set(versions)!= {'torch','torch_npu'} or any(not isinstance(v,str) or not v or v in ('None','unknown') for v in versions.values()):
        raise ValueError('Missing framework version identity')
    precision=content.get('precision',{})
    if (precision!= {'matmul_allow_hf32':False,'allow_internal_format':False} or
            any(type(value) is not bool for value in precision.values())):
        raise ValueError('Precision settings missing or differ from strict BF16/FP32 oracle contract')
    return identity['content_digest']


def build_identity(files,versions,precision):
    manifest=[];paths={}
    for role,path in sorted(files.items()):
        path=Path(path).resolve(strict=True)
        if not path.is_file() or path.stat().st_size<=0:
            raise ValueError(f'Identity role {role} is not a nonempty file: {path}')
        manifest.append({'role':role,'sha256':sha256_file(path),'bytes':path.stat().st_size})
        paths[role]=str(path)
    content={'files':manifest,'versions':versions,'precision':precision}
    result={'schema':SCHEMA,'content':content,'content_digest':canonical_digest(content),'diagnostic_paths':paths}
    validate_identity(result)
    return result


def mapped_libraries(maps_path=Path('/proc/self/maps')):
    """Actual mappings, never SDK install guesses or LD_LIBRARY_PATH guesses."""
    if not maps_path.is_file():
        raise ValueError('/proc/self/maps unavailable; cannot prove loaded native identity')
    paths=set()
    for line in maps_path.read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)<6 or not fields[5].startswith('/'):
            continue
        raw=fields[5].replace('\\040',' ')
        if '.so' not in Path(raw).name:
            continue
        if raw.endswith(' (deleted)'):
            raise ValueError('Deleted loaded shared object prevents reproducible identity: '+raw)
        paths.add(Path(raw).resolve(strict=True))
    return paths


def unique_loaded(paths,predicate,label):
    selected=sorted((p for p in paths if predicate(p.name)),key=str)
    if not selected:
        raise ValueError('Required actual loaded library missing: '+label)
    hashes={sha256_file(p) for p in selected}
    if len(hashes)!=1:
        raise ValueError('Ambiguous multiple loaded contents for '+label)
    return selected[0]


def ensure_required_cann_mapped(mapped):
    """Explicit formal-install load for lazy primary CANN libraries.

    ASCEND_HOME_PATH/lib64 is the deployed, audited official location. Never
    search other installations or accept a missing native mapping. ctypes loads
    the library only; it does not initialize a device or execute an operator.
    """
    import ctypes
    loads=[]
    for soname in ('libascendcl.so','libruntime.so','libopapi.so'):
        if any(path.name.startswith(soname) for path in mapped):
            continue
        configured=os.environ.get('ASCEND_HOME_PATH')
        if not configured:
            raise ValueError('Lazy '+soname+' is unmapped and ASCEND_HOME_PATH is absent')
        root=Path(configured).resolve(strict=True)
        declared=root/'lib64'/soname
        target=declared.resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Formal CANN library resolves outside ASCEND_HOME_PATH: '+str(declared))
        _EXPLICIT_LIBRARY_HANDLES.append(ctypes.CDLL(str(target)))
        mapped=mapped_libraries()
        if target not in mapped:
            raise ValueError('Explicitly loaded formal CANN library is still not mapped: '+str(target))
        loads.append({'declared_path':str(declared),'actual_path':str(target),'reason':'required primary library was lazy/unmapped'})
    return mapped,loads


def collect_runtime_identity():
    """Call after dist init and precision setup. No new SHMEM session is made."""
    import torch
    import torch_npu
    import deep_ep
    from deep_ep._native import require_native
    from megamoe_training.gemm import CatlassGemm
    import megamoe_training
    gemm=CatlassGemm()
    core=require_native('training qualification provenance')
    primary_shmem=Path(core.shmem_library_path()).resolve(strict=True)
    mapped,explicit_loads=ensure_required_cann_mapped(mapped_libraries())
    native_paths={'native.catlass':Path(gemm.path).resolve(strict=True),
                  'native.deepep_core':Path(core.__file__).resolve(strict=True),
                  'native.shmem':primary_shmem}
    for role,path in native_paths.items():
        if path not in mapped:
            raise ValueError(f'{role} was reported by its loader but is not mapped: {path}')
    # Reject same-basename ambiguity rather than silently choosing a library.
    seen={}
    for path in sorted(mapped,key=str):
        if any(key in path.name for key in ('shmem','ascendcl','runtime','opapi','hccl')):
            digest=sha256_file(path)
            if path.name in seen and seen[path.name]!=digest:
                raise ValueError('Same loaded basename has different contents: '+path.name)
            seen[path.name]=digest
    native_paths.update({
        'native.cann_acl':unique_loaded(mapped,lambda name:name.startswith('libascendcl.so'),'CANN libascendcl'),
        'native.cann_runtime':unique_loaded(mapped,lambda name:name.startswith('libruntime.so'),'CANN libruntime'),
        'native.cann_opapi':unique_loaded(mapped,lambda name:name.startswith('libopapi.so'),'CANN libopapi'),
    })
    # Include additional actual SHMEM/CANN/HCCL dependencies under stable roles.
    for path in sorted(mapped,key=str):
        if any(key in path.name for key in ('shmem','ascendcl','runtime','opapi','hccl')):
            native_paths['loaded_dependency/'+path.name]=path
    root=Path(megamoe_training.__file__).resolve().parents[1]
    files={role:root/relative for role,relative in SOURCE_FILES.items()}
    deepep_root=Path(deep_ep.__file__).resolve().parent
    for path in sorted(deepep_root.rglob('*.py')):
        files['deepep_python/'+path.relative_to(deepep_root).as_posix()]=path
    files.update(native_paths)
    # The public HF32 getter reads back actual native option state. Internal
    # format has no equivalent public getter; use the same native option store.
    internal=torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
    if isinstance(internal,bytes):internal=internal.decode()
    if internal not in ('enable','disable'):
        raise ValueError('Cannot read back native ALLOW_INTERNAL_FORMAT option')
    result=build_identity(files,{'torch':str(torch.__version__),'torch_npu':str(torch_npu.__version__)},
        {'matmul_allow_hf32':torch.npu.matmul.allow_hf32,'allow_internal_format':internal=='enable'})
    result['diagnostic_shmem_binding']='deep_ep._C.shmem_library_path(): dladdr(aclshmemx_init_attr), confirmed in /proc/self/maps'
    result['diagnostic_explicit_cann_loads']=explicit_loads
    return result


def collective_identity(torch,dist):
    """Exchange success on the existing group before raising; no new sessions."""
    identity=None;error=None
    try:identity=collect_runtime_identity()
    except Exception as exc:error=repr(exc)
    status=torch.tensor([int(error is None)],device=f'npu:{torch.npu.current_device()}',dtype=torch.int32)
    dist.all_reduce(status,op=dist.ReduceOp.MIN)
    if not status.item():
        raise ValueError(error or 'Another rank failed content identity collection; no qualification/performance is permitted')
    return identity


def require_training_qualification(directory,world,shape,current_identity):
    current_digest=validate_identity(current_identity)
    if directory is None:
        raise ValueError('--benchmark/--profile requires --training-qualification-dir')
    evidence=[]
    for rank in range(world):
        path=Path(directory)/f'rank{rank}.json'
        if not path.is_file():
            raise ValueError(f'Missing training qualification {path}; distribute all rank reports on every host')
        record=json.loads(path.read_text())
        if (record.get('all_passed') is not True or record.get('rank')!=rank or
                record.get('world')!=world or record.get('shape')!=shape or
                record.get('weight_dtype')!='torch.bfloat16' or record.get('backend')!='shmem_catlass'):
            raise ValueError(f'Training qualification rank/shape/BF16/backend/pass gate failed: {path}')
        if validate_identity(record.get('identity'))!=current_digest:
            raise ValueError(f'Qualified implementation/runtime/oracle content changed: {path}')
        balanced=[case for case in record.get('cases',[]) if case.get('case')=='balanced']
        if not balanced or not all(case.get('all_ranks_numerical_pass') is True for case in balanced):
            raise ValueError(f'No passing balanced full-gradient qualification: {path}')
        evidence.append({'rank':rank,'path':str(path.resolve()),'sha256':sha256_file(path),'identity_digest':current_digest})
    return evidence
