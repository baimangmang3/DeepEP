"""Variant/config/content qualification; stdlib import, no runtime creation."""
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
from mapped_identity import mapped_libraries,sha
from prepare_build_contract import REV,PATCH,EXPECTED_EP16_SOURCES,digest

LOCAL=('validate.py','qualification.py','mapped_identity.py','host_store.py','host_exit.py','prepare_build_contract.py')
NATIVE={'native.ep_gmm','native.deep_core','native.kernels','native.catlass','native.shmem'}
ABI16={'_MOE_ROUTE_MAX_RANKS':16,'_MOE_ROUTE_MAX_LOCAL_EXPERTS':64,'_MOE_ROUTE_MAX_SEGMENTS':1024,
       '_MOE_ROUTE_PEER_ROWS_OFFSET':16,'_MOE_ROUTE_EXPERT_ROWS_OFFSET':144,'_MOE_ROUTE_SEGMENTS_OFFSET':656,
       '_MOE_ROUTE_SEGMENT_SIZE':24,'_MOE_ROUTE_METADATA_SIZE':25232}


def require_module_paths(expected):
    """Insertion into sys.path cannot replace already cached modules."""
    for name,path in expected.items():
        module=importlib.import_module(name);origin=getattr(module,'__file__',None)
        if origin is None or Path(origin).resolve()!=Path(path).resolve():
            raise ValueError('Actual cached/imported module escaped explicit root: '+name)


def required_module_paths(root,parent,variant):
    expected={name:Path(root)/relative for name,relative in {
        'provenance':'tests/provenance.py','validate_training_npu':'tests/validate_training_npu.py',
        'megamoe_training':'megamoe_training/__init__.py','megamoe_training.training':'megamoe_training/training.py',
        'megamoe_training.gemm':'megamoe_training/gemm.py'}.items()}
    for package in ('ep_gmm_training',)+(() if variant=='ep8_original' else ('ep_gmm_training_ep16',)):
        expected[package]=Path(parent)/package/'__init__.py'
        expected[package+'.training_adapter']=Path(parent)/package/'training_adapter.py'
    return expected


def check_config(config,require_hardware=False):
    variant=config.get('variant');world=config.get('world');s=config.get('shape',{})
    if variant not in ('ep8_original','ep16_adapted') or type(world) is not int:
        raise ValueError('Explicit variant/world required')
    if not (1<=world<=8 if variant=='ep8_original' else world==16):raise ValueError('Original EP<=8 and adapted EXACT16 cannot mix')
    if set(s)!={'B','H','I','E','K'} or any(type(v) is not int or v<=0 for v in s.values()):raise ValueError('Invalid exact shape')
    if (s['B']>16384 or s['E']%world or s['E']//world>32 or s['K']>min(16,s['E'])
            or s['H']%128 or s['I']%128):raise ValueError('Invalid source-token/native128/expert ABI')
    if config.get('gemm_cores')!=32 or config.get('owner')!='one_unique_id_udma' or config.get('schedule')!='serial' or config.get('recompute') is not False:
        raise ValueError('Fixed engine/UDMA configuration required')
    if config.get('weight_dtype') not in ('torch.bfloat16','torch.float32') or type(config.get('shared')) is not bool:
        raise ValueError('Invalid dtype/shared')
    cases=config.get('cases',[])
    if not cases or len(cases)!=len(set(cases)) or set(cases)-{'balanced','zero_gate','empty','skew'}:raise ValueError('Unknown cases')
    if 'empty' in cases and s['K']>=s['E']:raise ValueError('Empty requires K<E')
    limits={n:dict(atol=.003,rtol=.03,rms_relative=.02) for n in ('Y','dx','dp','dW1','dW2')}
    if config.get('limits')!=limits:raise ValueError('Frozen precision limits changed')
    if (type(config.get('seed')) is not int or type(config.get('heap_size')) is not int
            or not 400*1024**2+4096<config['heap_size']<1<<63
            or type(config.get('oracle_chunk')) is not int or config['oracle_chunk']<=0
            or type(config.get('max_reference_gib')) not in (int,float)
            or not math.isfinite(config['max_reference_gib']) or config['max_reference_gib']<=0):
        raise ValueError('Incomplete reference/seed/UDMA execution binding')
    if require_hardware:
        h=config.get('actual_hardware',{})
        if (set(h)!={'aic_cores','ub_bytes'} or any(type(v) is not int for v in h.values())
                or h['aic_cores']<32 or h['ub_bytes']<=256 or (variant=='ep16_adapted' and h['aic_cores']!=32)):
            raise ValueError('Missing/incompatible actual hardware binding')


def check_contract(contract,variant):
    if contract.get('schema')!='ep-gmm-build-contract-1' or contract.get('content_digest')!=digest(contract.get('content')):
        raise ValueError('Native build contract digest/schema mismatch')
    c=contract['content']
    if c.get('variant')!=variant or c.get('base_revision')!=REV:raise ValueError('Native variant/revision mismatch')
    if c.get('patch_sha256')!=(PATCH if variant=='ep16_adapted' else None):raise ValueError('Reviewed EP16 patch differs')
    if variant=='ep16_adapted' and c.get('source_sha256')!=EXPECTED_EP16_SOURCES:raise ValueError('Missing coordinated11-source build provenance')
    if not c.get('dependencies') or not c.get('embedded_metadata') or not c.get('wheel_python'):raise ValueError('Missing build dependency/Python records')
    for role in ('native.ep_gmm','native.deep_core','native.kernels','native.shmem'):
        if not re.fullmatch('[0-9a-f]{64}',str(c.get('native',{}).get(role,{}).get('sha256'))):raise ValueError('Missing compiled native role '+role)
    return c


def validate_identity(identity,base_validate):
    if identity.get('schema')!='ep-gmm-training-1' or identity.get('content_digest')!=digest(identity.get('content')):
        raise ValueError('EP-GMM identity required; P0/P1 identities cannot qualify this backend')
    content=identity['content'];check_config(content['execution'],True);check_contract(content['build_contract'],content['execution']['variant'])
    base_validate(content['base_identity'])
    rows=content.get('files',[]);roles=[v['role'] for v in rows]
    wrappers={'wrapper/ep_gmm_training/'+name for name in ('__init__.py','training_adapter.py')}
    if content['execution']['variant']=='ep16_adapted':wrappers|={'wrapper/ep_gmm_training_ep16/'+name for name in ('__init__.py','training_adapter.py')}
    if len(roles)!=len(set(roles)) or NATIVE-set(roles) or ({'harness/'+n for n in LOCAL}|wrappers)-set(roles):raise ValueError('Missing/duplicate identity role')
    for row in rows:
        if set(row)!={'role','bytes','sha256'} or row['bytes']<=0 or not re.fullmatch('[0-9a-f]{64}',row['sha256']):raise ValueError('Invalid file role')
    actual={row['role']:row for row in rows}
    for role,expected in content['build_contract']['content']['native'].items():
        if role not in actual or actual[role]['sha256']!=expected['sha256'] or ('bytes' in expected and actual[role]['bytes']!=expected['bytes']):
            raise ValueError('Actual loaded SO differs from audited wheel/SDK: '+role)
    return identity['content_digest']


def require_identity_match(qualified,current,base_validate):
    if validate_identity(qualified,base_validate)!=validate_identity(current,base_validate):
        raise ValueError('EP-GMM variant/config/source/actual native identity changed')


def collect(config,contract,base,candidate_parent,variant_module,ep,ep_core,core,ops):
    """Explicit loaders already imported; no Session/ElasticBuffer constructed."""
    check_config(config);check_contract(contract,config['variant'])
    require_module_paths(required_module_paths(Path(base.__file__).resolve().parents[1],candidate_parent,config['variant']))
    if config['variant']=='ep16_adapted':
        if any(getattr(ops,k,None)!=v for k,v in ABI16.items()):raise ValueError('EP16 Python ABI mismatch')
    elif getattr(ops,'_MOE_ROUTE_MAX_RANKS',None)!=8 or getattr(ops,'_MOE_ROUTE_METADATA_SIZE',None)!=12880:
        raise ValueError('Pinned EP8 Python ABI mismatch')
    b=base.collect_runtime_identity()  # loads CATLASS/CANN libs, never a Session
    inventory=mapped_libraries()
    if inventory['errors']:raise ValueError('Unreliable actual maps identity: '+repr(inventory['errors']))
    mapped={str(Path(v['path']).resolve()):v for v in inventory['libraries']}
    # Two separately named Python extension roles can legitimately share _C
    # basename. All other same-basename native dependencies must agree.
    extension_paths={str(Path(ep_core.__file__).resolve()),str(Path(core.__file__).resolve())}
    basename={}
    for path,entry in mapped.items():
        if path in extension_paths:continue
        name=Path(path).name
        if name in basename and basename[name]!=entry['sha256']:raise ValueError('Multiple loaded contents for native basename: '+name)
        basename[name]=entry['sha256']
    expected_paths={'native.ep_gmm':Path(ep_core.__file__).resolve(),'native.deep_core':Path(core.__file__).resolve(),
        'native.shmem':Path(core.shmem_library_path()).resolve(),'native.catlass':Path(b['diagnostic_paths']['native.catlass']).resolve()}
    kernels=[Path(p) for p in mapped if Path(p).name=='libdeepep_kernels.so']
    if len(kernels)!=1:raise ValueError('Missing/ambiguous actual libdeepep_kernels.so mapping')
    expected_paths['native.kernels']=kernels[0]
    files=[];diagnostics={};used=set()
    for role,path in expected_paths.items():
        entry=mapped.get(str(path))
        if entry is None:raise ValueError('Reported loader path not actually mapped: '+role)
        files.append({'role':role,'sha256':entry['sha256'],'bytes':entry['bytes']});diagnostics[role]=entry;used.add(str(path))
    deps={}
    for path,entry in mapped.items():
        if path in used:continue
        name=Path(path).name
        if name in deps and deps[name]['sha256']!=entry['sha256']:raise ValueError('Ambiguous actual dependency basename: '+name)
        deps[name]=entry
    for name,entry in sorted(deps.items()):files.append({'role':'loaded/'+name,'sha256':entry['sha256'],'bytes':entry['bytes']})
    paths={'harness/'+n:Path(__file__).parent/n for n in LOCAL}
    parent=Path(candidate_parent)
    for name in ('ep_gmm_training',)+(() if config['variant']=='ep8_original' else ('ep_gmm_training_ep16',)):
        for relative in ('__init__.py','training_adapter.py'):paths['wrapper/'+name+'/'+relative]=parent/name/relative
    import deep_ep
    for package_name,root in [('deepep',Path(deep_ep.__file__).parent),('ep_gmm',Path(ep.__file__).parent)]:
        for path in root.rglob('*.py'):
            relative=path.relative_to(root).as_posix();member=('deep_ep/' if package_name=='deepep' else 'deep_ep_experimental/ep_gmm_fused/')+relative
            expected=contract['content']['wheel_python'].get(member)
            if expected is None or sha(path)!=expected['sha256'] or path.stat().st_size!=expected['bytes']:
                raise ValueError('Imported Python differs from audited wheel: '+member)
            paths['python/'+package_name+'/'+relative]=path
        for path in root.glob('*.json'):paths['metadata/'+package_name+'/'+path.name]=path
    namespace=Path(ep.__file__).parent.parent/'__init__.py'
    expected=contract['content']['wheel_python'].get('deep_ep_experimental/__init__.py')
    if expected is None or sha(namespace)!=expected['sha256']:raise ValueError('Experimental namespace differs from audited wheel')
    paths['python/experimental_namespace/__init__.py']=namespace
    for member,metadata in contract['content']['embedded_metadata'].items():
        path=Path(deep_ep.__file__).parent.parent/member
        if sha(path)!=metadata['sha256']:raise ValueError('Installed embedded metadata differs from audited wheel: '+member)
    for member,expected in contract['content']['wheel_python'].items():
        path=Path(deep_ep.__file__).parent.parent/member
        if sha(path)!=expected['sha256'] or path.stat().st_size!=expected['bytes']:raise ValueError('Missing/changed audited wheel Python: '+member)
    for role,path in sorted(paths.items()):
        files.append({'role':role,'sha256':sha(path),'bytes':path.stat().st_size})
    content={'execution':config,'build_contract':contract,'base_identity':{k:b[k] for k in ('schema','content','content_digest')},'files':sorted(files,key=lambda x:x['role'])}
    result={'schema':'ep-gmm-training-1','content':content,'content_digest':digest(content),'mapping_evidence':inventory,
            'diagnostic_native_roles':diagnostics,'diagnostic_paths':{k:str(v) for k,v in paths.items()}}
    validate_identity(result,base.validate_identity)
    return result
