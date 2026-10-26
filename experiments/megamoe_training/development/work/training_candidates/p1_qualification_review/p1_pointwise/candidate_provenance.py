"""P1-only content/config identity. Stdlib import; never accepts P0 identity."""
import hashlib
import json
import sys
from pathlib import Path

SOURCES=('__init__.py','training_adapter.py','pointwise.py','native/pointwise.cpp',
         'CMakeLists.txt','test_pointwise_npu.py','candidate_provenance.py','validate_training_p1.py','mapped_identity.py')
FIXED={'candidate':'P1_native_pointwise_v1','backend':'shmem_catlass_pointwise_p1',
       'vector_cores':64,'gemm_cores':32,'schedule':'serial','recompute':False}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_identity(identity,base):
    if not isinstance(identity,dict) or identity.get('schema')!='p1-training-2':
        raise ValueError('P1 identity required; old P0 reports cannot qualify P1')
    content=identity.get('content',{})
    if identity.get('content_digest')!=digest(content):raise ValueError('Candidate digest mismatch')
    base.validate_identity(content.get('base_identity'))
    config=content.get('execution',{})
    if any(config.get(k)!=v or type(config.get(k)) is not type(v) for k,v in FIXED.items()):
        raise ValueError('Candidate fixed execution configuration changed')
    if set(config)!=set(FIXED)|{'world','shape','seed','weight_dtype','shared','cases','oracle_chunk','max_reference_gib','limits'}:
        raise ValueError('Incomplete execution binding')
    shape=config['shape'];cases=config['cases']
    if (type(config['world']) is not int or config['world'] not in (2,16)
            or type(config['shared']) is not bool or type(config['seed']) is not int
            or config['weight_dtype'] not in ('torch.bfloat16','torch.float32')
            or not isinstance(shape,dict) or set(shape)!={'B','H','I','E','K'}
            or any(type(v) is not int or v<=0 for v in shape.values())
            or not isinstance(cases,list) or not cases or len(cases)!=len(set(cases))
            or set(cases)-{'balanced','zero_gate','empty','skew'}):
        raise ValueError('Malformed candidate execution configuration')
    limits={name:dict(atol=.003,rtol=.03,rms_relative=.02) for name in ('Y','dx','dp','dW1','dW2')}
    if config['limits']!=limits:raise ValueError('Candidate fixed precision limits changed')
    files=content.get('candidate_files')
    roles={'source/'+name for name in SOURCES}|{'native.pointwise'}
    if not isinstance(files,list) or {item.get('role') for item in files}!=roles or len(files)!=len(roles):
        raise ValueError('Missing/duplicate candidate role')
    for item in files:
        if (set(item)!={'role','sha256','bytes'} or type(item['bytes']) is not int or item['bytes']<=0
                or not isinstance(item['sha256'],str) or len(item['sha256'])!=64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('Invalid candidate file identity')
    maps=content.get('mapped_libraries')
    if not isinstance(maps,dict) or not maps:raise ValueError('Missing reliable FD canonical mappings')
    for role,item in maps.items():
        if (not valid_map_role(role) or set(item)!={'sha256','bytes'} or type(item['bytes']) is not int or item['bytes']<=0 or
            not isinstance(item['sha256'],str) or len(item['sha256'])!=64 or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('Malformed reliable mapped library role/content')
    return identity['content_digest']


def build_identity(base_identity,files,execution,base,mapped):
    base.validate_identity(base_identity)
    # Strip diagnostic host paths from nested identity before hashing.
    bound_base={k:base_identity[k] for k in ('schema','content','content_digest')}
    manifest=[];paths={}
    for role,path in sorted(files.items()):
        path=Path(path).resolve(strict=True)
        manifest.append(dict(role=role,bytes=path.stat().st_size,sha256=base.sha256_file(path)))
        paths[role]=str(path)
    content=dict(base_identity=bound_base,candidate_files=manifest,execution=execution,mapped_libraries=canonical_maps(mapped))
    result=dict(schema='p1-training-2',content=content,content_digest=digest(content),diagnostic_paths=paths)
    validate_identity(result,base)
    return result


def require_identity_match(qualified,current,base):
    if validate_identity(qualified,base)!=validate_identity(current,base):
        raise ValueError('P1 implementation/library/execution changed since qualification')


def collect_identity(root,library,execution,base):
    """Only loads native libraries; creates no ElasticBuffer/SHMEM domain."""
    from p1_pointwise.pointwise import TrainingPointwise
    pointwise=TrainingPointwise(library,vector_cores=64)
    base_identity=base.collect_runtime_identity()
    from p1_pointwise.mapped_identity import mapped_libraries
    import p1_pointwise.mapped_identity as fd_helper
    if Path(fd_helper.__file__).resolve()!=Path(root).resolve()/'mapped_identity.py':
        raise ValueError('FD helper escaped explicit candidate root')
    package_roots=imported_package_roots();raw=mapped_libraries();mapped=checked_maps(raw,package_roots)
    selected=Path(pointwise.path).resolve(strict=True)
    files={'source/'+name:Path(root)/name for name in SOURCES};files['native.pointwise']=selected
    identity=build_identity(base_identity,files,execution,base,mapped)
    bind_selected_maps(identity,base_identity,selected,mapped)
    identity.update(actual_maps=list(mapped.values()),raw_mapping_result=raw,
                    diagnostic_package_roots={k:str(v) for k,v in package_roots.items()})
    return identity,pointwise


def bind_selected_maps(identity,base_identity,pointwise_path,mapped):
    """All base native/dependency roles and selected pointwise must match exact mapped FD."""
    base_items={v['role']:v for v in base_identity['content']['files']}
    paths=base_identity.get('diagnostic_paths',{})
    required={role for role in base_items if role.startswith(('native.','loaded_dependency/'))}
    if not required<=paths.keys():raise ValueError('Base loader exact paths missing')
    roles={role:(paths[role],base_items[role]) for role in required}
    roles['native.pointwise']=(str(pointwise_path),next(v for v in identity['content']['candidate_files'] if v['role']=='native.pointwise'))
    for role,(path,expected) in roles.items():
        if path not in mapped or any(mapped[path][k]!=expected[k] for k in ('sha256','bytes')):
            raise ValueError('Selected native/dependency not exact-path FD bound: '+role)



def collective_checked(torch,dist,action,label):
    value=None;error=None
    try:value=action()
    except Exception as exc:error=repr(exc)
    flag=torch.tensor([int(error is None)],device=f'npu:{torch.npu.current_device()}',dtype=torch.int32)
    dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    if not flag.item():raise RuntimeError(label+': '+(error or 'another rank failed; no SHMEM engine may be entered'))
    return value


def final_identity(initial,current,base):
    """Immutable implementation/config/old dependencies; only lazy dependencies may add."""
    validate_identity(initial,base);validate_identity(current,base)
    a,b=initial['content'],current['content']
    if set(a)!=set(b) or any(a[k]!=b[k] for k in a if k not in ('base_identity','mapped_libraries')):
        raise ValueError('P1 candidate source/native/execution changed during run')
    old,new=a['base_identity']['content'],b['base_identity']['content']
    if set(old)!=set(new) or any(old[k]!=new[k] for k in old if k!='files'):
        raise ValueError('P1 base version/precision/config changed during run')
    af={v['role']:v for v in old['files']};bf={v['role']:v for v in new['files']}
    if any(k not in bf or bf[k]!=v for k,v in af.items()):
        raise ValueError('P1 initial source/native/dependency changed or removed')
    added=bf.keys()-af.keys()
    if any(not k.startswith('loaded_dependency/') for k in added):
        raise ValueError('P1 non-dependency role added during run')
    am,bm=a['mapped_libraries'],b['mapped_libraries']
    if any(k not in bm or bm[k]!=v for k,v in am.items()):raise ValueError('P1 actual mapped library changed/removed')
    return dict(added={k:bf[k] for k in sorted(added)},mapped_added={k:bm[k] for k in bm.keys()-am.keys()},changed={},removed={})


def finish_identity_host(host,report,collector,initial,base):
    """Host-only final observation; never hide a first local collector error."""
    error=None;current=None
    try:
        current=collector();report['identity_final']=current
        report['identity_difference']=final_identity(initial,current,base)
    except Exception as exc:
        import traceback
        error=exc;report.update(all_passed=False,final_identity_error=repr(exc),final_identity_traceback=traceback.format_exc())
    try:
        peers=host.all_gather(dict(rank=host.rank(),error=None if error is None else repr(error),
            digest=current['content_digest'] if current is not None else None),'P1/final-identity')
        report['final_identity_peer_status']=peers
        if (current is None or [v.get('rank') for v in peers]!=list(range(host.size())) or
            any(v.get('error') is not None for v in peers) or
            any(v.get('digest')!=current['content_digest'] for v in peers)):
            raise RuntimeError('A peer failed final P1 identity or ranks differ')
    except Exception as exc:
        report.update(all_passed=False,final_identity_host_error=repr(exc))
        return error or exc
    return error


def valid_map_role(role):
    if not isinstance(role,str) or '\\' in role:return False
    parts=role.split('/')
    if any(x in ('','.','..') for x in parts):return False
    return (len(parts)==2 and parts[0]=='loaded') or (len(parts)>=3 and parts[:2] in (['package','deep_ep'],['package','torch_npu']))

def imported_package_roots():
    roots={}
    for name in ('deep_ep','torch_npu'):
        module=sys.modules.get(name)
        if module is None or not getattr(module,'__file__',None):raise ValueError('Missing actual imported package '+name)
        roots[name]=Path(module.__file__).resolve(strict=True).parent
    return roots

def checked_maps(result,package_roots=None):
    if result.get('errors'):raise ValueError('Actual mapped dev/inode evidence failed: '+repr(result['errors']))
    roots={} if package_roots is None else {k:Path(v) for k,v in package_roots.items()}
    if roots and set(roots)!={'deep_ep','torch_npu'}:raise ValueError('Both actual package roots required')
    values=list(roots.values())
    if len(values)==2 and any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for a,b in ((values[0],values[1]),)):
        raise ValueError('Actual package roots overlap')
    paths={};roles={};names={}
    for v in result.get('libraries',[]):
        if v.get('map_inode',0)<=0 or v.get('inode')!=v['map_inode'] or not v.get('map_entries'):raise ValueError('Mapped object missing bound inode/segments')
        path=Path(v['path']);name=path.name;role='loaded/'+name
        for package,root in roots.items():
            if path.is_relative_to(root):role='package/'+package+'/'+path.relative_to(root).as_posix();break
        if not valid_map_role(role) or role in roles or v['path'] in paths:raise ValueError('Duplicate/ambiguous actual mapped logical object')
        roles[role]=v['path'];names.setdefault(name,[]).append(role)
        paths[v['path']]=dict(v,logical_role=role) # Retain every FD/dev/inode/segment object, not only its basename.
    if any(len(group)>1 and any(not role.startswith('package/') for role in group) for group in names.values()):
        raise ValueError('Unclassified same-basename mappings; explicit stable package roles required')
    if not paths:raise ValueError('Mapped library set empty')
    return paths

def canonical_maps(paths):
    return {v['logical_role']:{k:v[k] for k in ('bytes','sha256')} for v in paths.values()}
