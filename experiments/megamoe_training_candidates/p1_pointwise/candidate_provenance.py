"""P1-only content/config identity. Stdlib import; never accepts P0 identity."""
import hashlib
import json
from pathlib import Path

SOURCES=('__init__.py','training_adapter.py','pointwise.py','native/pointwise.cpp',
         'CMakeLists.txt','test_pointwise_npu.py','candidate_provenance.py','validate_training_p1.py')
FIXED={'candidate':'P1_native_pointwise_v1','backend':'shmem_catlass_pointwise_p1',
       'vector_cores':64,'gemm_cores':32,'schedule':'serial','recompute':False}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_identity(identity,base):
    if not isinstance(identity,dict) or identity.get('schema')!='p1-training-1':
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
    return identity['content_digest']


def build_identity(base_identity,files,execution,base):
    base.validate_identity(base_identity)
    # Strip diagnostic host paths from nested identity before hashing.
    bound_base={k:base_identity[k] for k in ('schema','content','content_digest')}
    manifest=[];paths={}
    for role,path in sorted(files.items()):
        path=Path(path).resolve(strict=True)
        manifest.append(dict(role=role,bytes=path.stat().st_size,sha256=base.sha256_file(path)))
        paths[role]=str(path)
    content=dict(base_identity=bound_base,candidate_files=manifest,execution=execution)
    result=dict(schema='p1-training-1',content=content,content_digest=digest(content),diagnostic_paths=paths)
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
    mapped=base.mapped_libraries();selected=Path(pointwise.path).resolve(strict=True)
    if selected not in mapped:raise ValueError('Selected pointwise SO is not actually mapped')
    for path in mapped:
        if path.name==selected.name and base.sha256_file(path)!=base.sha256_file(selected):
            raise ValueError('Ambiguous mapped pointwise SO contents')
    files={'source/'+name:Path(root)/name for name in SOURCES}
    files['native.pointwise']=selected
    return build_identity(base_identity,files,execution,base),pointwise


def collective_checked(torch,dist,action,label):
    value=None;error=None
    try:value=action()
    except Exception as exc:error=repr(exc)
    flag=torch.tensor([int(error is None)],device=f'npu:{torch.npu.current_device()}',dtype=torch.int32)
    dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    if not flag.item():raise RuntimeError(label+': '+(error or 'another rank failed; no SHMEM engine may be entered'))
    return value
