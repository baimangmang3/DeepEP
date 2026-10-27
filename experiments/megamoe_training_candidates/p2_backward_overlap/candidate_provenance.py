"""P2-only content/schedule identity and strict accuracy report gate; stdlib import."""
import hashlib,importlib,json,math,sys
from pathlib import Path

SCHEMA='p2-training-2'
ADAPTER_SHA='10b3c34d27dd97273c4c19c64c2416dfe76ccb953243c45e806cd8c3f750abce'
P1_ADAPTER_SHA='9cf3968261cd1c5823e573f12510bbc902937981352e84ad00b7a8e8f7294d44'
ELASTIC_SHA='117b01651bc503839ff2aa681fca55308867b09c8bda399b5a0e47ca8b9b8c1c'
FIXED=dict(candidate='P2_backward_return_overlap_v1_UNQUALIFIED',backend='shmem_catlass_backward_overlap_p2',
    vector_cores=64,gemm_cores=32,recompute=False)
SOURCES=('__init__.py','training_adapter.py','candidate_provenance.py','validate_training_p2.py','host_store.py','mapped_identity.py')
P1_SOURCES=('__init__.py','training_adapter.py','pointwise.py','native/pointwise.cpp','CMakeLists.txt')
LIMITS={n:dict(atol=.003,rtol=.03,rms_relative=.02) for n in ('Y','dx','dp','dW1','dW2')}
SMALL={2:(32,128,128,8,4),16:(32,128,128,64,8)}
MAIN=(8192,6144,2048,64,8)

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def check_execution(e):
    dynamic={'world','shape','seed','weight_dtype','shared','cases','oracle_chunk','max_reference_gib','limits','backward_schedule'}
    if not isinstance(e,dict) or set(e)!=set(FIXED)|dynamic or any(type(e[k]) is not type(v) or e[k]!=v for k,v in FIXED.items()):raise ValueError('P2 fixed execution mismatch')
    s=e['shape'];w=e['world'];cases=e['cases']
    if type(w) is not int or w not in (2,16) or not isinstance(s,dict) or set(s)!={'B','H','I','E','K'} or any(type(v) is not int or v<=0 for v in s.values()):raise ValueError('P2 world/shape malformed')
    shape=tuple(s[k] for k in ('B','H','I','E','K'))
    if shape!=SMALL[w] and not(w==16 and shape==MAIN):raise ValueError('Only reviewed EP2 E8K4/EP16 E64K8 small and exact main shape admitted')
    if (e['backward_schedule'] not in ('serial','overlap') or type(e['shared']) is not bool or type(e['seed']) is not int or
        e['weight_dtype'] not in ('torch.bfloat16','torch.float32') or not isinstance(cases,list) or not cases or
        len(cases)!=len(set(cases)) or set(cases)-{'balanced','zero_gate','empty','skew'} or
        type(e['oracle_chunk']) is not int or e['oracle_chunk']<=0 or type(e['max_reference_gib']) not in (int,float) or
        not 0<e['max_reference_gib']<float('inf') or e['limits']!=LIMITS):raise ValueError('P2 schedule/dtype/cases/limits malformed')
    if shape==MAIN and cases!=['balanced']:raise ValueError('Main shape admits balanced only')
    if shape!=MAIN and set(cases)!={'balanced','zero_gate','empty','skew'}:raise ValueError('Small qualification requires all four cases')

def execution_from_args(args,world):
    if args.world_size!=world:raise ValueError('CLI world-size differs from actual group')
    shape={k:getattr(args,k) for k in ('B','H','I','E','K')}
    if tuple(shape[k] for k in ('B','H','I','E','K'))==MAIN and args.cases!='balanced':raise ValueError('Main requires explicit --cases balanced')
    cases=(args.cases or 'balanced,zero_gate,empty,skew').split(',')
    e=dict(FIXED,world=world,shape=shape,seed=args.seed,weight_dtype={'bf16':'torch.bfloat16','fp32':'torch.float32'}[args.weight_dtype],
        shared=args.shared,cases=cases,oracle_chunk=args.oracle_chunk,max_reference_gib=args.max_reference_gib,limits=LIMITS,
        backward_schedule=args.backward_schedule)
    check_execution(e);return e

def check_source_layout(frozen,root,p1):
    for folder,name,files in ((root,'p2_backward_overlap',SOURCES),(p1,'p1_pointwise',P1_SOURCES)):
        folder=Path(folder).resolve(strict=True)
        if folder.name!=name or any(not(folder/f).is_file() for f in files):raise ValueError('Incomplete/incorrect package root '+name)
    if sha(Path(root)/'training_adapter.py')!=ADAPTER_SHA or sha(Path(p1)/'training_adapter.py')!=P1_ADAPTER_SHA:raise ValueError('Expected unchanged P2 and original P1 adapters')
    for f in ('tests/provenance.py','tests/validate_training_npu.py','megamoe_training/__init__.py','megamoe_training/training.py','megamoe_training/gemm.py','native/gemm.cpp','CMakeLists.txt'):
        if not(Path(frozen)/f).is_file():raise ValueError('Incomplete frozen P0 root')

def check_imports(frozen,root,p1):
    frozen,root,p1=[Path(p).resolve(strict=True) for p in (frozen,root,p1)]
    roots=(frozen,root,p1)
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]):raise ValueError('Source roots overlap')
    paths={'provenance':frozen/'tests/provenance.py','validate_training_npu':frozen/'tests/validate_training_npu.py',
        'candidate_provenance':root/'candidate_provenance.py','p2_backward_overlap':root/'__init__.py',
        'p2_backward_overlap.training_adapter':root/'training_adapter.py','p2_backward_overlap.mapped_identity':root/'mapped_identity.py',
        'p1_pointwise':p1/'__init__.py','p1_pointwise.training_adapter':p1/'training_adapter.py','p1_pointwise.pointwise':p1/'pointwise.py'}
    paths.update({'megamoe_training'+suffix:frozen/'megamoe_training'/name for suffix,name in (('','__init__.py'),('.training','training.py'),('.gemm','gemm.py'))})
    for name,path in paths.items():
        module=importlib.import_module(name)
        if Path(module.__file__).resolve()!=path:raise ValueError('Actual imported module escaped declared root: '+name)
    return paths

def original_dependencies(base_identity):
    manifest={v['role']:v for v in base_identity['content']['files']}
    if manifest.get('deepep_python/buffers/elastic.py',{}).get('sha256')!=ELASTIC_SHA:raise ValueError('Expected original DeepEP elastic.py, not hosttrace overlay')

def validate_identity(v,base):
    if not isinstance(v,dict) or v.get('schema')!=SCHEMA:raise ValueError('P2-only identity required; P0/P1/P3 reports rejected')
    c=v.get('content',{})
    if v.get('content_digest')!=digest(c):raise ValueError('P2 digest mismatch')
    base.validate_identity(c.get('base_identity'));original_dependencies(c['base_identity']);check_execution(c.get('execution'))
    files=c.get('files',[]);expected={f'{p}/{n}' for p,ns in (('p2',SOURCES),('p1',P1_SOURCES)) for n in ns}|{'native.catlass','native.pointwise'}
    roles=[x.get('role') for x in files]
    if set(roles)!=expected or len(roles)!=len(expected):raise ValueError('P2 source/native roles missing/duplicated')
    for item in files:
        if set(item)!={'role','bytes','sha256'} or type(item['bytes']) is not int or item['bytes']<=0 or not isinstance(item['sha256'],str) or len(item['sha256'])!=64 or any(c not in '0123456789abcdef' for c in item['sha256']):raise ValueError('Malformed P2 content entry')
    by={v['role']:v['sha256'] for v in files}
    if by['p2/training_adapter.py']!=ADAPTER_SHA or by['p1/training_adapter.py']!=P1_ADAPTER_SHA:raise ValueError('Adapter implementation differs')
    bound_catlass=next(v for v in c['base_identity']['content']['files'] if v['role']=='native.catlass')
    own_catlass=next(v for v in files if v['role']=='native.catlass')
    if any(own_catlass[k]!=bound_catlass[k] for k in ('sha256','bytes')):raise ValueError('P2 CATLASS binding differs from base loader')
    maps=c.get('mapped_libraries')
    if not isinstance(maps,dict) or not maps:raise ValueError('Actual mapped dependency manifest absent')
    for key,x in maps.items():
        if not valid_map_role(key) or set(x)!={'sha256','bytes'} or type(x['bytes']) is not int or x['bytes']<=0 or not isinstance(x['sha256'],str) or len(x['sha256'])!=64 or any(ch not in '0123456789abcdef' for ch in x['sha256']):raise ValueError('Malformed mapped dependency content')
    for role in ('native.catlass','native.pointwise'):
        item=next(x for x in files if x['role']==role)
        if not any(all(x[k]==item[k] for k in ('sha256','bytes')) for x in maps.values()):raise ValueError('Native role is not represented in actual mapped manifest')
    return v['content_digest']

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

def collect_identity(args,root,execution,base):
    check_source_layout(args.training_source_root,root,args.p1_source_root)
    imports=check_imports(args.training_source_root,root,args.p1_source_root)
    from p1_pointwise.pointwise import TrainingPointwise
    from p2_backward_overlap.mapped_identity import mapped_libraries
    pointwise=TrainingPointwise(args.pointwise_library,vector_cores=64)
    b=base.collect_runtime_identity();base.validate_identity(b);original_dependencies(b)
    package_roots=imported_package_roots()
    mapped=checked_maps(mapped_libraries(),package_roots)
    selected={'pointwise':Path(pointwise.path).resolve(),'catlass':Path(b['diagnostic_paths']['native.catlass']).resolve()}
    for role,path in b['diagnostic_paths'].items():
        if role.startswith(('native.','loaded_dependency/')):
            expected=next(v for v in b['content']['files'] if v['role']==role)
            if path not in mapped or any(mapped[path][k]!=expected[k] for k in ('bytes','sha256')):raise ValueError('Base native object is not actual FD-map bound: '+role)
    for path in selected.values():
        if str(path) not in mapped:raise ValueError('Selected native object not mapped: '+str(path))
    files={f'{prefix}/{n}':Path(folder)/n for prefix,folder,ns in (('p2',root,SOURCES),('p1',args.p1_source_root,P1_SOURCES)) for n in ns}
    entries=[dict(role=role,bytes=p.stat().st_size,sha256=sha(p)) for role,p in sorted(files.items())]
    entries.extend(dict(role='native.'+r,bytes=mapped[str(p)]['bytes'],sha256=mapped[str(p)]['sha256']) for r,p in selected.items())
    content=dict(base_identity={k:b[k] for k in ('schema','content','content_digest')},execution=execution,files=entries,
        mapped_libraries=canonical_maps(mapped))
    value=dict(schema=SCHEMA,content=content,content_digest=digest(content),actual_maps=list(mapped.values()),
        diagnostic_package_roots={k:str(p) for k,p in package_roots.items()},diagnostic_import_paths={k:str(p) for k,p in imports.items()})
    validate_identity(value,base);return value,pointwise

def require_identity_match(qualified,current,base):
    if validate_identity(qualified,base)!=validate_identity(current,base):raise ValueError('P2 source/SO/dependency/execution/schedule changed')

def final_identity(initial,current,base):
    validate_identity(initial,base);validate_identity(current,base)
    a,b=initial['content'],current['content']
    if any(a[k]!=b[k] for k in ('files','execution')):raise ValueError('P2 source/native/config changed during run')
    x,y=a['base_identity']['content'],b['base_identity']['content']
    if any(x[k]!=y[k] for k in ('versions','precision')):raise ValueError('Base precision/framework changed')
    old={v['role']:v for v in x['files']};new={v['role']:v for v in y['files']}
    if any(k not in new or new[k]!=v for k,v in old.items()) or any(not k.startswith('loaded_dependency/') for k in new.keys()-old.keys()):raise ValueError('Base source/native changed')
    old=a['mapped_libraries'];new=b['mapped_libraries']
    if any(k not in new or new[k]!=v for k,v in old.items()):raise ValueError('Initially mapped content changed/removed')
    return dict(added={k:v for k,v in new.items() if k not in old},changed={},removed={})

def metric_passes(m,dtype,elements):
    if (not isinstance(m,dict) or m.get('passed') is not True or m.get('finite') is not True or m.get('out_of_tolerance_elements')!=0 or
        m.get('limits')!=LIMITS['Y'] or m.get('actual_dtype')!=dtype or m.get('reference_dtype')!=dtype or m.get('elements')!=elements):raise ValueError('Missing/failed full metric')
    nums=('RMS_error','reference_RMS','RMS_relative_error','max_abs_error','max_relative_error_floor_1e_8','max_reference_abs')
    if any(type(m.get(k)) not in (int,float) or not math.isfinite(m[k]) or m[k]<0 for k in nums):raise ValueError('Nonfinite/invalid metric summary')
    rms,ref=m['RMS_error'],m['reference_RMS'];ratio=rms/max(ref,1e-12)
    if not math.isclose(m['RMS_relative_error'],ratio,rel_tol=1e-9,abs_tol=1e-12) or (ratio>.02 if ref else rms>.003):raise ValueError('Fixed RMS failed')
    if m['max_abs_error']>.003+.03*m['max_reference_abs']+1e-12:raise ValueError('Metric max error contradicts zero out-of-tolerance count')

def expected_counts(execution,rank,case):
    s=execution['shape'];world=execution['world'];b,e,k=(s[x] for x in ('B','E','K'))
    if case=='skew':return [world*b if rank*4+j<k else 0 for j in range(4)]
    if case=='empty':
        q,r=divmod(world*b*k,e-1)
        return [q+int(rank*4+j<r) if rank*4+j<e-1 else 0 for j in range(4)]
    return [world*b*k//e]*4 # All admitted spread-balanced shapes divide exactly.

def require_reports(directory,execution,current,base):
    """Accuracy evidence gate only. No timing/performance entry is implemented."""
    check_execution(execution);validate_identity(current,base)
    if current['content']['execution']!=execution:raise ValueError('Current identity execution mismatch')
    world=execution['world'];s=execution['shape'];b,h,i,k=(s[n] for n in ('B','H','I','K'));evidence=[]
    for rank in range(world):
        r=json.loads((Path(directory)/f'rank{rank}.json').read_text(encoding='utf-8'))
        if (r.get('rank')!=rank or r.get('world')!=world or r.get('backend')!=FIXED['backend'] or r.get('all_passed') is not True or
            r.get('diagnostic_only') or r.get('error') or r.get('cleanup_errors') or r.get('final_identity_error') or r.get('final_identity_host_error') or
            r.get('execution')!=execution or r.get('shape')!=s or r.get('seed')!=execution['seed'] or
            r.get('weight_dtype')!=execution['weight_dtype'] or r.get('shared') is not execution['shared'] or r.get('limits')!=LIMITS):raise ValueError('P2 rank report/config/exit failed')
        validate_identity(r.get('identity'),base);final_identity(r.get('identity'),r.get('identity_final'),base)
        require_identity_match(r['identity_final'],current,base)
        cases=r.get('cases',[])
        if [v.get('case') for v in cases]!=execution['cases']:raise ValueError('P2 cases/order incomplete')
        for c in cases:
            if any(c.get(key) is not True for key in ('all_ranks_numerical_pass','counts_match','shapes_match','dtypes_match','schedule_match')) or c.get('backward_schedule')!=execution['backward_schedule']:raise ValueError('P2 numerical/schedule gates failed')
            counts=c.get('candidate_counts')
            if not isinstance(counts,list) or len(counts)!=4 or any(type(v) is not int or v<0 for v in counts) or c.get('oracle_counts')!=counts:raise ValueError('P2 counts invalid')
            if counts!=expected_counts(execution,rank,c['case']):raise ValueError('Deterministic case receive counts differ')
            rows=sum(counts)
            shapes=dict(x=[b,h],xe=[rows,h],w1=[4,2*i,h],gu=[rows,2*i],a=[rows,i],w2=[4,h,i],z=[rows,h],y=[b,h])
            if c.get('candidate_shapes')!=shapes:raise ValueError('P2 full shapes invalid')
            sizes=dict(Y=b*h,dx=b*h,dp=b*k,dW1=8*i*h,dW2=4*h*i,leaf_dW1=8*i*h,leaf_dW2=4*h*i)
            dtypes=dict(Y='torch.bfloat16',dx='torch.bfloat16',dp='torch.float32',dW1='torch.float32',dW2='torch.float32',leaf_dW1=execution['weight_dtype'],leaf_dW2=execution['weight_dtype'])
            if execution['shared']:
                sizes.update(dSW1=2*i*h,dSW2=i*h,leaf_dSW1=2*i*h,leaf_dSW2=i*h)
                dtypes.update(dSW1='torch.float32',dSW2='torch.float32',leaf_dSW1=execution['weight_dtype'],leaf_dSW2=execution['weight_dtype'])
            if c.get('dtypes')!=dtypes or set(c.get('metrics',{}))!=set(sizes):raise ValueError('P2 metrics/dtypes incomplete')
            for name,n in sizes.items():metric_passes(c['metrics'][name],dtypes[name],n)
        cleanup=r.get('case_cleanup',[])
        if len(cleanup)!=len(cases):raise ValueError('Missing per-case cleanup')
        for j,c in enumerate(cleanup):
            stage=f"P2/case-close/{j}/{execution['cases'][j]}"
            if c.get('case_index')!=j or c.get('case')!=execution['cases'][j] or c.get('stage')!=stage or c.get('all_ranks_closed') is not True or c.get('all_ranks_case_passed') is not True or 'close_error' not in c or 'body_error' not in c or c.get('close_error') or c.get('body_error') or c.get('host_exchange_error'):raise ValueError('Case cleanup failure')
            peers=c.get('peer_status',[])
            if [p.get('rank') for p in peers]!=list(range(world)) or any('close_error' not in p or 'body_error' not in p or p.get('close_error') or p.get('body_error') or p.get('case_index')!=j or p.get('case')!=c['case'] or p.get('stage')!=stage for p in peers):raise ValueError('Peer case cleanup failure')
        for field,error_key in (('peer_before_PG_teardown','error'),('peer_exit_status','errors'),('final_identity_peer_status','error')):
            peers=r.get(field,[])
            if [p.get('rank') for p in peers]!=list(range(world)) or any(error_key not in p or p.get(error_key) for p in peers):raise ValueError('Host/PG/final identity peer agreement incomplete')
        evidence.append(rank)
    return evidence

def finish_identity_host(host,report,collector,initial):
    error=None
    try:
        current=collector();report['identity_final']=current
        report['identity_difference']=final_identity(initial,current,importlib.import_module('provenance'))
    except Exception as exc:error=repr(exc);report['all_passed']=False;report['final_identity_error']=error
    try:
        peers=host.all_gather(dict(rank=host.rank(),error=error),'P2/final-identity');report['final_identity_peer_status']=peers
        if [p.get('rank') for p in peers]!=list(range(host.size())) or any(p.get('error') for p in peers):raise RuntimeError('P2 peer identity failed')
    except Exception as exc:report['all_passed']=False;report['final_identity_host_error']=repr(exc);return RuntimeError(error or repr(exc))
    return RuntimeError(error) if error else None

def collective_checked(torch,dist,action,label):
    value=None;error=None
    try:value=action()
    except Exception as exc:error=repr(exc)
    status=torch.tensor([int(error is None)],device=f'npu:{torch.npu.current_device()}',dtype=torch.int32)
    dist.all_reduce(status,op=dist.ReduceOp.MIN)
    if not status.item():raise RuntimeError(label+': '+(error or 'peer failed; no SHMEM engine permitted'))
    return value
