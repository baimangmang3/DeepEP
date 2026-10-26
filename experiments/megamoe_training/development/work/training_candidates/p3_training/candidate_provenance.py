"""Independent P3 content identity and standalone precondition; stdlib until collection."""
import hashlib,importlib,json,math,sys
from collections import Counter
from pathlib import Path

SCHEMA='p3-training-2'
FIXED=dict(candidate='P3_grouped_up_swiglu_v1',backend='shmem_catlass_grouped_up_swiglu_p3',
    vector_cores=64,gemm_cores=32,schedule='serial',recompute=False,local_experts=4,mix_ratio=[1,2],fresh_flags=True)
P3=('__init__.py','training_adapter.py','candidate_provenance.py','validate_training_p3.py','host_store.py','mapped_identity.py')
P1=('__init__.py','training_adapter.py','pointwise.py','native/pointwise.cpp','CMakeLists.txt')
UP=('up_swiglu.py','contract.py','native/up_swiglu.cpp','CMakeLists.txt','mapped_identity.py','test_up_swiglu_npu.py')
UP_PACKAGE=UP+('__init__.py',)
LIMITS={n:dict(atol=.003,rtol=.03,rms_relative=.02) for n in ('Y','dx','dp','dW1','dW2')}

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def check_source_layout(frozen,root,p1,up):
    for folder,name,files in ((root,'p3_training',P3),(p1,'p1_pointwise',P1),(up,'p3_up_swiglu',UP_PACKAGE)):
        folder=Path(folder).resolve(strict=True)
        if folder.name!=name:raise ValueError('Require explicit Python package directory named '+name+'; standalone deployment root is not importable')
        missing=[f for f in files if not (folder/f).is_file()]
        if missing:raise ValueError('Incomplete '+name+' package: '+repr(missing))
    for file in ('tests/provenance.py','tests/validate_training_npu.py','megamoe_training/__init__.py',
                 'megamoe_training/training.py','megamoe_training/gemm.py','native/gemm.cpp','CMakeLists.txt'):
        if not (Path(frozen)/file).is_file():raise ValueError('Incomplete frozen P0 root: '+file)

def check_imports(frozen,root,p1,up):
    roots=[Path(p).resolve(strict=True) for p in (frozen,root,p1,up)]
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for n,a in enumerate(roots) for b in roots[n+1:]):
        raise ValueError('Explicit source roots overlap')
    frozen,root,p1,up=roots
    roles={'provenance':frozen/'tests/provenance.py','validate_training_npu':frozen/'tests/validate_training_npu.py',
        'p3_training':root/'__init__.py','p3_training.training_adapter':root/'training_adapter.py',
        'p3_training.mapped_identity':root/'mapped_identity.py',
        'p1_pointwise':p1/'__init__.py','p1_pointwise.training_adapter':p1/'training_adapter.py','p1_pointwise.pointwise':p1/'pointwise.py',
        'p3_up_swiglu':up/'__init__.py','p3_up_swiglu.up_swiglu':up/'up_swiglu.py','p3_up_swiglu.contract':up/'contract.py',
        'p3_up_swiglu.mapped_identity':up/'mapped_identity.py'}
    roles.update({'megamoe_training'+suffix:frozen/'megamoe_training'/file for suffix,file in
        (('','__init__.py'),('.training','training.py'),('.gemm','gemm.py'))})
    for name,path in roles.items():
        module=importlib.import_module(name)
        if Path(module.__file__).resolve()!=path:raise ValueError('Imported module escaped explicit root: '+name)
    return roles

def check_execution(e):
    dynamic={'world','shape','seed','weight_dtype','shared','cases','oracle_chunk','max_reference_gib','limits'}
    if set(e)!=set(FIXED)|dynamic or any(type(e[k]) is not type(v) or e[k]!=v for k,v in FIXED.items()):raise ValueError('P3 fixed execution mismatch')
    s=e['shape'];w=e['world'];cases=e['cases']
    if (type(w) is not int or w not in (2,16) or type(e['shared']) is not bool or type(e['seed']) is not int or
        e['weight_dtype'] not in ('torch.bfloat16','torch.float32') or not isinstance(s,dict) or set(s)!={'B','H','I','E','K'} or
        any(type(v) is not int or v<=0 for v in s.values()) or s['E']!=4*w or s['B']>16384 or
        s['K']>min(16,s['E']) or any(s[k]%128 or s[k]>2**20 for k in ('H','I')) or
        not isinstance(cases,list) or not cases or len(cases)!=len(set(cases)) or set(cases)-{'balanced','zero_gate','empty','skew'} or
        type(e['oracle_chunk']) is not int or e['oracle_chunk']<=0 or type(e['max_reference_gib']) not in (int,float) or
        not 0<e['max_reference_gib']<float('inf') or e['limits']!=LIMITS):raise ValueError('Malformed P3 execution')

def validate_identity(value,base):
    if not isinstance(value,dict) or value.get('schema')!=SCHEMA:raise ValueError('Independent P3 identity required')
    c=value.get('content',{})
    if value.get('content_digest')!=digest(c):raise ValueError('P3 digest mismatch')
    base.validate_identity(c.get('base_identity'));check_execution(c.get('execution',{}))
    required={f'{prefix}/{f}' for prefix,files in (('p3',P3),('p1',P1),('up',UP_PACKAGE)) for f in files}|{'native.catlass','native.pointwise','native.up_swiglu'}
    files=c.get('files',[]);roles=[v.get('role') for v in files]
    if len(roles)!=len(set(roles)) or not required<=set(roles):raise ValueError('Missing/duplicate P3 source/native roles')
    for v in files:
        if set(v)!={'role','bytes','sha256'} or type(v['bytes']) is not int or v['bytes']<=0 or not isinstance(v['sha256'],str) or len(v['sha256'])!=64 or any(ch not in '0123456789abcdef' for ch in v['sha256']):raise ValueError('Malformed P3 content item')
    if not c.get('standalone',{}).get('report_sha256') or not c.get('mapped_libraries'):raise ValueError('Missing standalone/mapped evidence')
    for role,item in c['mapped_libraries'].items():
        if not valid_map_role(role) or set(item)!={'sha256','bytes'} or type(item['bytes']) is not int or item['bytes']<=0 or not isinstance(item['sha256'],str) or len(item['sha256'])!=64 or any(ch not in '0123456789abcdef' for ch in item['sha256']):raise ValueError('Malformed P3 namespace mapped content')
    return value['content_digest']

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

def historical_maps(result):
    """Archived standalone FD snapshots: exact absolute paths only, no inferred package roots."""
    if result.get('errors'):raise ValueError('Historical actual FD mapping errors')
    paths={}
    for v in result.get('libraries',[]):
        if v.get('map_inode',0)<=0 or v.get('inode')!=v['map_inode'] or not v.get('map_entries') or v['path'] in paths:
            raise ValueError('Historical FD map missing inode/segments or duplicate exact path')
        paths[v['path']]=v
    return paths


def historical_selected(paths,basename,sha256):
    values=[v for p,v in paths.items() if Path(p).name==basename]
    if len(values)!=1 or values[0]['sha256']!=sha256:raise ValueError('Historical selected native mapping missing/ambiguous/mismatched: '+basename)
    return values[0]


def require_standalone(report,up_root,up_sha,catlass_sha,execution):
    if not isinstance(report,dict) or report.get('all_passed') is not True or report.get('error') or report.get('identity_final_error'):raise ValueError('Standalone GU/A qualification absent or failed')
    if report.get('specification',{}).get('limits')!={'atol':.003,'rtol':.03,'rms_relative':.02}:raise ValueError('Standalone precision limits changed')
    if report.get('identity_difference',{}).get('changed') or report.get('identity_difference',{}).get('removed'):raise ValueError('Standalone mapping changed during run')
    expected={f:sha(Path(up_root)/f) for f in UP}
    if report.get('source_sha256')!=expected:raise ValueError('Standalone current component source mismatch')
    final=report.get('identity_final',{});mapped=historical_maps(final.get('mapped',{}))
    initial=historical_maps(report.get('identity_initial',{}).get('mapped',{}))
    if not initial or not mapped:raise ValueError('Standalone initial/final actual identity missing')
    if any(p not in mapped or any(mapped[p][k]!=v[k] for k in ('sha256','bytes')) for p,v in initial.items()):raise ValueError('Standalone initial mapped content changed/removed')
    historical_selected(mapped,'libmegamoe_up_swiglu.so',up_sha)
    historical_selected(mapped,'libmegamoe_training.so',catlass_sha)
    cases=report.get('cases',[])
    def key(v):return (tuple(v['counts']),v['H'],v['I'],v['pattern'],v['concurrent_streams'])
    expected=[dict(counts=[n,0,1 if n else 0,n//3],H=h,I=i,pattern='random',concurrent_streams=False)
        for n in (0,1,7,8,9,17,63,64,65,127,128,129) for h,i in ((128,128),(256,256),(384,384),(6144,128),(128,2048))]
    expected.extend(dict(counts=c,H=128,I=384,pattern='random',concurrent_streams=False) for c in ([0]*4,[32]*4,[0,0,0,257],[65,1,129,0]))
    expected.extend(dict(counts=[9,0,1,65],H=128,I=128,pattern=p,concurrent_streams=False) for p in ('zeros','sentinels'))
    expected.append(dict(counts=[17,0,65,1],H=128,I=384,pattern='random',concurrent_streams=True))
    main=dict(counts=[16384]*4,H=6144,I=2048,pattern='random',concurrent_streams=False)
    actual=Counter(key(v) for v in cases)
    if actual not in (Counter(key(v) for v in expected),Counter(key(v) for v in expected+[main])) or any(v.get('passed') is not True for v in cases):raise ValueError('Incomplete/duplicated standalone case coverage')
    s=execution['shape']
    if not any(v['H']==s['H'] and v['I']==s['I'] and v['M']>0 for v in cases):raise ValueError('Standalone lacks this native H/I')
    if (s['B'],s['H'],s['I'],s['K'])==(8192,6144,2048,8) and actual[key(main)]!=1:raise ValueError('Target main requires exact65536 balanced standalone case')
    for v in cases:
        if v.get('M')!=sum(v['counts']):raise ValueError('Standalone counts/rows mismatch')
        if v.get('independent_outputs_and_flags') is not True or v.get('inputs_unchanged') is False:raise ValueError('Standalone ownership/immutability failed')
        if len(v.get('invocations',[]))!=2:raise ValueError('Missing standalone repeated/concurrent invocations')
        for invocation in v['invocations']:
            if invocation.get('flags_exact') is not True or invocation.get('passed') is not True or set(invocation.get('metrics',{}))!={'GU','A'}:raise ValueError('Standalone metrics missing')
            for name,m in invocation['metrics'].items():
                width=v['I']*(2 if name=='GU' else 1)
                numbers=('RMS_error','reference_RMS','RMS_relative_error','max_abs_error','max_relative_error_floor_1e_8')
                if (m.get('shape')!=[v['M'],width] or m.get('elements')!=v['M']*width or m.get('dtype')!='torch.bfloat16' or
                    m.get('finite') is not True or m.get('badcount')!=0 or m.get('passed') is not True or
                    any(type(m.get(k)) not in (int,float) or not math.isfinite(m[k]) or m[k]<0 for k in numbers)):
                    raise ValueError('Standalone metric shape/dtype/finite/count failed')
                rms,ref=m['RMS_error'],m['reference_RMS'];ratio=rms/max(ref,1e-12)
                if not math.isclose(m['RMS_relative_error'],ratio,rel_tol=1e-9,abs_tol=1e-12) or (ratio>.02 if ref else rms>.003):raise ValueError('Standalone fixed RMS gate failed')
    return dict(report_sha256=digest(report),component_sha256=up_sha,catlass_sha256=catlass_sha)

def require_standalone_audit(audit,report_path,up_sha):
    report=json.loads(Path(report_path).read_text());process=audit.get('process',{})
    if (audit.get('passed') is not True or audit.get('report_sha256')!=sha(report_path) or audit.get('source_sha256')!=report['source_sha256'] or
        audit.get('cases')!=len(report['cases']) or audit.get('GU_A_metric_checks')!=4*len(report['cases']) or audit.get('limits')!=report['specification']['limits'] or
        any(type(process.get(k)) is not int or process[k]!=0 for k in ('node_rc','supervisor_returncode')) or
        process.get('timeout') is not False or any(process.get(k) is not True for k in ('reaped','all_own_released')) or
        not audit.get('release_evidence') or audit.get('identity_difference',{}).get('changed') or audit.get('identity_difference',{}).get('removed')):
        raise ValueError('Standalone real process rc/release/artifact audit missing or mismatched')
    for stage in ('identity_initial','identity_final'):
        item=audit.get('maps',{}).get(stage,{}).get('candidate_library',{})
        if item.get('sha256')!=up_sha or item.get('map_inode',0)<=0 or item.get('inode')!=item['map_inode'] or not item.get('map_entries'):raise ValueError('Audit candidate actual mapping mismatch')
        selected=historical_selected(historical_maps(report[stage]['mapped']),'libmegamoe_up_swiglu.so',up_sha)
        if any(item.get(k)!=selected.get(k) for k in ('path','sha256','bytes','device','inode','map_device_major','map_device_minor','map_inode')):
            raise ValueError('Audit selected candidate does not match historical exact-path FD object')
    return dict(report_sha256=audit['report_sha256'],audit_sha256=digest(audit),process_rc=0,reaped=True,all_own_released=True,component_so_sha256=up_sha)

def collect_identity(args,root,execution,base):
    paths=check_imports(args.training_source_root,root,args.p1_source_root,args.component_source_root)
    from p1_pointwise.pointwise import TrainingPointwise
    from p3_up_swiglu.up_swiglu import GroupedUpSwiGLU
    from p3_training.mapped_identity import mapped_libraries
    pointwise=TrainingPointwise(args.pointwise_library,vector_cores=64);up=GroupedUpSwiGLU(args.up_swiglu_library)
    b=base.collect_runtime_identity();package_roots=imported_package_roots()
    mapped=checked_maps(mapped_libraries(),package_roots)
    selected=dict(pointwise=Path(pointwise.path).resolve(),up_swiglu=Path(up.path).resolve(),catlass=Path(b['diagnostic_paths']['native.catlass']).resolve())
    for path in selected.values():
        if str(path) not in mapped:raise ValueError('Required selected native library not actually mapped: '+str(path))
    # Every native object in base identity must also be bound by reliable actual mapping.
    for role,path in b['diagnostic_paths'].items():
        if role.startswith(('native.','loaded_dependency/')):
            if path not in mapped or mapped[path]['sha256']!=next(v['sha256'] for v in b['content']['files'] if v['role']==role):raise ValueError('Base library is not fstat-map bound: '+role)
    unit=json.loads(Path(args.standalone_report).read_text())
    standalone=require_standalone(unit,args.component_source_root,
        mapped[str(selected['up_swiglu'])]['sha256'],mapped[str(selected['catlass'])]['sha256'],execution)
    standalone['release_audit']=require_standalone_audit(json.loads(Path(args.standalone_audit).read_text()),
        args.standalone_report,mapped[str(selected['up_swiglu'])]['sha256'])
    expected_p0={'megamoe_training':sha(Path(args.training_source_root)/'megamoe_training/__init__.py'),
                 'megamoe_training.gemm':sha(Path(args.training_source_root)/'megamoe_training/gemm.py')}
    for stage in ('identity_initial','identity_final'):
        if {n:v.get('sha256') for n,v in unit[stage].get('p0_imported_sources',{}).items()}!=expected_p0:
            raise ValueError('Standalone actual imported P0 init/gemm source changed')
    files={f'{prefix}/{f}':Path(folder)/f for prefix,folder,items in (('p3',root,P3),('p1',args.p1_source_root,P1),('up',args.component_source_root,UP_PACKAGE)) for f in items}
    manifest=[dict(role=role,bytes=p.stat().st_size,sha256=sha(p)) for role,p in sorted(files.items())]
    manifest.extend(dict(role='native.'+role,bytes=mapped[str(path)]['bytes'],sha256=mapped[str(path)]['sha256']) for role,path in selected.items())
    canonical=canonical_maps(mapped)
    content=dict(base_identity={k:b[k] for k in ('schema','content','content_digest')},files=manifest,execution=execution,
        standalone=standalone,mapped_libraries=canonical)
    value=dict(schema=SCHEMA,content=content,content_digest=digest(content),diagnostic_import_paths={n:str(p) for n,p in paths.items()},actual_maps=list(mapped.values()),diagnostic_package_roots={k:str(v) for k,v in package_roots.items()})
    validate_identity(value,base);return value,(pointwise,up)

def final_identity(initial,current,base):
    validate_identity(initial,base);validate_identity(current,base)
    a=initial['content'];b=current['content']
    for key in ('files','execution','standalone'):
        if a[key]!=b[key]:raise ValueError('P3 content/config changed during run: '+key)
    oldbase=a['base_identity']['content'];newbase=b['base_identity']['content']
    if any(oldbase[k]!=newbase[k] for k in ('versions','precision')):raise ValueError('Base precision/version changed')
    oldfiles={v['role']:v for v in oldbase['files']};newfiles={v['role']:v for v in newbase['files']}
    if any(k not in newfiles or newfiles[k]!=v for k,v in oldfiles.items()) or any(not k.startswith('loaded_dependency/') for k in newfiles.keys()-oldfiles.keys()):raise ValueError('Base source/native content changed')
    old=a['mapped_libraries'];new=b['mapped_libraries']
    if any(k not in new or new[k]!=v for k,v in old.items()):raise ValueError('Initially mapped dependency changed/removed')
    return {'added':{k:v for k,v in new.items() if k not in old},'changed':{},'removed':{}}

def finish_identity_host(host,report,collector,initial):
    error=None
    try:
        current=collector();report['identity_final']=current
        base=importlib.import_module('provenance');report['identity_difference']=final_identity(initial,current,base)
    except Exception as exc:error=repr(exc);report['all_passed']=False;report['final_identity_error']=error
    try:
        peers=host.all_gather(dict(rank=host.rank(),error=error),'P3/final-identity')
        report['final_identity_peer_status']=peers
        if any(p['error'] for p in peers):raise RuntimeError('A peer failed final P3 identity')
    except Exception as exc:
        report['all_passed']=False;report['final_identity_host_error']=repr(exc)
        return RuntimeError(error or repr(exc))
    return RuntimeError(error) if error else None

def collective_checked(torch,dist,action,label):
    value=None;error=None
    try:value=action()
    except Exception as exc:error=repr(exc)
    flag=torch.tensor([int(error is None)],device=f'npu:{torch.npu.current_device()}',dtype=torch.int32)
    dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    if not flag.item():raise RuntimeError(label+': '+(error or 'peer failed; no SHMEM engine permitted'))
    return value
