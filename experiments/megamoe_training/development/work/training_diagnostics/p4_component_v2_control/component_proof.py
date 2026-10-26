"""P4 component-only actual FD identity; stdlib until explicit collection."""
import hashlib,importlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCES=('__init__.py','contract.py','wave_compute.py','component_identity.py','mapped_identity.py',
         'validate_component.py','CMakeLists.txt','native/wave_compute.cpp')
SCHEMA='p4-compute-component-1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inventory():return {name:sha(ROOT/name) for name in SOURCES}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def mapped_table(raw):
    if raw.get('errors'):raise ValueError('Actual FD collector errors: '+repr(raw['errors']))
    result={}
    for v in raw.get('libraries',[]):
        if v['path'] in result or v.get('map_inode',0)<=0 or v.get('map_inode')!=v.get('inode') or not v.get('map_entries'):
            raise ValueError('Missing/ambiguous FD map object')
        result[v['path']]=v
    if not result:raise ValueError('No actual native mappings')
    return result
def selected_library(table,path,expected_sha):
    if path not in table or table[path]['sha256']!=expected_sha:raise ValueError('Actual selected P4 SO SHA missing/mismatched')
    if any(Path(p).name==Path(path).name and v['sha256']!=expected_sha for p,v in table.items()):
        raise ValueError('Conflicting actual P4 SO basename')
    return table[path]
def collect(ops,torch,torch_npu,config,hardware,expected_sha):
    from p4_wave_compute.mapped_identity import mapped_libraries
    for name,file in (('p4_wave_compute','__init__.py'),('p4_wave_compute.wave_compute','wave_compute.py'),
                      ('p4_wave_compute.contract','contract.py'),('p4_wave_compute.mapped_identity','mapped_identity.py')):
        module=importlib.import_module(name)
        if Path(module.__file__).resolve()!=ROOT/file:raise ValueError('P4 module escaped explicit package: '+name)
    raw=mapped_libraries();table=mapped_table(raw);selected=str(Path(ops.path).resolve(strict=True))
    selected_record=selected_library(table,selected,expected_sha)
    for token in ('libascendcl.so','libruntime.so'):
        if not any(Path(path).name.startswith(token) for path in table):raise ValueError('Required actual CANN runtime missing: '+token)
    c=importlib.import_module('torch_npu._C');cpath=str(Path(c.__file__).resolve(strict=True))
    if cpath not in table:raise ValueError('Actual imported torch_npu._C FD missing')
    option=torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
    if isinstance(option,bytes):option=option.decode()
    if option!='disable' or torch.npu.matmul.allow_hf32 is not False:raise ValueError('Strict precision readback differs')
    if torch.distributed.is_initialized():raise ValueError('Single-worker component test must not own a PG')
    content=dict(sources=inventory(),configuration=config,torch=str(torch.__version__),torch_npu=str(torch_npu.__version__),
        torch_npu_git=getattr(getattr(torch_npu,'version',None),'git_version',None),precision=dict(hf32=False,internal_raw=option),
        selected_SO_SHA256=expected_sha,hardware=hardware,backend='experimental-CATLASS950-BF16-512wave-GU-A-Z-v1')
    return dict(schema=SCHEMA,content=content,content_digest=digest(content),actual_maps=raw,
                selected_library=selected_record,torch_npu_C=table[cpath])
def compare(initial,final):
    if initial.get('schema')!=SCHEMA or final.get('schema')!=SCHEMA:raise ValueError('P4 independent component identity required')
    for value in (initial,final):
        if value.get('content_digest')!=digest(value['content']):raise ValueError('Identity digest corrupt')
    if initial['content']!=final['content']:raise ValueError('P4 source/config/library/framework/precision/hardware changed')
    old=mapped_table(initial['actual_maps']);new=mapped_table(final['actual_maps'])
    keys=('sha256','bytes','map_inode','map_device_major','map_device_minor')
    if any(path not in new or any(v[k]!=new[path][k] for k in keys) for path,v in old.items()):raise ValueError('Initial mapped object changed/removed')
    return dict(added=[new[p] for p in new.keys()-old.keys()],changed=[],removed=[])
def observe_final(item,collector,initial,prior_error=None):
    try:
        current=collector();item['identity_final']=current
        item['identity_difference']=compare(initial,current)
    except BaseException as exc:
        import traceback
        item.update(passed=False,identity_final_error=repr(exc),identity_final_traceback=traceback.format_exc())
        return prior_error or exc
    return prior_error
