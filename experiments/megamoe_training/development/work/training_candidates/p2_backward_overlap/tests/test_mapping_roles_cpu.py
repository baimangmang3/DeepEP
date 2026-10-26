"""Namespace regression from both actual CPU FD-map collision snapshots; no imports/native run."""
import copy,json,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import candidate_provenance as P
from test_qualification_cpu import identity,execution,BASE

def run():
    checks=[];canonical=[];archives=[]
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Expected rejection')
    for host in ('rack05os7','rack05os8'):
        file=WORK.parent/'outputs/p2_qualification_deployment'/host/'collision-cpu.json'
        r=json.loads(file.read_bytes().decode('utf-8'));raw=r['raw_mapping_result'];original=copy.deepcopy(raw)
        collision=next(iter(r['collision_groups'].values()))
        roots={Path(v['path']).parent.name:Path(v['path']).parent for v in collision}
        assert set(roots)=={'deep_ep','torch_npu'} and not raw['errors']
        paths=P.checked_maps(raw,roots);maps=P.canonical_maps(paths)
        for package in ('deep_ep','torch_npu'):
            entry=next(v for v in collision if Path(v['path']).parent.name==package)
            role=f"package/{package}/{Path(entry['path']).name}"
            assert maps[role]=={k:entry[k] for k in ('sha256','bytes')}
            assert all(paths[entry['path']][k]==entry[k] for k in ('device','inode','map_device_major','map_device_minor','map_inode','map_entries'))
        assert len(paths)==len(raw['libraries']) and raw==original
        checks.append('real_dual_C_both_preserved_with_all_FD_evidence:'+host)
        canonical.append(maps);archives.append((r,roots,paths,file))
        # Only logical root location changes; never pretend relocated entries were freshly fstat'ed here.
        moved=copy.deepcopy(raw);newroots={name:Path('/relocated-host/python')/name for name in roots}
        for entry in moved['libraries']:
            for name,oldroot in roots.items():
                if Path(entry['path']).is_relative_to(oldroot):
                    entry['path']=str(newroots[name]/Path(entry['path']).relative_to(oldroot));entry['mapped_path']=entry['path'];break
        assert P.canonical_maps(P.checked_maps(moved,newroots))==maps
        checks.append('relocated_package_roots_same_content_identity:'+host)
        reject(lambda:P.checked_maps(raw));checks.append('missing_classification_rejects_dual_C:'+host)
        unknown=copy.deepcopy(raw);entry=copy.deepcopy(collision[0]);entry['path']='/unknown/extension/'+Path(entry['path']).name
        unknown['libraries'].append(entry)
        reject(lambda:P.checked_maps(unknown,roots));checks.append('unknown_third_same_basename_rejected:'+host)
    assert canonical[0]==canonical[1] and P.digest(canonical[0])==P.digest(canonical[1])
    checks.append('actual_two_hosts_same_content_roles_different_inodes')
    # Full schema gate validates role-format; initial->final must still detect replacement/removal.
    current=identity(execution());current['content']['mapped_libraries'].update(canonical[0]);current['content_digest']=P.digest(current['content'])
    P.validate_identity(current,BASE)
    torch_role=next(k for k in canonical[0] if k.startswith('package/torch_npu/') and '/_C.' in k)
    for change in ('extension_replaced','extension_removed'):
        after=copy.deepcopy(current)
        if change=='extension_replaced':after['content']['mapped_libraries'][torch_role]['sha256']='b'*64
        else:after['content']['mapped_libraries'].pop(torch_role)
        after['content_digest']=P.digest(after['content'])
        reject(lambda:P.final_identity(current,after,BASE));checks.append('final_rejects_'+change)
    after=copy.deepcopy(current);after['content']['mapped_libraries']['loaded/libnew_lazylib.so']=dict(bytes=7,sha256='c'*64)
    after['content_digest']=P.digest(after['content']);P.final_identity(current,after,BASE)
    checks.append('only_new_stable_role_allowed_final')
    legacy=copy.deepcopy(current);legacy['schema']='p2-training-1'
    reject(lambda:P.validate_identity(legacy,BASE));checks.append('legacy_basename_schema_rejected')
    with tempfile.TemporaryDirectory() as tmp:
        roots={}
        for name in ('deep_ep','torch_npu'):
            folder=Path(tmp)/name;folder.mkdir();file=folder/'__init__.py';file.write_bytes(b'')
            roots[name]=folder.resolve(strict=True)
        modules={name:types.SimpleNamespace(__file__=str(path/'__init__.py')) for name,path in roots.items()}
        with patch.dict(sys.modules,modules):assert P.imported_package_roots()==roots
        checks.append('roots_from_actual_imported_module_file_not_basename_guess')
    # Selected DeepEP native role remains bound to its exact loader path, never the Torch_NPU _C.
    for r,roots,paths,file in archives:
        path=r['base_identity']['diagnostic_paths']['native.deepep_core']
        expected=next(v for v in r['base_identity']['content']['files'] if v['role']=='native.deepep_core')
        assert all(paths[path][k]==expected[k] for k in ('bytes','sha256'))
        assert paths[path]['logical_role'].startswith('package/deep_ep/')
    checks.append('actual_base_selected_core_exact_path_binding_preserved')
    assert not any(n=='torch' or n.startswith(('torch.','torch_npu','deep_ep')) for n in sys.modules)
    return dict(passed=True,checks=len(checks),details=checks,canonical_maps_digest=P.digest(canonical[0]),
        source_archives=[str(a[3]) for a in archives],schema=P.SCHEMA,
        scope='Offline archived actual CPU fstat-map evidence and pure policy tests; no collector rerun/native import/NPU/PG. Root relocation is identity-model simulation, not a new FD capture.')

if __name__=='__main__':
    result=run();(ROOT/'tests/mapping_roles_cpu_results.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'));print(json.dumps(result))
