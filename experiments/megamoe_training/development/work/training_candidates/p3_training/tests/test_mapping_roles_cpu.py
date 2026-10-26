"""Current package namespace vs historical standalone exact-path evidence; no native import."""
import ast,copy,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
import candidate_provenance as P

def run():
    checks=[];maps_by_host=[];artifacts=[]
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Expected rejection')
    for host in ('rack05os7','rack05os8'):
        file=WORK.parent/'outputs/p2_qualification_deployment'/host/'collision-cpu.json'
        data=json.loads(file.read_bytes().decode('utf-8'));raw=data['raw_mapping_result'];before=copy.deepcopy(raw)
        group=next(iter(data['collision_groups'].values()));roots={Path(v['path']).parent.name:Path(v['path']).parent for v in group}
        paths=P.checked_maps(raw,roots);canonical=P.canonical_maps(paths)
        assert len(paths)==len(raw['libraries']) and raw==before
        for item in group:
            package=Path(item['path']).parent.name;role=f"package/{package}/{Path(item['path']).name}"
            assert canonical[role]=={k:item[k] for k in ('bytes','sha256')}
            assert all(paths[item['path']][k]==item[k] for k in ('path','device','inode','map_inode','map_entries'))
        maps_by_host.append(canonical);artifacts.append(str(file))
        checks.append('real_dual_C_complete_FD_preserved:'+host)
        moved=copy.deepcopy(raw);newroots={name:Path('/other-host/python')/name for name in roots}
        for v in moved['libraries']:
            for name,old in roots.items():
                if Path(v['path']).is_relative_to(old):v['path']=str(newroots[name]/Path(v['path']).relative_to(old));break
        assert P.canonical_maps(P.checked_maps(moved,newroots))==canonical
        checks.append('current_package_roots_relocated_same_content:'+host)
        reject(lambda:P.checked_maps(raw));checks.append('current_unknown_dual_C_without_roots_reject:'+host)
        unknown=copy.deepcopy(raw);item=copy.deepcopy(group[0]);item['path']='/unclassified/'+Path(item['path']).name;unknown['libraries'].append(item)
        reject(lambda:P.checked_maps(unknown,roots));checks.append('current_unknown_third_C_reject:'+host)
        historical=P.historical_maps(raw)
        assert len(historical)==len(raw['libraries']) and all('logical_role' not in v for v in historical.values())
        checks.append('historical_reader_keeps_absolute_FD_objects_without_inventing_roots:'+host)
        selected=data['base_identity']['diagnostic_paths']['native.deepep_core']
        assert paths[selected]['logical_role'].startswith('package/deep_ep/')
    assert maps_by_host[0]==maps_by_host[1];checks.append('real_hosts_same_canonical_maps_different_physical_inodes')
    # Original reviewed P3 standalone report remains valid with no package-root metadata.
    unit_path=WORK.parent/'outputs/training_results_20261025_1413/p3-up-swiglu-unit/rack05os7/up-swiglu-results.json'
    unit=json.loads(unit_path.read_bytes().decode('utf-8'));audit=json.loads((unit_path.parent.parent/'unit-audit.json').read_bytes().decode('utf-8'))
    historical=P.historical_maps(unit['identity_final']['mapped'])
    up=next(v['sha256'] for name,v in historical.items() if Path(name).name=='libmegamoe_up_swiglu.so')
    cat=next(v['sha256'] for name,v in historical.items() if Path(name).name=='libmegamoe_training.so')
    e=dict(P.FIXED,world=16,shape=dict(B=8192,H=6144,I=2048,E=64,K=8),seed=4317,shared=False,
        weight_dtype='torch.bfloat16',cases=['balanced'],oracle_chunk=1024,max_reference_gib=12.,limits=P.LIMITS)
    assert all('diagnostic_package_roots' not in unit[stage] for stage in ('identity_initial','identity_final'))
    P.require_standalone(unit,ROOT.parent/'p3_up_swiglu',up,cat,e);P.require_standalone_audit(audit,unit_path,up)
    checks.append('real_legacy68_and_exact_path_release_audit_accepted_no_roots')
    changed=copy.deepcopy(audit);changed['maps']['identity_final']['candidate_library']['path']='/wrong/path/libmegamoe_up_swiglu.so'
    reject(lambda:P.require_standalone_audit(changed,unit_path,up));checks.append('historical_audit_wrong_selected_exact_path_reject')
    for basename in ('libmegamoe_up_swiglu.so','libmegamoe_training.so'):
        changed=copy.deepcopy(unit)
        for stage in ('identity_initial','identity_final'):
            entries=changed[stage]['mapped']['libraries'];item=copy.deepcopy(next(v for v in entries if Path(v['path']).name==basename))
            item['path']='/ambiguous/'+basename;entries.append(item)
        reject(lambda:P.require_standalone(changed,ROOT.parent/'p3_up_swiglu',up,cat,e))
        checks.append('historical_selected_same_basename_ambiguity_reject:'+basename)
    # New full-layer final identity uses stable roles, independently of the old raw report representation.
    spec=importlib.util.spec_from_file_location('frozen_base_for_p3_namespace',WORK/'DeepEP/experiments/megamoe_training/tests/provenance.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    bc=dict(files=[dict(role=r,bytes=1,sha256='a'*64) for r in sorted(base.REQUIRED_ROLES)],
        versions=dict(torch='fixture',torch_npu='fixture'),precision=dict(matmul_allow_hf32=False,allow_internal_format=False))
    b=dict(schema=1,content=bc,content_digest=base.canonical_digest(bc))
    roles={f'{prefix}/{name}' for prefix,files in (('p3',P.P3),('p1',P.P1),('up',P.UP_PACKAGE)) for name in files}|{'native.catlass','native.pointwise','native.up_swiglu'}
    c=dict(base_identity=b,execution=e,files=[dict(role=r,bytes=1,sha256='a'*64) for r in sorted(roles)],standalone=dict(report_sha256='a'*64),mapped_libraries=maps_by_host[0])
    def sealed(c):return dict(schema=P.SCHEMA,content=c,content_digest=P.digest(c))
    initial=sealed(copy.deepcopy(c));P.validate_identity(initial,base)
    target=next(k for k in c['mapped_libraries'] if k.startswith('package/torch_npu/') and '/_C.' in k)
    for change in ('replaced','removed'):
        final=copy.deepcopy(c)
        if change=='replaced':final['mapped_libraries'][target]['sha256']='b'*64
        else:final['mapped_libraries'].pop(target)
        reject(lambda:P.final_identity(initial,sealed(final),base));checks.append('full_final_extension_'+change+'_reject')
    final=copy.deepcopy(c);final['mapped_libraries']['loaded/liblazy.so']=dict(sha256='b'*64,bytes=7)
    final['base_identity']['content']['files'].append(dict(role='loaded_dependency/liblazy.so',sha256='b'*64,bytes=7))
    final['base_identity']['content_digest']=base.canonical_digest(final['base_identity']['content'])
    P.final_identity(initial,sealed(final),base);checks.append('full_final_only_lazy_added_accept')
    old=copy.deepcopy(initial);old['schema']='p3-training-1'
    reject(lambda:P.validate_identity(old,base));checks.append('old_full_training_schema_reject')
    # The actual policy bodies were copied, not imported at runtime from P2.
    p2=ast.parse((ROOT.parent/'p2_backward_overlap/candidate_provenance.py').read_text(encoding='utf-8'))
    p3=ast.parse((ROOT/'candidate_provenance.py').read_text(encoding='utf-8'))
    for name in ('valid_map_role','imported_package_roots','checked_maps','canonical_maps'):
        a=next(n for n in p2.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in p3.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    checks.append('current_namespace_policy_matches_reviewed_P2_without_runtime_import')
    assert not any(n=='torch' or n.startswith(('torch.','torch_npu','deep_ep')) for n in sys.modules)
    return dict(passed=True,checks=len(checks),details=checks,schema=P.SCHEMA,archives=artifacts,
        scope='Offline archived FD fixtures and identity policy only; no native collector/import/device/68case or CPU16 math rerun',
        limitation='Historical standalone exact paths remain historical evidence; no actual package roots inferred or supplied to that report. Revised full-layer collection still needs separate CPU import preflight.')

if __name__=='__main__':
    result=run();(ROOT/'tests/mapping_roles_cpu_results.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'));print(json.dumps(result))
