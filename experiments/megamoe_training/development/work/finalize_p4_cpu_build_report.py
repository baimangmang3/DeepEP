import pathlib,json,hashlib,re,datetime
OUT=pathlib.Path('outputs/p4_compute_wave_build');load=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
r=load(OUT/'latest-build-status.json');pre=load(OUT/'preflight.json');diag=load(OUT/'compiler-diagnostics.json')
assert r['manifest_sha256']=='deec8aea2e29c9a365fbbc6ddbc51fe3a64328db4173f5b9b72f97ead3e3f9d9'
assert r['sources']['native/wave_compute.cpp']['sha256']=='fd605673edab709f39e4d8399af3c157f42688ccfba04275d27b1b2a21974a61'
assert all(v==0 for v in r['rc'].values()) and not r['process']['same_PID_start_alive']
assert sha(OUT/'libmegamoe_wave_compute.so')==r['SO']['sha256']
elf=(OUT/'readelf.txt').read_text();ldd=(OUT/'ldd.txt').read_text();build=(OUT/'build.log').read_text()
assert 'AArch64' in elf and 'libmegamoe_wave_compute.so' in elf and 'not found' not in ldd
assert '-O2' in build and '-ffast-math' not in build
exports=sorted({line.split()[-1] for line in elf.splitlines() if ' GLOBAL ' in line and ' UND ' not in line and 'megamoe_train_wave_compute' in line})
assert exports==['megamoe_train_wave_compute','megamoe_train_wave_compute_backend']
imports=sorted({fields[7] for line in elf.splitlines() if (fields:=line.split()) and len(fields)>=8 and fields[0].rstrip(':').isdigit() and fields[4]=='GLOBAL' and fields[6]=='UND'})
warning_categories={name:[v for v in diag['warnings'] if token in v['text']] for name,token in [('deprecated_DCache_API','-Wdeprecated-declarations'),('vector_loop_uint16','-Wcce-compat'),('ignored_GM_ADDR_cast_attribute','-Wignored-attributes')]}
assert {k:len(v) for k,v in warning_categories.items()}=={'deprecated_DCache_API':6,'vector_loop_uint16':1,'ignored_GM_ADDR_cast_attribute':7}
audit={'CPU_compile_passed':True,'configure_rc':0,'build_rc':0,'readelf_rc':0,'ldd_rc':0,'build_process_PID_start_gone':True,'manifest_sha256':r['manifest_sha256'],'source_inventory':r['sources'],'compiler':pre['compiler'],'catlass_head':pre['catlass_head'],'SO':r['SO'],'ELF_exports':exports,'ELF_strong_undefined_import_names':imports,'ldd_missing_dependencies':[],'scope_of_symbol_check':'Saved ELF imported/exported symbols and resolved dependency paths only; no dlopen/CDLL, relocation execution or kernel test performed.','warnings':warning_categories,'Torch_imported':False,'native_loaded':False,'NPU_queried_or_initialized':False,'training_qualified':False,'device_authorized':False,'source_modified':False}
(OUT/'CPU-build-audit.json').write_text(json.dumps(audit,indent=2))
(OUT/'README.md').write_text(f'''P4 compute-wave isolated CPU native build

Configure/build/overall/readelf/ldd true rc all0; recorded build PID4035691/start23659991 is gone. Exactly6 approved source files from manifest {r['manifest_sha256']} were deployed and rehashed after compilation; native fd605673edab709f39e4d8399af3c157f42688ccfba04275d27b1b2a21974a61 unchanged. Original P0-P3 and new validator files were not included or changed.

Remote root /home/zzy/deploy/megamoe-20261024/training/candidate-p4-compute-wave-build-20261026. SO build/libmegamoe_wave_compute.so,606568bytes,SHA {r['SO']['sha256']}; identical local SO retained. Actual compiler path/version/SHA and clean CATLASS2b4bd7c identity are in preflight.json. O2 and explicit CATLASS_ROOT,32AIC/64AIV MIX source declaration, nofastmath. Exact configure/build commands are in build-script.sh and raw verbose logs,120s configure/600s build+30kill, nice10/jobs2.

All14 compiler warning lines are preserved:6 deprecated DCache read/write declarations,1 vector-loop condition uint16_t compatibility warning,7 cce_global ignored attributes on GM_ADDR host casts at cpp204/205. The MIX entry declaration is at cpp193; ignored diagnostics identify the cast sites. No warning was suppressed or source changed. Native execution questions, including vector-loop and fence/core progress behavior, remain unvalidated.

ELF AArch64 exports megamoe_train_wave_compute and megamoe_train_wave_compute_backend. ldd resolves all listed dependencies, CANN dependencies from formal9.2 root, system driver libascend_hal and original conda/system C++ dependencies as recorded. Strong UND imports retained in CPU-build-audit.json are ELF import declarations; this check did not load the library or prove runtime symbol/kernel execution. No Torch import, CDLL/dlopen, NPU query, initialization, unit test, timing or training/performance qualification. Parent owns any next device decision.
''')
(OUT/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'},indent=2))
p=pathlib.Path('outputs/training_device_progress.json');s=load(p);s['cpu_p4_build'].update(active=False,CPU_compile_passed=True,source_unchanged=True,SO_sha256=r['SO']['sha256'],build_audit=str(OUT/'CPU-build-audit.json'),device_authorized=False,native_loaded=False)
s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(s,indent=2))
print(json.dumps({'CPU_build_passed':True,'warning_categories':{k:len(v) for k,v in warning_categories.items()},'SO_sha256':r['SO']['sha256'],'audit_sha256':sha(OUT/'CPU-build-audit.json'),'device_authorized':False}))
