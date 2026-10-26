"""Local-only five-field binding/result audit; no tool execution."""
import pathlib,json,hashlib,collections,datetime
OUT=pathlib.Path('outputs/p5_identity64_tool_dependency_binding');load=lambda p:json.loads(pathlib.Path(p).read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
r=load(OUT/'full-result.json');b=load(OUT/'bindings.json');req=load(OUT/'approved-binding-request.json');cmd=load(OUT/'read-command-result.json');inputs=load(OUT/'input-hashes.json')
assert cmd['rc']==0 and r['passed'] and not r['errors'] and len(r['dependencies'])==594
assert set(b)=={'compiler','readelf','external_dependencies','device_elf_machine','device_symbols'}
assert set(b['external_dependencies'])==set(req['external_dependency_paths']) and len(b['external_dependencies'])==594
assert b['compiler']['path']==req['compiler']['path'] and b['compiler']['sha256']==req['compiler']['expected_sha256']
assert b['readelf']['path']==r['readelf']['realpath'] and pathlib.PurePosixPath(b['readelf']['path']).is_absolute()
assert b['device_elf_machine']==4137 and b['device_symbols']==['dispatch_kernel','dispatch_profile_kernel']
for path,value in b['external_dependencies'].items():
 assert set(value)=={'bytes','sha256'} and value=={k:r['dependencies'][path][k] for k in value} and type(value['bytes']) is int and value['bytes']>0 and len(value['sha256'])==64
assert sha(OUT/'bindings.json')==r['bindings_sha256']
assert sha(OUT/'approved-binding-request.json')==inputs['approved_request_sha256']=='57db77e06a1b82522e79afa633b79c67a6a77edeee92832e7d3b09004d6c47dc'
assert sha(OUT/'binding_reader.py')==inputs['reader_sha256']==r['reader_sha256']
aliases=collections.defaultdict(list)
for path,row in r['dependencies'].items():aliases[row['realpath']].append(path)
for paths in aliases.values():assert len({(b['external_dependencies'][p]['bytes'],b['external_dependencies'][p]['sha256']) for p in paths})==1
audit={'passed':True,'actual_command_rc':0,'requested_path_keys':594,'distinct_resolved_dependency_paths':len(aliases),'missing_or_read_errors':[],'five_field_schema_verified':True,'binding_sha256':sha(OUT/'bindings.json'),'compiler':b['compiler'],'readelf':b['readelf'],'remote_host':'177.22.27.1','remote_allowed_binding_root':req['remote_new_files_root'],'only_disk_bytes_read':True,'compiler_executed':False,'readelf_executed':False,'compile_objects_executed':False,'P5_source_deployed':False,'Torch_or_native_loaded':False,'NPU_or_occupancy_queried':False,'ELF_machine_and_symbols_origin':'Archived CPU ELF identifiers supplied by approved request, not newly observed device data','remote_read_duration_ns':r['end_monotonic_ns']-r['start_monotonic_ns']}
(OUT/'binding-audit.json').write_text(json.dumps(audit,indent=2))
(OUT/'README.md').write_text(f'''P5 identity64 tool/dependency disk binding

One approved bounded120s host27-only disk reader returned real rc0;594 requested dependency path keys all retain positive bytes/SHA256, no missing/read errors. Resolved aliases ({len(aliases)} distinct files) retain all original594 keys and consistent hashes. Compiler actual146056368bytes/SHAfb635ef2090ff89f81c7962399b561d617c5a86417260c662d9bc6f84016f3c9 matches recorded build. Existing PATH readelf resolved to /usr/bin/readelf,862928bytes/SHAb1ac76b13058557de6d4d3cc9c54c05dc74c7ff125f0f6b0aced9b798f9972c5. Neither executable was invoked.

bindings.json has exactly the expected5 fields: compiler, readelf, external_dependencies, device_elf_machine, device_symbols. Its SHA is {sha(OUT/'bindings.json')}. Machine4137 and dispatch_kernel/dispatch_profile_kernel are copied from the approved archived-CPU-ELF request; they are not a new device observation. The object compiler script was read locally for schema, never run or deployed.

Full per-file requested/real paths, fstat metadata and hashes are in full-result.json. Input hashes, approved request, exact reader source, real command/rc/raw stdout/stderr and local/remote clock windows are preserved. Only new remote files are under {req['remote_new_files_root']}; no credentials, P5 candidate source, Torch/native load, compilation, device or occupancy query. Binding success grants no compile/device permission. No repeat was performed.
''')
(OUT/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'},indent=2))
p=pathlib.Path('outputs/training_device_progress.json');state=load(p);state['p5_identity64_CPU_dependency_binding']={'passed':True,'actual_rc':0,'dependency_path_keys':594,'bindings':str(OUT/'bindings.json'),'bindings_sha256':sha(OUT/'bindings.json'),'audit':str(OUT/'binding-audit.json'),'compiler_or_readelf_executed':False,'P5_source_deployed':False,'device_authorized':False};state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(state,indent=2))
print(json.dumps({'passed':True,'dependencies':594,'distinct_realpaths':len(aliases),'bindings_sha256':sha(OUT/'bindings.json'),'audit_sha256':sha(OUT/'binding-audit.json'),'nothing_executed_or_queried':True}))
