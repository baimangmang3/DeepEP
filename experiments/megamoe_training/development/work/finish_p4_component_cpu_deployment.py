import pathlib,json,hashlib
out=pathlib.Path('outputs/p4_component_validation_v2_deployment');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
disk=json.loads((out/'source-and-SO-disk-verification.json').read_text());state=json.loads((out/'default-plan-state.json').read_text());plan=json.loads((out/'default-plan.json').read_text());command=json.loads((out/'default-plan-command.json').read_text())
assert disk['passed'] and len(disk['source_inventory'])==8 and command['rc']==0 and state['passed'] and len(plan['configuration']['cases'])==8
assert not any(state['before'].values()) and not any(state['after'].values())
(out/'README.md').write_text('''P4 v2 isolated component CPU assembly/default plan

Host27 package /home/zzy/deploy/megamoe-20261024/training/candidate-p4-component-validation-v2-20261026/p4_wave_compute assembled from exactly8 approved inventory entries. Runtime contract/wrapper/native CPP/CMake come from explicit v2 sources (CPP80a4c88e...); harness/FD identity helpers come from reviewed additional manifest ace5458f11ffda50a9ea1c4321923d76eae8bb9593d424e2a1b8cec3bf10a1dd. No adjacent v1 implicit source selection. Additional manifest and first-device draft retained separately; old build directories unchanged.

All8 disk source hashes and already-built v2 SO eea6c53d1a9ffd7d06aed510b9d2f80e59a9112476008a3796f25ab57fc797ac verified. Actual default-plan process rc0, raw command/stdout/stderr saved. Before/after Torch/PTA absent and selected P4 SO unmapped. No --run-npu, native load, recompile, NPU query, tensor, session or PG. These are disk/CPU-plan results, not actual mapped device-library identity or component accuracy.

The unexecuted8-case plan covers16 planned invocations (14 native,2 host-empty) with48 GU/A/Z metric summaries,6 empty. No invocations or metrics were executed. Device/throughput/full-training qualification remains false; separate root-reviewed concrete device permission required. The sole existing05:25:04.814088 EP2 overlap deadline/priority is unchanged.
''')
(out/'CPU-deployment-audit.json').write_text(json.dumps({'passed':True,'source_count':8,'plan_cases':8,'process_rc':0,'before':state['before'],'after':state['after'],'source_verification_sha256':sha(out/'source-and-SO-disk-verification.json'),'stdout_sha256':sha(out/'default-plan.stdout'),'stderr_sha256':sha(out/'default-plan.stderr'),'default_plan_command_sha256':sha(out/'default-plan-command.json'),'device_authorized':False,'component_accuracy_qualified':False},indent=2))
(out/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'},indent=2))
print(json.dumps({'passed':True,'audit_sha256':sha(out/'CPU-deployment-audit.json'),'no_device':True}))
