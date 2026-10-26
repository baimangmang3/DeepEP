import pathlib,json,hashlib
out=pathlib.Path('outputs/p4_component_v2_control_deployment');r=json.loads((out/'CPU-deployment-summary.json').read_text());assert r['passed'] and len(r['hosts'])==2
assert [v['returncode'] for v in r['plans']['rows']]==[0,0]
for v in r['plans']['rows']:
 state=[json.loads(line) for line in v['stdout'].splitlines() if line.startswith('{')][-1]
 assert state['CPU_state_passed'] and not any(state['before'].values()) and not any(state['after'].values())
(out/'README.md').write_text('''P4 v2 single-worker control CPU deployment only

Both hosts created training/validation-p4-component-v2-control-20261026 and copied exactly manifest.json plus manifest.files under scripts. Manifest SHA04d326db4f19b535231ce15e4360658810003b25c06251b8f2d885f2ea7ce363, controllerc9f86760c319a818cc209179fb93a390f0a5f2bed3982106a7192cd2c9247da4. Byte verification and start_supervisor.sh bash-n passed on both hosts; only27 additionally verified deployed component package/v2 SO/activation via --worker-source. Original package/build directories untouched.

Host27 ran worker_recorder.py and supervisor.py once each with no --execute, both actual process rc0. Raw commands/stdout/stderr and state are retained. Torch/PTA absent and P4 SO unmapped before/after. These are plan-only program invocations, no worker created, NPU/PG initialized, selected native load, compilation or occupancy query. Remote controller was not executed; no SSH backend/credentials or device authorization file was copied/created. Component package default plan and prior matrices were not repeated.

This consumes only the CPU preparation permission. P4 device/performance/full-training qualification remains false. The unique05:56:51.864556 EP16 serial check remains higher priority and unchanged.
''')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
(out/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'},indent=2))
print(json.dumps({'CPU_verified':True,'plans_real_rc':[0,0],'summary_sha256':sha(out/'CPU-deployment-summary.json'),'device_authorized':False}))
