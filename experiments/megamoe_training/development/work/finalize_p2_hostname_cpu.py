import pathlib,json,hashlib,subprocess
ROOT=pathlib.Path('outputs/p2_ep2_serial_hostname_preparation');SRC=pathlib.Path('work/training_diagnostics/p2_ep2_serial_hostname_control')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
files={p.name:sha(p) for p in SRC.iterdir() if p.is_file()}
(ROOT/'control-file-sha256.json').write_text(json.dumps(files,indent=2))
result=subprocess.run([str(pathlib.Path(__import__('sys').executable)),str(SRC/'controller.py'),'--run-tag','CPU_hostname_plan_only'],capture_output=True,text=True)
assert result.returncode==0,result.stderr
(ROOT/'controller-plan.json').write_text(result.stdout)
help_summary=json.loads((ROOT/'deployment-cpu-help-summary.json').read_text());smoke=json.loads((ROOT/'rack05os7/cpu-worker-smoke.json').read_text());gate=json.loads((ROOT/'fresh-gate-once/snapshot.json').read_text())
assert smoke['passed'] and smoke['returncode']==0 and smoke['reaped'] and smoke['worker_pid_start_gone']
for host in ('rack05os7','rack05os8'):
 assert json.loads((ROOT/host/'help-command-result.json').read_text())['rc']==0
 report=json.loads((ROOT/host/'wrapper-help.json').read_text());assert report['passed'] and not report['npu_initialized'] and not report['PG_initialized']
m=json.loads((ROOT/'rack05os7-diagnostic-deployment-manifest.json').read_text())
(ROOT/'README.md').write_text(f'''P2 EP2 serial: finite hostname-wrapper control integration

Isolated deployment /home/zzy/deploy/megamoe-20261024/training/validation-p2-ep2-serial-hostname-20261026. Prior controls/P0/P1/P2 source and public installation remain unchanged. Production CMD changes only from python -m torch.distributed.run to same-directory pinned launcher.py; all subsequent original arguments, precision/cases/serial settings, endpoints and600s+30 watchdog remain unchanged. The exact wrapper path is recognized as an inner process before workers spawn and as a known foreign wrapper by the external gate; the current RUN is exempted. Child-first/parent-recording policy is unchanged from prior review. DNS wrapper affects only the exact8184 pinned elastic agent event-metadata hostname helper; no global resolver/hosts/library/config edits or worker/rendezvous replacement.

Exact wrapper7e15cc8ebaf480b09ab7d53185b9ad8077990a3a163adac7b7281cb4bf299e4d. Both manifests identical, SHA{sha(ROOT/'rack05os7-diagnostic-deployment-manifest.json')}; source pins now{len(m['source_files'])}, dependency pins{len(m['external_native'])}. Actual installed API SHA8184afa022897c622de9ff1b76504e2210550fa772ccbc2fba44279c913bc1fe is a prelaunch pin. New helpers and raw wrapper/probe bytes are manifest-pinned. All source/native hash checks and bash-n passed on both machines. Minimal-wrapper-integration.patch and control-file-sha256.json retain exact changes.

Actual CPU --help import: both outer process rc0, NPU/PG false, DeepEP not imported. PTA was auto-imported but NPU remained uninitialized. Inner run.main return is None in this help path; raw report rc=null denotes that function return, not missing process status. Actual process rc0 is separately saved in help-command-result.json. No previous CPU matrix rerun.

One host27 CPU standard-library worker: rc0, timeout=false, launcher reaped; worker PID/start gone. Worker rank0/world1, Torch/PTA absent. Actual preworker wrapper PID/start/PGID/argv and exact inner-selector match retained; known-wrapper foreign versus own RUN classification passed. No cleanup signal was necessary or sent. Wrapper stderr preserves hostname/source AST proof. No worker tensor, NPU or PG operation.

Fresh read-only CPU/source/native/own/known-external-launcher and all16 card snapshot recorded in fresh-gate-once/snapshot.json: all16 free/own0, no known external launcher or reserved listener at observation. No device was started. This is not a reservation or device permission. New device launch still requires parent authorization and a fresh startup gate; no automatic successor queue.
''',encoding='utf-8')
(ROOT/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'},indent=2))
print(json.dumps({'CPU_passed':True,'controller':files['controller.py'],'shell_launcher':files['run_dispatch_host_trace_checked.sh'],'manifest':sha(ROOT/'rack05os7-diagnostic-deployment-manifest.json'),'device_started':False}))
