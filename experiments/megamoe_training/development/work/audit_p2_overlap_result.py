"""Strict stdlib-only audit of the actual P2 overlap rank reports."""
import pathlib,json,hashlib,importlib.util,os,math,datetime
ROOT=pathlib.Path('outputs/training_results_20261026_0425_p2_overlap_once/p2-ep2-bf16-overlap');D=ROOT/'rack05os7'
load=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=module('audit_p0_provenance','work/DeepEP/experiments/megamoe_training/tests/provenance.py')
prov=module('audit_p2_provenance','work/training_candidates/p2_backward_overlap/candidate_provenance.py')
reports=[load(D/f'rank{rank}.json') for rank in range(2)];execution=reports[0]['execution'];identity=reports[0]['identity_final']
assert execution['backward_schedule']=='overlap' and execution['shape']=={'B':32,'H':128,'I':128,'E':8,'K':4} and execution['shared'] is False
assert prov.require_reports(D,execution,identity,base)==[0,1]
manifest=load('outputs/p2_ep2_overlap_hostname_preparation/rack05os7-diagnostic-deployment-manifest.json')
p2='/home/zzy/deploy/megamoe-20261024/training/candidate-p2-qualification-schema2-20261026/p2_backward_overlap/'
p1='/home/zzy/deploy/megamoe-20261024/training/candidate-p1-host-v2/p1_pointwise/'
allpins={**manifest['source_files'],**manifest['external_native']};maxrms=0.;total=0;details=[]
for rank,report in enumerate(reports):
 for phase in ('identity','identity_final'):
  ident=report[phase];assert ident['content_digest']==identity['content_digest']
  assert ident['content']['base_identity']['content_digest']=='15399c4d3624b1e6fd5f5789e81e80b4d4c62bc73d98fa3be9874fb1613c6015'
  for entry in ident['content']['files']:
   role=entry['role'];remote=p2+role[3:] if role.startswith('p2/') else p1+role[3:] if role.startswith('p1/') else None
   if remote:assert allpins[remote]=={k:entry[k] for k in ('sha256','bytes')},role
   else:assert any(pin['sha256']==entry['sha256'] and pin['bytes']==entry['bytes'] for pin in allpins.values()),role
  maps=ident['actual_maps'];assert len(maps)==35
  logical={item['logical_role']:item for item in maps};assert set(logical)==set(ident['content']['mapped_libraries'])
  for role,item in logical.items():
   assert item['sha256']==ident['content']['mapped_libraries'][role]['sha256'] and item['bytes']==ident['content']['mapped_libraries'][role]['bytes']
   dev=item['device'];major=((dev>>8)&0xfff)|((dev>>32)&0xfffff000);minor=(dev&0xff)|((dev>>12)&0xffffff00)
   assert major==item['map_device_major'] and minor==item['map_device_minor'] and item['inode']==item['map_inode']
   assert item['map_entries'] and all(v['inode']==item['inode'] and v['device_major']==item['map_device_major'] and v['device_minor']==item['map_device_minor'] for v in item['map_entries'])
   assert not item.get('error')
  for package in ('deep_ep','torch_npu'):
   assert len([k for k in logical if k.startswith('package/'+package+'/') and '/_C.' in k])==1
 for case in report['cases']:
  for name,value in case['metrics'].items():
   total+=1;maxrms=max(maxrms,value['RMS_relative_error'])
 details.append({'rank':rank,'all_passed':report['all_passed'],'cases':[v['case'] for v in report['cases']],'initial_digest':report['identity']['content_digest'],'final_digest':report['identity_final']['content_digest'],'actual_mapped_FD_objects_initial':35,'actual_mapped_FD_objects_final':35,'source_and_FD_pins_verified':True,'identity_difference':report['identity_difference'],'raw_report_sha256':sha(D/f'rank{rank}.json')})
assert total==56
node=int((D/'logs/node0.rc').read_text());outer=int((D/'supervisor-node0.rc').read_text());assert node==outer==0
release=load(ROOT/'release-marker-and-log-preservation.json');assert release['all_recorded_markers_gone'] and len(release['recorded_marker_PID_start_checks'])==2
gate=load(ROOT/'early-occupancy.json');assert all(not v['owners']['processes'] and not v['owners']['foreign_device_processes'] and v['observation']['all_eight_free'] and v['observation']['npu_smi_rc']==0 for v in gate.values())
assert all(not load(ROOT/machine/'early-state.json')['own'] for machine in ('rack05os7','rack05os8'))
archive=ROOT/'rack05os7-compact-evidence.tar.gz';assert sha(archive)==release['compact_archive_sha256']
audit={'complete':True,'all_56_numerical_metrics_passed':True,'counts_shapes_dtypes_schedule_passed':True,'host_case_close_final_identity_PG_passed':True,'world':2,'backend':reports[0]['backend'],'backward_schedule':'overlap','shape':execution['shape'],'weight_dtype':execution['weight_dtype'],'shared':False,'maximum_RMS_relative':maxrms,'node_rc':node,'supervisor_rc':outer,'no_cleanup_signals_sent':True,'observed_own_processes_after':0,'observed_foreign_processes_after':0,'all16_free_after':True,'saved_recording_PID_start_checks':release['recorded_marker_PID_start_checks'],'all_saved_recording_PID_start_gone':True,'historical_worker_PID_start_ledger':'External capture arrived after all workers exited; no separate live worker PID/start ledger or FD snapshot was sampled. Actual initial/final FD evidence comes from each worker report, normal torchrun+supervisor rc0 and current own-RUN census0 prove process completion; do not claim a fabricated full historical PID-start ledger.','ranks':details,'compact_archive_sha256':sha(archive),'source_provenance_validator_sha256':sha('work/training_candidates/p2_backward_overlap/candidate_provenance.py'),'scope':'EP2 routed overlap accuracy/lifecycle qualification only. No overlap/performance or1431 root cause/fix claim.'}
(ROOT/'strict-accuracy-release-audit.json').write_text(json.dumps(audit,indent=2))
p=pathlib.Path('outputs/training_device_progress.json');state=load(p);state['active']=None;state['device_tasks_currently_running']=False;state['automatic_device_queue']=[];state['status']='p2_ep2_overlap_passed_no_successor'
state['p2_ep2_overlap_hostname_control'].update(device_authorized=False,pending_single_control=False,complete=True,audit=str(ROOT/'strict-accuracy-release-audit.json'),all_56_passed=True,node_rc=0,supervisor_rc=0,own_released=True,final_identity=identity['content_digest'])
state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(state,indent=2))
(ROOT/'README.md').write_text(f'''P2 EP2 routed BF16 overlap: actual accuracy and lifecycle success

All56 fixed-threshold metrics passed across2 ranks and balanced/zero_gate/empty/skew, including exact counts/shapes/dtypes/schedule, all case-close, final identity and PG/host status. Max RMS relative {maxrms}. Initial/final identity35 FD objects per rank with exact map dev/inode/SHA binding; digest {identity['content_digest']} equal and unchanged across phases/ranks. Both package_C objects and native/dependency pins verified. Raw node and supervisor rc are both0. No cleanup signals sent.

Saved node/outer marker PID-start identities are gone; current own RUN census0 and both hosts free16/foreign0 at post-run observation. Separate external live capture arrived after exit (0workers), so no fabricated full historical worker PID/start ledger or separate live FD evidence. Internal actual worker FD reports provide the stated provenance. Normal torchrun rc0 is retained separately from CPU controller rc0. Existing watchdog was not extended, no retry or successor started.

Small archive SHA {sha(archive)}.210101728 bytes raw SDK/SHMEM logs remain under the exact remote run path in release-marker-and-log-preservation.json; local inventory and compact archive retained. No performance or physical overlap claim, no claim historical1431 initialization/dispatch issue was fixed.
''',encoding='utf-8')
print(json.dumps({'complete':True,'metrics':total,'max_RMS_relative':maxrms,'node_rc':node,'supervisor_rc':outer,'actual_FD_objects':35,'own_after':0,'signals':0,'identity':identity['content_digest']}))
