"""Stdlib actual EP16 accuracy/FD/exit audit, no device or live process actions."""
import pathlib,json,hashlib,importlib.util,shutil,sys
TAG=sys.argv[1];ROOT=pathlib.Path('outputs/training_results_'+TAG)/'p2-ep16-bf16-serial'
load=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module('audit_base','work/DeepEP/experiments/megamoe_training/tests/provenance.py');prov=module('audit_P2','work/training_candidates/p2_backward_overlap/candidate_provenance.py')
reports=[];sources=[]
for rank in range(16):
 p=ROOT/('rack05os7' if rank<8 else 'rack05os8')/f'rank{rank}.json';r=load(p);assert r['rank']==rank;sources.append(p);reports.append(r)
execution=reports[0]['execution'];current=reports[0]['identity_final']
assert execution['world']==16 and execution['backward_schedule']=='serial' and execution['shape']==dict(B=32,H=128,I=128,E=64,K=8) and execution['shared'] is False and execution['weight_dtype']=='torch.bfloat16'
allranks=ROOT/'all-ranks';allranks.mkdir(exist_ok=True)
for rank,p in enumerate(sources):
 dst=allranks/p.name
 if dst.exists():assert dst.read_bytes()==p.read_bytes(),'Refuse qualification file overwrite'
 else:shutil.copyfile(p,dst)
assert prov.require_reports(allranks,execution,current,base)==list(range(16))
manifest=load('outputs/p2_ep16_serial_hostname_preparation/rack05os7-diagnostic-deployment-manifest.json');pins={**manifest['source_files'],**manifest['external_native']}
p2='/home/zzy/deploy/megamoe-20261024/training/candidate-p2-qualification-schema2-20261026/p2_backward_overlap/';p1='/home/zzy/deploy/megamoe-20261024/training/candidate-p1-host-v2/p1_pointwise/'
details=[];metrics=0;maxrms=0.
for rank,r in enumerate(reports):
 for key in ('identity','identity_final'):
  ident=r[key];assert ident['schema']=='p2-training-2';by={v['logical_role']:v for v in ident['actual_maps']};assert len(by)==len(ident['actual_maps']) and set(by)==set(ident['content']['mapped_libraries'])
  for role,item in by.items():
   assert not item.get('error') and item['bytes']==ident['content']['mapped_libraries'][role]['bytes'] and item['sha256']==ident['content']['mapped_libraries'][role]['sha256']
   dev=item['device'];assert (((dev>>8)&0xfff)|((dev>>32)&0xfffff000))==item['map_device_major'] and ((dev&0xff)|((dev>>12)&0xffffff00))==item['map_device_minor'] and item['inode']==item['map_inode']
   assert item['map_entries'] and all(m['inode']==item['inode'] and m['device_major']==item['map_device_major'] and m['device_minor']==item['map_device_minor'] for m in item['map_entries'])
  for package in ('deep_ep','torch_npu'):assert len([v for v in by if v.startswith('package/'+package+'/') and '/_C.' in v])==1
  for item in ident['content']['files']:
   role=item['role'];path=p2+role[3:] if role.startswith('p2/') else p1+role[3:] if role.startswith('p1/') else None
   if path:assert pins[path]=={k:item[k] for k in ('bytes','sha256')}
   else:assert any(pin['bytes']==item['bytes'] and pin['sha256']==item['sha256'] for pin in pins.values())
 for case in r['cases']:
  for metric in case['metrics'].values():metrics+=1;maxrms=max(maxrms,metric['RMS_relative_error'])
 details.append({'rank':rank,'initial_digest':r['identity']['content_digest'],'final_digest':r['identity_final']['content_digest'],'initial_actual_FD_objects':len(r['identity']['actual_maps']),'final_actual_FD_objects':len(r['identity_final']['actual_maps']),'identity_difference':r['identity_difference'],'raw_report_sha256':sha(sources[rank]),'all_passed':True})
assert metrics==448
node_rc={};supervisor_rc={}
for node,machine in enumerate(('rack05os7','rack05os8')):
 node_rc[machine]=int((ROOT/machine/f'logs/node{node}.rc').read_text());supervisor_rc[machine]=int((ROOT/machine/f'supervisor-node{node}.rc').read_text());assert node_rc[machine]==supervisor_rc[machine]==0
audit={'numerical_identity_lifecycle_complete':True,'all_448_passed':True,'metric_count':metrics,'maximum_RMS_relative':maxrms,'world':16,'shape':execution['shape'],'schedule':'serial','weight_dtype':execution['weight_dtype'],'shared':False,'cases':execution['cases'],'node_rc':node_rc,'supervisor_rc':supervisor_rc,'ranks':details,'final_identity_all16_equal':True,'final_identity':current['content_digest'],'qualification_source_files':{p.name:sha(p) for p in allranks.iterdir()},'scope':'Actual precision/identity/close/PG/node audit. Release and post occupancy recorded separately; no timing/physical-overlap or automatic successor.'}
(ROOT/'strict-accuracy-identity-exit-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({'all448_passed':True,'maxRMS':maxrms,'node_rc':node_rc,'supervisor_rc':supervisor_rc,'identity':current['content_digest']}))
