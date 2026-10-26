"""CPU evidence completion: exact own PID/start release; no device launch or reset."""
import pathlib,json,sys,hashlib
from training_native_round import SshBackend,parallel,HOSTS
TAG,STAGE,KEY=sys.argv[1:4]
OUT=pathlib.Path('outputs/training_results_'+TAG)/STAGE
A=json.loads((OUT.parent/'p1-stage-accuracy-audit.json').read_text(encoding='utf-8'));assert A['complete'] and A['all_rank_accuracy_passed']
x=SshBackend();x.connect()
try:
 def one(h):
  machine='rack05os7' if h==HOSTS[0] else 'rack05os8';d=OUT/machine
  early=json.loads((d/'early-state.json').read_text(encoding='utf-8'));ledger={str(v['pid']):v['stat'].rsplit(')',1)[1].split()[19] for v in early['processes']}
  result=x.python(h,TAG,f'''import pathlib,json,time
rows=[]
for pid,start in {ledger!r}.items():
 try:
  st=(pathlib.Path('/proc')/pid/'stat').read_text().rsplit(')',1)[1].split();rows.append({{'pid':int(pid),'saved_start':start,'actual_start':st[19],'state':st[0],'own_gone':st[19]!=start}})
 except FileNotFoundError:rows.append({{'pid':int(pid),'saved_start':start,'own_gone':True}})
print(json.dumps({{'time_ns':time.time_ns(),'all_own_released':all(v['own_gone'] for v in rows),'processes':rows}}))''')
  assert result['all_own_released'];(d/'own-release-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
 releases=parallel(one)
finally:x.close()
metrics=0;ranks=[]
for rank in range(16):
 host='rack05os7' if rank<8 else 'rack05os8';d=OUT/host;r=json.loads((d/f'rank{rank}.json').read_text(encoding='utf-8'))
 metrics+=sum(len(c['metrics']) for c in r['cases'])
 assert all(c['candidate_counts']==[16384]*4 and c['candidate_shapes']['xe']==[65536,6144] for c in r['cases'])
 ranks.append({'rank':rank,'report_sha256':hashlib.sha256((d/f'rank{rank}.json').read_bytes()).hexdigest()})
for host,n in [('rack05os7',0),('rack05os8',1)]:
 assert int((OUT/host/f'supervisor-node{n}.rc').read_text())==0
 assert not json.loads((OUT/host/'final-state.json').read_text(encoding='utf-8'))['processes']
expected=176 if A['shared'] else 112;assert metrics==expected
g=json.loads((OUT/'final-occupancy.json').read_text(encoding='utf-8'));occ={h:{'time_utc':v['observation']['time_utc'],'free_cards':v['observation']['free_cards'],'processes':v['owners']['processes']} for h,v in g.items()}
result={'run_tag':TAG,'stage':STAGE,'strict_accuracy_passed':True,'metric_checks':metrics,'counts65536_per_rank_verified':True,'counts16384_per_expert_verified':True,'shape':A['shape'],'shared':A['shared'],'weight_dtype':A['weight_dtype'],'node_rc':A['node_rc'],'supervisor_rc':{'rack05os7':0,'rack05os8':0},'all_own_released':True,'ledger_processes':sum(len(v['processes']) for v in releases.values()),'identity':A['identity'],'base_identity':A['base_identity'],'post_release_occupancy':occ,'signals_sent':False,'timing_performed':False,'root_cause_repaired':False,'rank_reports':ranks,'native_FD_evidence':'See own live snapshots; do not infer unobserved post-PG FD identities'}
path=OUT/'control-completion-audit.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
p=pathlib.Path('outputs/training_device_progress.json');d=json.loads(p.read_text(encoding='utf-8-sig'));d[KEY].update(finished=True,pending_single_control=False,strict_accuracy_passed=True,metric_checks=metrics,audit=str(path),all_own_released=True,node_rc=A['node_rc']);d.update(device_tasks_currently_running=False,active=None,active_stage=None,automatic_device_queue=[],status=KEY+'_passed_released');p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result));print('audit_sha256',hashlib.sha256(path.read_bytes()).hexdigest())
