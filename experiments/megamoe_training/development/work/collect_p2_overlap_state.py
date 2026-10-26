import pathlib,json,sys,argparse
from training_native_round import SshBackend,T,HOSTS,parallel
TAG='20261026_0526_p2_overlap_once';STAGE='p2-ep2-bf16-overlap';R=T+'/validation-p2-ep2-overlap-hostname-20261026';RUN=T+'/runs/'+TAG+'/'+STAGE
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--capture',action='store_true');p.add_argument('--gate',action='store_true');a=p.parse_args()
OUT=pathlib.Path('outputs/training_results_'+TAG)/STAGE;OUT.mkdir(parents=True,exist_ok=True)
x=SshBackend();x.connect()
try:
 def one(host):
  machine='rack05os7' if host==HOSTS[0] else 'rack05os8';d=OUT/machine;d.mkdir(exist_ok=True)
  state=x.python(host,'observe_'+TAG,f'''import pathlib,json,time,os,subprocess
run=pathlib.Path({RUN!r});rows=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\\0') if b'=' in v)
  if env.get(b'RUN_TAG')!={TAG!r}.encode():continue
  args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
  s=(p/'stat').read_text().rsplit(')',1)[1].split()
  rows.append({{'pid':int(p.name),'ppid':int(s[1]),'pgid':int(s[2]),'state':s[0],'start':s[19],'argv':args,'wchan':(p/'wchan').read_text()}})
 except OSError:pass
values={{}}
for rel in ('logs/node0.rc','supervisor-node0.rc'):
 p=run/rel;values[rel]=int(p.read_text()) if p.exists() else None
capture=None
if {a.capture!r} and run.exists() and {host==HOSTS[0]!r}:
 p=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',{R!r}+'/scripts/capture_live_workers.py',str(run),str(run/{('live-workers-'+a.name+'.json')!r})],capture_output=True,text=True,timeout=75)
 capture={{'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}}
print(json.dumps({{'time_ns':time.time_ns(),'run_exists':run.exists(),'own':rows,'rc':values,'capture':capture}}))''')
  (d/(a.name+'-state.json')).write_text(json.dumps(state,indent=2))
  if host==HOSTS[0]:
   with x.clients[host].open_sftp() as ftp:
    for rel in ['logs/node0.log','logs/launch-node0.json','logs/node0.rc','supervisor-node0.rc','supervisor-node0.log','rank0.json','rank1.json','live-workers-'+a.name+'.json','own-child-first-cleanup.json']:
     target=d/rel;target.parent.mkdir(parents=True,exist_ok=True)
     try:ftp.get(RUN+'/'+rel,str(target))
     except FileNotFoundError:pass
  return state
 rows=parallel(one)
 if a.gate:
  gates=parallel(lambda h:x.gate(h,'observe_'+TAG));(OUT/(a.name+'-occupancy.json')).write_text(json.dumps(gates,indent=2))
  for h,g in gates.items():rows[h]['gate_summary']={'npu_rc':g['observation']['npu_smi_rc'],'free':g['observation']['all_eight_free'],'owners':{k:len(v) for k,v in g['owners'].items() if isinstance(v,list)}}
 print(json.dumps({h:{'own_count':len(v['own']),'rc':v['rc'],'capture':v['capture'],'gate':v.get('gate_summary')} for h,v in rows.items()}))
finally:x.close()
