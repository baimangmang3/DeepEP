"""Read-only bounded observations for the one separately authorized EP16 run."""
import pathlib,json,argparse
from training_native_round import SshBackend,T,HOSTS,parallel
p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('name');p.add_argument('--capture',action='store_true');p.add_argument('--gate',action='store_true');a=p.parse_args()
assert a.tag and all(c.isalnum() or c in '_-' for c in a.tag)
STAGE='p2-ep16-bf16-serial';R=T+'/validation-p2-ep16-serial-hostname-20261026';RUN=T+'/runs/'+a.tag+'/'+STAGE
OUT=pathlib.Path('outputs/training_results_'+a.tag)/STAGE;OUT.mkdir(parents=True,exist_ok=True)
x=SshBackend();x.connect()
try:
 def one(host):
  node=HOSTS.index(host);machine='rack05os7' if node==0 else 'rack05os8';d=OUT/machine;d.mkdir(exist_ok=True)
  state=x.python(host,'observe_'+a.tag,f'''import pathlib,json,time,subprocess
run=pathlib.Path({RUN!r});rows=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\\0') if b'=' in v)
  if env.get(b'RUN_TAG')!={a.tag!r}.encode():continue
  fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  rows.append({{'pid':int(p.name),'ppid':int(fields[1]),'pgid':int(fields[2]),'start':fields[19],'state':fields[0],'argv':[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\\0') if v],'wchan':(p/'wchan').read_text()}})
 except OSError:pass
status={{}}
for rel in ({('logs/node'+str(node)+'.rc')!r},{('supervisor-node'+str(node)+'.rc')!r}):
 p=run/rel;status[rel]=int(p.read_text()) if p.exists() else None
capture=None
if {a.capture!r} and run.exists():
 c=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',{R!r}+'/scripts/capture_live_workers.py',str(run),str(run/{('live-workers-'+a.name+'.json')!r})],capture_output=True,text=True,timeout=75)
 capture={{'rc':c.returncode,'stdout':c.stdout,'stderr':c.stderr}}
print(json.dumps({{'time_ns':time.time_ns(),'own':rows,'rc':status,'capture':capture}}))''')
  (d/(a.name+'-state.json')).write_text(json.dumps(state,indent=2))
  with x.clients[host].open_sftp() as ftp:
   paths=[f'logs/node{node}.log',f'logs/launch-node{node}.json',f'logs/node{node}.rc',f'supervisor-node{node}.rc',f'supervisor-node{node}.log','live-workers-'+a.name+'.json','own-child-first-cleanup.json']+[f'rank{rank}.json' for rank in range(node*8,node*8+8)]
   for rel in paths:
    target=d/rel;target.parent.mkdir(parents=True,exist_ok=True)
    try:ftp.get(RUN+'/'+rel,str(target))
    except FileNotFoundError:pass
  return state
 states=parallel(one)
 if a.gate:
  gates=parallel(lambda h:x.gate(h,'observe_'+a.tag));(OUT/(a.name+'-occupancy.json')).write_text(json.dumps(gates,indent=2))
  for h,g in gates.items():states[h]['gate_summary']={'rc':g['observation']['npu_smi_rc'],'free':g['observation']['all_eight_free'],'own_device':len(g['owners']['own_device_processes']),'foreign_device':len(g['owners']['foreign_device_processes'])}
 print(json.dumps({h:{'own':len(v['own']),'rc':v['rc'],'capture':v['capture'],'gate':v.get('gate_summary')} for h,v in states.items()}))
finally:x.close()
