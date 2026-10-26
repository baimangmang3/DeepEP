"""One bounded read-only snapshot of the approved single diagnostic run."""
import pathlib,json,sys,shlex
from training_native_round import SshBackend,T,parallel,HOSTS
TAG='20261026_0310_p1_shared_once';STAGE='p1-host-v2-main-shared-bf16-balanced'
RUN=T+'/runs/'+TAG+'/'+STAGE;R=T+'/validation-p1-main-shared-20261026'
SNAP=sys.argv[1];CAPTURE='--capture' in sys.argv;GATE='--gate' in sys.argv
assert SNAP and all(v.isalnum() or v in '_-' for v in SNAP)
OUT=pathlib.Path('outputs/training_results_'+TAG)/STAGE;OUT.mkdir(parents=True,exist_ok=True)
x=SshBackend();x.connect()
try:
 def one(h):
  n=HOSTS.index(h);m='rack05os7' if n==0 else 'rack05os8';dest=OUT/m;dest.mkdir(exist_ok=True)
  if CAPTURE:
   cmd='nice -n 10 timeout 60 /root/miniconda3/envs/zzy/bin/python '+shlex.join([R+'/scripts/capture_live_workers.py',RUN,RUN+'/live/'+SNAP+'.json'])
   _,o,e=x.clients[h].exec_command(cmd,timeout=75);stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status()
   (dest/(SNAP+'-capture-result.json')).write_text(json.dumps({'rc':rc,'stdout':stdout,'stderr':stderr},indent=2))
  r=x.python(h,TAG,f'''import pathlib,json,time,os
p=pathlib.Path({RUN!r});rows=[];files=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  env=dict(v.split(b'=',1) for v in (q/'environ').read_bytes().split(b'\\0') if b'=' in v)
  if env.get(b'RUN_TAG')!={TAG!r}.encode():continue
  a=[v.decode(errors='replace') for v in (q/'cmdline').read_bytes().split(b'\\0') if v]
  if not a or not any(v=={R!r}+'/scripts/run_dispatch_host_trace_checked.sh' or v=={T!r}+'/candidate-p1-host-v2/p1_pointwise/validate_training_p1.py' or v=='torch.distributed.run' for v in a):continue
  rows.append({{'pid':int(q.name),'pgid':os.getpgid(int(q.name)),'argv':a,'rank':env.get(b'RANK',b'').decode(),'stat':(q/'stat').read_text(),'wchan':(q/'wchan').read_text()}})
 except OSError:pass
for pattern in ('rank*.json','host-trace/*.jsonl','logs/*','supervisor*','.node*/identity.json','live/{SNAP}.json'):
 for f in p.glob(pattern):
  if f.is_file():files.append({{'path':str(f.relative_to(p)),'bytes':f.stat().st_size}})
rc={{}}
for f in (p/'logs/node{n}.rc',p/'supervisor-node{n}.rc'):
 rc[f.name]=f.read_text().strip() if f.exists() else None
log=p/'logs/node{n}.log'
tail=log.read_bytes()[-48000:].decode(errors='replace') if log.exists() else ''
traces=[]
for f in sorted((p/'host-trace').glob('rank*.jsonl')):
 vals=[json.loads(v) for v in f.read_text().splitlines()]
 traces.append({{'file':f.name,'records':len(vals),'last':[{{k:v.get(k) for k in ('event','call_sequence','case_name','phase','host_time_ns','exception_message')}} for v in vals[-2:]]}})
print(json.dumps({{'host':{m!r},'time_ns':time.time_ns(),'processes':rows,'rc':rc,'files':files,'log_tail':tail,'trace_summary':traces}}))''')
  (dest/(SNAP+'-state.json')).write_text(json.dumps(r,indent=2))
  with x.clients[h].open_sftp() as s:
   for f in r['files']:
    rel=f['path']
    if rel.startswith('logs/') or rel.startswith('supervisor') or rel.startswith('.node') or rel.startswith('rank') or rel.startswith('host-trace/') or rel=='live/'+SNAP+'.json':
     target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
     try:s.get(RUN+'/'+rel,str(target))
     except FileNotFoundError:pass
  return r
 rows=parallel(one)
 if GATE:
  gates=parallel(lambda h:x.gate(h,TAG));(OUT/(SNAP+'-occupancy.json')).write_text(json.dumps(gates,indent=2))
  p=pathlib.Path('outputs/training_device_progress.json');d=json.loads(p.read_text(encoding='utf-8-sig'));d['latest_occupancy']={'source':str(OUT/(SNAP+'-occupancy.json')),'observations':[{k:g['observation'][k] for k in ('host','time_utc','free_cards','npu_smi_rc')}|{'own_device_processes':g['owners']['own_device_processes'],'foreign_device_processes':g['owners']['foreign_device_processes']} for g in gates.values()]};p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
  print(json.dumps({'gate':{h:{'time':g['observation']['time_utc'],'free':g['observation']['free_cards'],'own':len(g['owners']['own_device_processes']),'foreign':len(g['owners']['foreign_device_processes'])} for h,g in gates.items()}}))
 print(json.dumps({h:{'time_ns':r['time_ns'],'own_processes':len(r['processes']),'rc':r['rc'],'trace_records':sum(v['records'] for v in r['trace_summary']),'trace_last':r['trace_summary'][-1:] ,'tail':r['log_tail'][-3000:]} for h,r in rows.items()}))
finally:x.close()
