"""Read-only CPU process identity capture, filtering exact own --output run path."""
import json
from pathlib import Path
import sys
from training_native_round import SshBackend, parallel, T

tag, stage = sys.argv[1:3]
backend = SshBackend()
try:
    backend.connect()
    def capture(host):
        r = backend.python(host, tag, f'''
import pathlib,os,json,datetime
run={T!r}+'/runs/'+{tag!r}+'/'+{stage!r}
result={{'host':os.uname().nodename,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_path':run,'processes':[]}}
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  cmd=p.read_bytes().decode(errors='replace').replace('\\0',' ')
  if run not in cmd:continue
  status=(p.parent/'stat').read_text();env={{}}
  for pair in (p.parent/'environ').read_bytes().split(b'\\0'):
   key,sep,value=pair.partition(b'=')
   if sep and key.decode(errors='replace') in ('RANK','LOCAL_RANK','WORLD_SIZE','RUN_TAG'):
    env[key.decode()]=value.decode(errors='replace')
  result['processes'].append({{'pid':int(p.parent.name),'cmdline':cmd,'stat':status,'wchan':(p.parent/'wchan').read_text(),'rank_environment':env}})
 except (OSError,ValueError):pass
print(json.dumps(result))
''')
        out=Path('outputs','training_results_'+tag,stage,r['host'])
        out.mkdir(parents=True,exist_ok=True)
        target=out/'own-process-start-snapshot.json'
        if target.exists():
            old=json.loads(target.read_text());merged={p['pid']:p for p in old['processes']}
            merged.update({p['pid']:p for p in r['processes']});r['processes']=list(merged.values())
            r['observation_times']=old.get('observation_times',[old['utc']])+[r['utc']]
        target.write_text(json.dumps(r,indent=2))
        return dict(host=host, pids=[p['pid'] for p in r['processes']], ranks=[p['rank_environment'].get('RANK') for p in r['processes']])
    print(json.dumps(parallel(capture)))
finally:
    backend.close()
