"""Bounded foreground observation of an already-authorized running job, no launch."""
import argparse,pathlib,json,shlex
from training_native_round import SshBackend,T,parallel,HOSTS
p=argparse.ArgumentParser();p.add_argument('run_tag');a=p.parse_args();assert all(v.isalnum() or v in '_-' for v in a.run_tag)
STAGE='p1-host-v2-main-shared-bf16-balanced';RUN=T+'/runs/'+a.run_tag+'/'+STAGE;R=T+'/validation-p1-main-shared-20261026'
OUT=pathlib.Path('outputs/training_results_'+a.run_tag)/STAGE
x=SshBackend();x.connect()
try:
 def one(h):
  n=HOSTS.index(h);host='rack05os7' if n==0 else 'rack05os8'
  code=f'''import pathlib,json,time,subprocess
run=pathlib.Path({RUN!r});log=run/'logs/node{n}.log';out=run/'live/case-start-FD.json';deadline=time.monotonic()+45;reason='phase_not_yet_seen'
if out.exists():print(json.dumps({{'status':'existing_snapshot_no_duplicate','path':str(out)}}))
else:
 while time.monotonic()<deadline:
  text=log.read_text(errors='replace') if log.exists() else ''
  if '"phase": "case_start"' in text:
   r=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',{R!r}+'/scripts/capture_live_workers.py',str(run),str(out)],capture_output=True,text=True,timeout=60)
   print(json.dumps({{'status':'case_phase_triggered','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'path':str(out)}}));break
  rc=run/'logs/node{n}.rc'
  if rc.exists() or 'Traceback (most recent call last)' in text:
   print(json.dumps({{'status':'job_exit_or_error_before_case','rc':rc.read_text() if rc.exists() else None}}));break
  time.sleep(2)
 else:print(json.dumps({{'status':reason,'wait_seconds':45,'no_device_operation':True}}))'''
  r=x.python(h,a.run_tag,code,timeout=140);d=OUT/host;d.mkdir(parents=True,exist_ok=True)
  with (d/'case-FD-observer-attempts.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
  if r.get('path'):
   with x.clients[h].open_sftp() as s:
    try:(d/'live').mkdir(exist_ok=True);s.get(r['path'],str(d/'live/case-start-FD.json'))
    except FileNotFoundError:pass
  return r
 print(json.dumps(parallel(one)))
finally:x.close()
