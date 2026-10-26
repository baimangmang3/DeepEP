"""Exact RUN process-group cleanup; separate recorders survive inner termination."""
import pathlib,json,time,os,argparse
from cleanup_policy import cleanup
T=pathlib.Path('/home/zzy/deploy/megamoe-20261024/training')
STAGE='p2-ep2-bf16-serial'
VALIDATOR=str(T/'candidate-p2-qualification-schema2-20261026/p2_backward_overlap/validate_training_p2.py')

class ProcIO:
 def __init__(self,tag):
  assert tag and all(c.isalnum() or c in '_-' for c in tag)
  self.tag=tag;self.run=T/'runs'/tag/STAGE
  assert self.run.is_dir(),'No existing own run, refuse cleanup'
  # Previously saved launch identities anchor both recorders, not a fresh PID guess.
  outer=json.loads((self.run/'.node0-controller-once/identity.json').read_text())
  nodepath=self.run/'.node0-launch-once/identity.json'
  self.outer=outer
  self.node=json.loads(nodepath.read_text()) if nodepath.exists() else None
  self.anchors={int(outer['pgid']):'outer_recorder'}
  if self.node:self.anchors[int(self.node['pgid'])]='node_recorder'
  snapshot=self.scan();self.ledger=[]
  innergroups={r['pgid'] for r in snapshot if r['run_tag']==tag and any(a in ('torch.distributed.run',VALIDATOR) for a in r['argv']) and r['pgid'] not in self.anchors}
  for row in snapshot:
   if row['run_tag']!=tag:continue
   role=self.anchors.get(row['pgid']) or ('worker' if row['pgid'] in innergroups else 'unknown')
   self.ledger.append(dict(row,role=role))
  # Exact marker PID-start must still match whenever an anchored group is alive.
  for marker in (self.outer,self.node):
   if not marker:continue
   r=next((v for v in snapshot if v['pid']==marker['pid']),None)
   if r and (str(r['start'])!=str(marker.get('start_ticks',marker.get('process_start_ticks'))) or r['run_tag']!=tag):
    raise RuntimeError('Recorder PID/start/RUN mismatch')
 def scan(self):
  rows=[]
  for p in pathlib.Path('/proc').glob('[0-9]*'):
   try:
    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
    rows.append(dict(pid=int(p.name),ppid=int(fields[1]),pgid=int(fields[2]),state=fields[0],start=fields[19],run_tag=env.get(b'RUN_TAG',b'').decode(errors='replace'),argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]))
   except (OSError,ValueError):pass
  return rows
 def members(self,group):return [r for r in self.scan() if r['pgid']==group]
 def time_ns(self):return time.time_ns()
 def monotonic_ns(self):return time.monotonic_ns()
 def signal_group(self,group,sig):os.killpg(group,sig)
 def wait_gone(self,roles,seconds):
  wanted={(r['pid'],r['start']) for r in self.ledger if r['role'] in roles};end=time.monotonic()+seconds
  while True:
   if not any((r['pid'],r['start']) in wanted for r in self.scan()):return True
   if time.monotonic()>=end:return False
   time.sleep(.2)
 def read_rc(self):
  result={}
  for key,p in [('node_rc',self.run/'logs/node0.rc'),('supervisor_rc',self.run/'supervisor-node0.rc')]:
   result[key]=int(p.read_text()) if p.exists() else None
  return result
 def harvest_rc(self,deadline_seconds):
  end=time.monotonic()+deadline_seconds
  while True:
   r=self.read_rc()
   if all(v is not None for v in r.values()) or time.monotonic()>=end:return r
   time.sleep(.2)
 def recorders_done(self):return all(v is not None for v in self.read_rc().values())
 def save_summary(self,report):
  report.update(run_tag=self.tag,ledger=self.ledger,remaining=[r for r in self.scan() if (r['pid'],r['start']) in {(p['pid'],p['start']) for p in self.ledger}],saved_wall_ns=time.time_ns(),saved_monotonic_ns=time.monotonic_ns())
  (self.run/'own-child-first-cleanup.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-tag',required=True);a=p.parse_args()
 io=ProcIO(a.run_tag);result=cleanup(io,io.ledger,a.run_tag)
 print(json.dumps(result))
