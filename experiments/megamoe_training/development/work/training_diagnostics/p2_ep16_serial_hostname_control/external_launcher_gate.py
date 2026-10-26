"""Stdlib read-only active launcher census; does not initialize/query an NPU."""
import os,pathlib,json,argparse,time
from cleanup_policy import external_launcher_evidence
WRAPPER='/home/zzy/deploy/megamoe-20261024/training/validation-p2-ep16-serial-hostname-20261026/scripts/launcher.py'
VISIBLE=('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES')
def collect(tag):
 rows=[];errors=[]
 for p in pathlib.Path('/proc').glob('[0-9]*'):
  candidate=False
  try:
   argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
   candidate=external_launcher_evidence([dict(argv=argv,run_tag='__candidate__')],'__gate__',known_wrappers=(WRAPPER,))['busy_or_uncertain']
   if not candidate:continue
   before=(p/'stat').read_text().rsplit(')',1)[1].split()
   env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
   after=(p/'stat').read_text().rsplit(')',1)[1].split()
   if before[19]!=after[19]:
    errors.append(dict(pid=int(p.name),error='PID start changed during recognized launcher read'));continue
   rows.append(dict(pid=int(p.name),start=before[19],ppid=int(before[1]),pgid=int(before[2]),state=before[0],argv=argv,run_tag=env.get(b'RUN_TAG',b'').decode(errors='replace'),visible={k:env[k.encode()].decode(errors='replace') for k in VISIBLE if k.encode() in env}))
  except (OSError,ValueError) as exc:
   if candidate:errors.append(dict(pid=int(p.name),error=repr(exc)))
 result=external_launcher_evidence(rows,tag,known_wrappers=(WRAPPER,));result.update(time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),scope='All accessible recognizable torchrun process groups, PPID/start/argv and explicit visible whitelist only. Other scheduler names unrecognized; past controller existence/use unknown.')
 result['recognized_launcher_read_errors']=errors;result['busy_or_uncertain']=result['busy_or_uncertain'] or bool(errors)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-tag',required=True);a=p.parse_args()
 result=collect(a.run_tag);print(json.dumps(result));raise SystemExit(12 if result['busy_or_uncertain'] else 0)
