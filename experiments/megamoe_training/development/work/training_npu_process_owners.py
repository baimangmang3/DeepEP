"""Resolve PIDs from the existing npu-smi snapshot, without querying devices."""
import json,pathlib,pwd,re,socket,os
t=pathlib.Path('/home/zzy/deploy/megamoe-20261024/training')
run=t/'runs/20261025_0300/preflight'
snapshot=sorted(run.glob(socket.gethostname()+'-*.txt'))[-1]
pids=[int(x) for x in re.findall(r'^\|\s*\d+\s*\|\s*(\d+)\s*\|\s*python\s*\|',snapshot.read_text(),re.M)]
def cmd(pid):return [x.decode(errors='replace') for x in (pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
rows=[]
for pid in pids:
 row={'pid':pid}
 try:
  root=pathlib.Path('/proc')/str(pid);status=(root/'status').read_text()
  ppid=int(re.search(r'^PPid:\s*(\d+)',status,re.M)[1]);uid=int(re.search(r'^Uid:\s*(\d+)',status,re.M)[1])
  args=cmd(pid)
  row.update(ppid=ppid,uid=uid,user=pwd.getpwuid(uid).pw_name,args=args,cwd=os.readlink(root/'cwd'),parent_args=cmd(ppid),own=any(str(t) in x for x in args))
 except (OSError,ValueError,KeyError) as error:row['read_error']=str(error)
 rows.append(row)
result={'host':socket.gethostname(),'npu_snapshot':str(snapshot),'processes':rows,'own_device_processes':[r for r in rows if r.get('own')],'foreign_device_processes':[r for r in rows if r.get('own') is False]}
out=snapshot.with_name(snapshot.stem+'-owners.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));print('OWNERS_EVIDENCE='+str(out))
