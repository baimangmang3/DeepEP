"""Snapshot only this run's logs and exact owned worker-PID legacy logs."""
import concurrent.futures,hashlib,json,os,pathlib,paramiko,sys
TAG,STAGE,LABEL=sys.argv[1:4]
T='/home/zzy/deploy/megamoe-20261024/training';RUN=T+'/runs/'+TAG+'/'+STAGE
BASE=pathlib.Path('outputs/training_results_'+TAG)/STAGE

def one(host):
 c=paramiko.SSHClient();c.load_system_host_keys();c.connect(host,username='root',password=os.environ['DEPLOY_PASSWORD_'+host.split('.')[-2]],timeout=15)
 code=f'''import pathlib,json,hashlib,os,tarfile,re,shutil
run=pathlib.Path({RUN!r});label={LABEL!r};pairs={{}}
for p in run.glob('rank*.json'):
 try:
  r=json.loads(p.read_text());pairs[r['pid']]=r['rank']
 except (OSError,ValueError,KeyError):pass
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=p.joinpath('cmdline').read_bytes().decode(errors='replace').replace('\\0',' ')
  if str(run) not in args or '/candidate-p1-host-v2/p1_pointwise/validate_training_p1.py' not in args:continue
  env=dict(x.split(b'=',1) for x in p.joinpath('environ').read_bytes().split(b'\\0') if b'=' in x)
  if b'RANK' in env:pairs[int(p.name)]=int(env[b'RANK'])
 except (OSError,ValueError):pass
paths=set(p for p in (run/'ascend-log').rglob('*') if p.is_file())
for pid in pairs:
 for pattern in ('debug/device-*/device-'+str(pid)+'*','run/device-*/device-'+str(pid)+'*','debug/plog/plog-'+str(pid)+'*','run/plog/plog-'+str(pid)+'*'):
  paths.update(p for p in pathlib.Path('/root/ascend/log').glob(pattern) if p.is_file())
snapshot=run/('log-snapshot-'+label);snapshot.mkdir(exist_ok=False);rows=[];total=0
for p in sorted(paths):
 row={{'path':str(p)}};rows.append(row)
 try:
  st=p.stat();row['bytes_observed']=st.st_size
  if st.st_size>128000000 or total+st.st_size>1000000000:row['skipped']='bounded cap';continue
  dest=snapshot/str(p).lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  raw=dest.read_bytes();row.update(snapshot_path=dest.relative_to(snapshot).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());total+=len(raw)
 except OSError as exc:row['snapshot_error']=repr(exc);continue
 rows[-1]=row
manifest={{'scope':'own run subtree and exact owned host PID logs only','pairs':pairs,'files':rows}};(snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=run/('log-snapshot-'+label+'.tar.gz')
with tarfile.open(archive,'w:gz') as tar:tar.add(snapshot,arcname=snapshot.name)
print(json.dumps({{'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'files':len(rows),'raw_bytes':total,'pairs':pairs,'skipped':sum('skipped' in r for r in rows)}}))
'''
 _,o,e=c.exec_command("nice -n 10 timeout 75 /root/miniconda3/envs/zzy/bin/python - <<'PY'\n"+code+"\nPY",timeout=85)
 raw=o.read();error=e.read();assert o.channel.recv_exit_status()==0,error.decode();r=json.loads(raw)
 dest=BASE/('rack05os7' if '.27.' in host else 'rack05os8');dest.mkdir(parents=True,exist_ok=True);local=dest/pathlib.PurePosixPath(r['archive']).name
 with c.open_sftp() as s:s.get(r['archive'],str(local))
 assert hashlib.sha256(local.read_bytes()).hexdigest()==r['sha256'];c.close();(dest/('log-snapshot-'+LABEL+'-transfer.json')).write_text(json.dumps(r,indent=2));return {'host':host,**r}

with concurrent.futures.ThreadPoolExecutor(2) as ex:
 for r in ex.map(one,['177.22.27.1','177.22.28.1']):print(json.dumps(r))
