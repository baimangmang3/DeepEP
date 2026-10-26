"""Read last two exact own-worker DEBUG plogs; preserve error-bearing files only."""
import pathlib,paramiko,json,os,concurrent.futures,hashlib
T='/home/zzy/deploy/megamoe-20261024/training';RUN=T+'/runs/20261025_1431/p1-host-v2-debug-shared-fp32-small'
B=pathlib.Path('outputs/training_results_20261025_1431/p1-host-v2-debug-shared-fp32-small')
def one(host):
 machine='rack05os7' if '.27.' in host else 'rack05os8';pairs=[(v['rank'],v['pid']) for v in json.loads((B/machine/'live-worker-evidence.json').read_text())['workers']]
 c=paramiko.SSHClient();c.load_system_host_keys();c.connect(host,username='root',password=os.environ['DEPLOY_PASSWORD_'+host.split('.')[-2]],timeout=15)
 code=f'''import pathlib,json,hashlib,re
root=pathlib.Path({RUN!r})/'ascend-log/debug/plog';rows=[]
for rank,pid in {pairs!r}:
 files=sorted(root.glob('plog-'+str(pid)+'*'))[-2:];r={{'rank':rank,'pid':pid,'files_checked':[],'errors':[]}}
 for p in files:
  raw=p.read_bytes();lines=raw.decode(errors='replace').splitlines();meta={{'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}};r['files_checked'].append(meta)
  for i,line in enumerate(lines):
   if '[ERROR]' in line or '[EXCEPTION]' in line:
    r['errors'].append({{'path':str(p),'line_number':i+1,'line':line,'context':lines[max(0,i-1):min(len(lines),i+6)]}})
  meta['error_bearing']=any(v['path']==str(p) for v in r['errors'])
 rows.append(r)
print(json.dumps(rows))'''
 _,o,e=c.exec_command("nice -n 10 timeout 80 /root/miniconda3/envs/zzy/bin/python - <<'PY'\n"+code+"\nPY",timeout=90);data=json.loads(o.read());assert o.channel.recv_exit_status()==0,e.read().decode()
 (B/machine/'targeted-debug-error-excerpts.json').write_text(json.dumps(data,indent=2))
 with c.open_sftp() as s:
  for r in data:
   if r['rank'] not in (0,5,8,13):continue
   chosen=[v for v in r['files_checked'] if v['error_bearing']]
   if chosen:r['selected_raw_path']=chosen[-1]['path']
   for v in r['files_checked']:
    if v['path']!=r.get('selected_raw_path'):continue
    if v['bytes']>32000000:v['download_skipped']='32MB cap';continue
    p=B/machine/'targeted-error-debug'/('rank'+str(r['rank']))/pathlib.PurePosixPath(v['path']).name;p.parent.mkdir(parents=True,exist_ok=True);s.get(v['path'],str(p));assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];v['local_path']=str(p)
 c.close();(B/machine/'targeted-debug-error-excerpts.json').write_text(json.dumps(data,indent=2));return {'host':machine,'workers':len(data),'with_errors':sum(bool(r['errors']) for r in data),'download_bytes':sum(v['bytes'] for r in data for v in r['files_checked'] if 'local_path' in v)}
with concurrent.futures.ThreadPoolExecutor(2) as e:
 for r in e.map(one,['177.22.27.1','177.22.28.1']):print(json.dumps(r))
