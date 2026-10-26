import pathlib,paramiko,os,json,hashlib,concurrent.futures
TAG='20261025_1431';STAGE='p1-host-v2-debug-shared-fp32-small';BASE=pathlib.Path('outputs/training_results_'+TAG)/STAGE

def one(host):
 machine='rack05os7' if '.27.' in host else 'rack05os8';workers=json.loads((BASE/machine/'live-worker-evidence.json').read_text())['workers'];pairs=[(v['rank'],v['pid']) for v in workers];c=paramiko.SSHClient();c.load_system_host_keys();c.connect(host,username='root',password=os.environ['DEPLOY_PASSWORD_'+host.split('.')[-2]],timeout=15)
 code=f'''import pathlib,json,hashlib
pairs={pairs!r};roots=[pathlib.Path('/home/zzy/deploy/megamoe-20261024/training/runs/20261025_1431/p1-host-v2-debug-shared-fp32-small/ascend-log'),pathlib.Path('/root/ascend/log')];rows=[]
for rank,pid in pairs:
 for root in roots:
  for pattern in ('run/device-*/device-'+str(pid)+'*','run/plog/plog-'+str(pid)+'*'):
   for p in root.glob(pattern):
    if p.is_file():rows.append({{'rank':rank,'pid':pid,'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}})
print(json.dumps(rows))'''
 _,o,e=c.exec_command("timeout 45 /root/miniconda3/envs/zzy/bin/python - <<'PY'\n"+code+"\nPY",timeout=55);rows=json.loads(o.read());assert o.channel.recv_exit_status()==0,e.read().decode()
 with c.open_sftp() as s:
  for row in rows:
   if row['bytes']>32000000:row['skipped']='over32MB cap; explicitly not complete';continue
   remote=pathlib.PurePosixPath(row['path']);dest=BASE/machine/'targeted-run-logs'/('rank'+str(row['rank']));dest.mkdir(parents=True,exist_ok=True);local=dest/(remote.parent.parent.name+'-'+remote.parent.name+'-'+remote.name);s.get(row['path'],str(local));assert hashlib.sha256(local.read_bytes()).hexdigest()==row['sha256'];row['local_path']=str(local)
 c.close();(BASE/machine/'targeted-run-log-manifest.json').write_text(json.dumps({'pairs':pairs,'files':rows},indent=2));return {'host':machine,'ranks':len(pairs),'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'skipped':sum('skipped' in r for r in rows)}
with concurrent.futures.ThreadPoolExecutor(2) as ex:
 for r in ex.map(one,['177.22.27.1','177.22.28.1']):print(json.dumps(r))
