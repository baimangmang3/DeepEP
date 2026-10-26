import pathlib,os,json,paramiko,hashlib,concurrent.futures
TAG='20261025_0829';STAGE='p1-ep16-bf16'
def one(spec):
 host,machine,pid=spec;c=paramiko.SSHClient();c.load_system_host_keys();c.connect(host,username='root',password=os.environ['DEPLOY_PASSWORD_'+host.split('.')[-2]],timeout=15)
 code="""import pathlib,json,hashlib
pid=PID;root=pathlib.Path('/root/ascend/log');rows=[]
for pattern in ('debug/device-*/device-'+str(pid)+'*','run/device-*/device-'+str(pid)+'*','debug/plog/plog-'+str(pid)+'*','run/plog/plog-'+str(pid)+'*'):
 for p in root.glob(pattern):
  if p.is_file():rows.append(dict(path=str(p),bytes=p.stat().st_size))
for p in pathlib.Path('/root/shmem/log').glob('aclshmem_'+str(pid)+'_*'):
 if p.is_file():rows.append(dict(path=str(p),bytes=p.stat().st_size))
print(json.dumps(rows))
""".replace('PID',str(pid))
 _,o,e=c.exec_command("timeout 30 /root/miniconda3/envs/zzy/bin/python - <<'PY'\n"+code+"\nPY",timeout=40);rows=json.loads(o.read());assert o.channel.recv_exit_status()==0,e.read().decode();dest=pathlib.Path('outputs/training_results_'+TAG)/STAGE/machine/'targeted-pid-logs';dest.mkdir(parents=True,exist_ok=True);s=c.open_sftp()
 for row in rows:
  if row['bytes']>12000000:row['skipped']='size cap 12MB';continue
  remote=pathlib.PurePosixPath(row['path']);local=dest/(remote.parent.parent.name+'-'+remote.parent.name+'-'+remote.name);s.get(row['path'],str(local));row['local_path']=str(local);row['sha256']=hashlib.sha256(local.read_bytes()).hexdigest()
 (dest/'manifest.json').write_text(json.dumps({'pid':pid,'files':rows},indent=2));s.close();c.close();return {'host':host,'pid':pid,'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'manifest':str(dest/'manifest.json')}
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 for v in pool.map(one,[('177.22.27.1','rack05os7',1670834),('177.22.28.1','rack05os8',1747163)]):print(json.dumps(v))
