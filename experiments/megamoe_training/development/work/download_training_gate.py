import os,paramiko,pathlib,sys
host,stamp=sys.argv[1:];machine='rack05os7' if host=='177.22.27.1' else 'rack05os8'
c=paramiko.SSHClient();c.load_system_host_keys();c.connect(host,username='root',password=os.environ['DEPLOY_PASSWORD'],timeout=15)
s=c.open_sftp();out=pathlib.Path('outputs/training_occupancy_20261025_0300');out.mkdir(parents=True,exist_ok=True)
prefix=machine+'-'+stamp
for suffix in ('.json','.txt','-owners.json'):
 name=prefix+suffix;s.get('/home/zzy/deploy/megamoe-20261024/training/runs/20261025_0300/preflight/'+name,str(out/name))
s.close();c.close();print('downloaded',machine,'3 evidence files')
