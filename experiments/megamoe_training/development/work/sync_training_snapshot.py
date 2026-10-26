import os,paramiko,pathlib,hashlib,json
local=pathlib.Path('work/DeepEP/experiments/megamoe_training')
source_manifest=json.loads(pathlib.Path('outputs/training_source_manifest.json').read_text())
for item in source_manifest['files']:
 p=pathlib.Path('work/DeepEP')/item['path']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],str(p)+' differs from approved snapshot'
assert hashlib.sha256((local/'native/gemm.cpp').read_bytes()).hexdigest()=='a7c4363f9ce555436ef53b340a20f538e29f01e2560c1f5739c1fcd28f13e520','Native source changed: rebuild before deploying'
files=[p for p in local.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'build' not in p.parts]
c=paramiko.SSHClient();c.load_system_host_keys();c.connect(os.environ['DEPLOY_HOST'],username='root',password=os.environ['DEPLOY_PASSWORD'],timeout=20)
d='/home/zzy/deploy/megamoe-20261024/training/DeepEP/experiments/megamoe_training'
dirs={str(p.relative_to(local).parent).replace('\\','/') for p in files}|{'build'}
_,o,e=c.exec_command('mkdir -p '+' '.join(d+'/'+name for name in sorted(dirs)));assert o.channel.recv_exit_status()==0
s=c.open_sftp();manifest=[]
for p in files:
 name=p.relative_to(local).as_posix();target=d+'/'+name
 s.put(str(p),target+'.tmp');s.posix_rename(target+'.tmp',target)
 manifest.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
lib=pathlib.Path('work/training_native/libmegamoe_training.so');target=d+'/build/libmegamoe_training.so'
s.put(str(lib),target+'.tmp');s.posix_rename(target+'.tmp',target)
manifest.append(dict(path='build/libmegamoe_training.so',sha256=hashlib.sha256(lib.read_bytes()).hexdigest()))
s.put('outputs/training_source_manifest.json','/home/zzy/deploy/megamoe-20261024/training/logs/training_source_manifest.json')
with s.open('/home/zzy/deploy/megamoe-20261024/training/logs/source-snapshot.json.tmp','w') as f:f.write(json.dumps(manifest,indent=2))
s.posix_rename('/home/zzy/deploy/megamoe-20261024/training/logs/source-snapshot.json.tmp','/home/zzy/deploy/megamoe-20261024/training/logs/source-snapshot.json')
s.close();c.close();print('Source files',len(files),'native SHA',manifest[-1]['sha256'])
