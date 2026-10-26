import pathlib,json,hashlib,paramiko,os,concurrent.futures
SRC=pathlib.Path('work/training_candidates/p1_pointwise');M=json.loads((SRC/'training_manifest.json').read_text());FILES=M['sources_sha256'];assert len(FILES)==8
for n,h in FILES.items():assert hashlib.sha256((SRC/n).read_bytes()).hexdigest()==h
ROOT='/home/zzy/deploy/megamoe-20261024/training/candidate-p1-host-v2/p1_pointwise';SO='/home/zzy/deploy/megamoe-20261024/training/candidate-p1/build/libmegamoe_pointwise.so';OUT=pathlib.Path('outputs/p1_host_v2_cpu_preparation');OUT.mkdir(exist_ok=True)
def one(host):
 c=paramiko.SSHClient();c.load_system_host_keys();c.connect(host,username='root',password=os.environ['DEPLOY_PASSWORD_'+host.split('.')[-2]],timeout=15)
 def call(code):
  _,o,e=c.exec_command("timeout 30 /root/miniconda3/envs/zzy/bin/python - <<'PY'\n"+code+"\nPY",timeout=40);r=o.read();assert o.channel.recv_exit_status()==0,e.read().decode();return json.loads(r)
 call(f'''import pathlib,json,hashlib
+root=pathlib.Path({ROOT!r});files={FILES!r}
+for n,h in files.items():
+ p=root/n
+ if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==h,'Existing v2 path differs: '+n
+(root/'native').mkdir(parents=True,exist_ok=True)
+print(json.dumps({{'ready':True}}))'''.replace('\n+','\n'))
 with c.open_sftp() as s:
  for n in FILES:
   target=ROOT+'/'+n;s.put(str(SRC/n),target+'.tmp');s.posix_rename(target+'.tmp',target)
 r=call(f'''import pathlib,json,hashlib
+root=pathlib.Path({ROOT!r});files={FILES!r}
+assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in files.items())
+assert hashlib.sha256(pathlib.Path({SO!r}).read_bytes()).hexdigest()=='e189f39a8448df078e082ba90b3147e593962bbcc374aa388d69344cfc125ca1'
+r={{'root':str(root),'sources_sha256':files,'pointwise_library':{SO!r},'pointwise_sha256':'e189f39a8448df078e082ba90b3147e593962bbcc374aa388d69344cfc125ca1','status':'old8-source staging only; awaiting reviewed host-v2 validator; NOT QUALIFIED; no activation/launcher installed','torch_imported':False,'npu_initialized':False}}
+(root/'cpu-staging.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))'''.replace('\n+','\n'));c.close();(OUT/('rack05os7.json' if '.27.' in host else 'rack05os8.json')).write_text(json.dumps(r,indent=2));return {'host':host,'root':ROOT,'files':len(FILES),'status':r['status']}
with concurrent.futures.ThreadPoolExecutor(2) as ex:
 for r in ex.map(one,['177.22.27.1','177.22.28.1']):print(json.dumps(r))
