import pathlib,paramiko,json,os,sys,concurrent.futures
TAG,STAGE=sys.argv[1:3];T='/home/zzy/deploy/megamoe-20261024/training';RUN=T+'/runs/'+TAG+'/'+STAGE

def one(host):
 c=paramiko.SSHClient();c.load_system_host_keys();c.connect(host,username='root',password=os.environ['DEPLOY_PASSWORD_'+host.split('.')[-2]],timeout=15)
 code=f'''import pathlib,os,json,time,hashlib
run={RUN!r};rows=[];cache={{}}
def hfile(path,ino,major,minor):
 try:
  with open(path,'rb') as f:
   st=os.fstat(f.fileno());assert st.st_ino==ino and os.major(st.st_dev)==major and os.minor(st.st_dev)==minor
   key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
   if key not in cache:
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    cache[key]=h.hexdigest()
   return {{'sha256':cache[key],'bytes':st.st_size,'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns}}
 except (OSError,AssertionError) as e:return {{'error':repr(e)}}
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=p.joinpath('cmdline').read_bytes().decode(errors='replace').replace('\\0',' ')
  if run not in args or '/candidate-p1-host-v2/p1_pointwise/validate_training_p1.py' not in args:continue
  env={{}}
  for x in p.joinpath('environ').read_bytes().split(b'\\0'):
   if b'=' in x:
    k,v=x.split(b'=',1)
    if k in (b'RANK',b'LOCAL_RANK',b'WORLD_SIZE',b'ASCEND_GLOBAL_LOG_LEVEL',b'ASCEND_PROCESS_LOG_PATH'):env[k.decode()]=v.decode()
  if 'RANK' not in env:continue
  maps=p.joinpath('maps').read_text();r={{'pid':int(p.name),'rank':int(env['RANK']),'utc_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'stat':p.joinpath('stat').read_text(),'wchan':p.joinpath('wchan').read_text(),'limits':p.joinpath('limits').read_text(),'argv':args,'environment':env,'raw_maps':maps,'selected_mapped_libraries':[]}}
  seen=set()
  for line in maps.splitlines():
   bits=line.split(maxsplit=5)
   if len(bits)!=6 or not bits[5].startswith('/'):continue
   name=pathlib.PurePosixPath(bits[5]).name
   if not any(x in name for x in ('libhccl','libhcomm','libruntime','libascendcl','libtorch_npu','libshmem','libmegamoe')) and '/deep_ep/_C' not in bits[5]:continue
   ma,mi=[int(v,16) for v in bits[3].split(':')];ino=int(bits[4]);key=(bits[5],ino,ma,mi)
   if key in seen:continue
   seen.add(key);r['selected_mapped_libraries'].append({{'path':bits[5],'map_inode':ino,'map_device_major':ma,'map_device_minor':mi,**hfile(bits[5],ino,ma,mi)}})
  rows.append(r)
 except (OSError,ValueError):continue
r={{'run':run,'time_ns':time.time_ns(),'scope':'live own workers only; stdlib read-only, no device calls','loadavg':pathlib.Path('/proc/loadavg').read_text(),'meminfo':pathlib.Path('/proc/meminfo').read_text(),'workers':rows}};out=pathlib.Path(run)/'live-worker-evidence.json';out.write_text(json.dumps(r,indent=2));print(json.dumps({{'path':str(out),'workers':len(rows),'ranks':sorted(x['rank'] for x in rows),'mapping_errors':sum('error' in x for r in rows for x in r['selected_mapped_libraries'])}}))
'''
 _,o,e=c.exec_command("nice -n 10 timeout 60 /root/miniconda3/envs/zzy/bin/python - <<'PY'\n"+code+"\nPY",timeout=70);r=json.loads(o.read());assert o.channel.recv_exit_status()==0,e.read().decode();dest=pathlib.Path('outputs/training_results_'+TAG)/STAGE/('rack05os7' if '.27.' in host else 'rack05os8');dest.mkdir(parents=True,exist_ok=True)
 with c.open_sftp() as s:s.get(r['path'],str(dest/'live-worker-evidence.json'))
 c.close();return {'host':host,**r}
with concurrent.futures.ThreadPoolExecutor(2) as ex:
 for r in ex.map(one,['177.22.27.1','177.22.28.1']):print(json.dumps(r))
