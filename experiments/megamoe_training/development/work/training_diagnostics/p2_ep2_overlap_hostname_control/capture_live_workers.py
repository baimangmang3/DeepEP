"""Read only own live worker /proc metadata; no framework or device APIs."""
import os,sys,pathlib,json,time,hashlib
run=pathlib.Path(sys.argv[1]);output=pathlib.Path(sys.argv[2])
root=pathlib.Path('/home/zzy/deploy/megamoe-20261024/training')
assert run.is_relative_to(root/'runs') and output.is_relative_to(run)
script='/home/zzy/deploy/megamoe-20261024/training/candidate-p2-qualification-schema2-20261026/p2_backward_overlap/validate_training_p2.py'
cache_path=run/'native-fd-hash-cache.json'
cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
workers=[]
def readargv(p):return [v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
def start(p):return (p/'stat').read_text().rsplit(')',1)[1].split()[19]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  argv=readargv(p)
  if script not in argv or '--output' not in argv or argv[argv.index('--output')+1]!=str(run):continue
  env={}
  for item in (p/'environ').read_bytes().split(b'\0'):
   if b'=' not in item:continue
   k,v=item.split(b'=',1)
   if k in (b'ASCEND_RT_VISIBLE_DEVICES',b'MASTER_ADDR',b'MASTER_PORT',b'TORCHELASTIC_RESTART_COUNT',b'RANK',b'LOCAL_RANK',b'WORLD_SIZE',b'RUN_TAG',b'ASCEND_PROCESS_LOG_PATH',b'ASCEND_GLOBAL_LOG_LEVEL',b'ASCEND_HOME_PATH',b'PYTHONPATH',b'HCCL_SOCKET_IFNAME',b'HCCL_OP_EXPANSION_MODE'):env[k.decode()]=v.decode(errors='replace')
  if 'RANK' not in env or env.get('RUN_TAG')!=run.parent.name or env.get('WORLD_SIZE')!='2' or env['RANK'] not in ('0','1'):continue
  proc_start=start(p);maps=(p/'maps').read_text()
  row={'pid':int(p.name),'rank':int(env['RANK']),'utc_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'process_start_ticks':proc_start,'uid':p.stat().st_uid,'cwd':os.readlink(p/'cwd'),'stat':(p/'stat').read_text(),'argv':argv,'environment':env,'raw_maps':maps,'mapped_native_fd_identity':[]}
  seen=set()
  for line in maps.splitlines():
   parts=line.split(maxsplit=5)
   if len(parts)!=6 or not parts[5].startswith('/'):continue
   name=pathlib.PurePath(parts[5]).name
   if not any(v in name for v in ('libopapi','libhccl','libhcomm','libascendcl','libruntime','libshmem','libtorch_npu','libdeepep','libmegamoe')) and '/deep_ep/_C' not in parts[5] and '/torch_npu/_C' not in parts[5]:continue
   ma,mi=[int(v,16) for v in parts[3].split(':')];ino=int(parts[4]);key=(ma,mi,ino)
   if key in seen:continue
   seen.add(key);item={'maps_path':parts[5],'maps_device':parts[3],'maps_inode':ino,'range':parts[0]}
   try:
    try:f=open(p/'map_files'/parts[0],'rb');item['fd_open_source']='proc_map_files'
    except OSError as exc:
     item['map_files_open_error']=repr(exc);f=open(parts[5],'rb');item['fd_open_source']='mapped_path_with_dev_inode_check'
    with f:
     st=os.fstat(f.fileno());assert (os.major(st.st_dev),os.minor(st.st_dev),st.st_ino)==(ma,mi,ino),'FD object does not match actual map'
     ck=json.dumps([st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns],separators=(',',':'))
     if ck not in cache:cache[ck]=hashlib.file_digest(f,'sha256').hexdigest()
     after=os.fstat(f.fileno());assert (after.st_size,after.st_mtime_ns)==(st.st_size,st.st_mtime_ns),'FD changed while hashing'
     item.update(fd_device=st.st_dev,fd_inode=st.st_ino,bytes=st.st_size,sha256=cache[ck],fd_verified_against_actual_map=True)
   except Exception as exc:item['error']=repr(exc)
   row['mapped_native_fd_identity'].append(item)
  row['same_worker_after_capture']=start(p)==proc_start and readargv(p)==argv
  workers.append(row)
 except (OSError,ValueError,IndexError) as exc:continue
output.parent.mkdir(parents=True,exist_ok=True)
cache_path.write_text(json.dumps(cache,indent=2))
report={'scope':'read-only own live workers; host/FD identity not device payload proof','run':str(run),'time_ns':time.time_ns(),'workers':workers}
output.write_text(json.dumps(report,indent=2))
print(json.dumps({'path':str(output),'workers':len(workers),'ranks':sorted(r['rank'] for r in workers),'errors':sum('error' in v for r in workers for v in r['mapped_native_fd_identity'])}))
