import os,paramiko,pathlib,json,sys
host,tag,stage=sys.argv[1:4];skip_traces='--skip-traces' in sys.argv[4:];skip_debug='--skip-debug' in sys.argv[4:];root='/home/zzy/deploy/megamoe-20261024/training/runs/'+tag
c=paramiko.SSHClient();c.load_system_host_keys();c.connect(host,username='root',password=os.environ['DEPLOY_PASSWORD'],timeout=15)
s=c.open_sftp();dest=pathlib.Path('outputs')/('training_results_'+tag)/stage/('rack05os7' if host=='177.22.27.1' else 'rack05os8');dest.mkdir(parents=True,exist_ok=True);trace_inventory=[]
def getdir(remote,local):
 local.mkdir(parents=True,exist_ok=True)
 for row in s.listdir_attr(remote):
  if row.filename.endswith('_trace.json'):
   trace_inventory.append({'path':remote+'/'+row.filename,'bytes':row.st_size})
   if not skip_traces:
    temporary=local/(row.filename+'.download');s.get(remote+'/'+row.filename,str(temporary));temporary.replace(local/row.filename)
  elif row.filename in ('logs','shmem-debug','shmem-info'):
   if not (skip_debug and row.filename.startswith('shmem-')):getdir(remote+'/'+row.filename,local/row.filename)
  elif row.filename.endswith(('.json','.log','.txt','.rc')) or row.filename in ('rc','supervisor.pid'):
   s.get(remote+'/'+row.filename,str(local/row.filename))
getdir(root+'/'+stage,dest)
if stage=='native-gemm' or stage.startswith('core-query'):
 for name in s.listdir(root+'/preflight'):
  if name.endswith('core-info.json'):s.get(root+'/preflight/'+name,str(dest/name))
summary={'host':host,'stage':stage,'rc':None,'rank_reports':[],'trace_inventory':trace_inventory,'trace_download_skipped':skip_traces,'incomplete_json_snapshots':[]}
for file in dest.rglob('*'):
 if file.name=='rc' or file.suffix=='.rc':summary[file.relative_to(dest).as_posix()]=file.read_text().strip()
 if file.name.startswith(('rank','compare_rank','perf_rank','lifecycle_rank')) and file.suffix=='.json':
  try:r=json.loads(file.read_text())
  except (json.JSONDecodeError,UnicodeError) as error:
   summary['incomplete_json_snapshots'].append({'file':str(file),'error':str(error)})
   file.rename(file.with_suffix('.json.partial'));continue
  summary['rank_reports'].append({'rank':r.get('rank'),'pid':r.get('pid'),'all_passed':r.get('all_passed'),'passed':r.get('passed'),'last_stage':r.get('stages',[])[-1:] ,'case_count':len(r.get('cases',[])),'error':r.get('error')})
  if r.get('pid'):
   try:
    with s.open('/proc/'+str(r['pid'])+'/cmdline','rb') as f:cmd=f.read().decode(errors='replace')
    if any('/home/zzy/deploy/megamoe-20261024/training/scripts/'+name in cmd for name in ('hccl_minimal_smoke.py','hccl_paired_smoke.py')):
     with s.open('/proc/'+str(r['pid'])+'/maps','rb') as f:maps=f.read()
     (dest/('rank'+str(r['rank'])+'-live-maps.txt')).write_bytes(maps)
   except OSError as error:summary.setdefault('optional_live_maps_errors',[]).append({'pid':r['pid'],'error':str(error)})
for name in ('core-query.log','native.log','supervisor.log','logs/node0.log','logs/node1.log'):
 f=dest/name
 if f.exists():summary[name]=f.read_text(errors='replace')[-6000:]
(dest/'collection-summary.json').write_text(json.dumps(summary,indent=2));s.close();c.close();print(json.dumps(summary))
