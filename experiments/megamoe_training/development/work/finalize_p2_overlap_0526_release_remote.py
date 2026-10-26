import pathlib,json
from training_native_round import SshBackend,T,HOSTS,parallel
TAG='20261026_0526_p2_overlap_once';STAGE='p2-ep2-bf16-overlap';RUN=T+'/runs/'+TAG+'/'+STAGE
OUT=pathlib.Path('outputs/training_results_'+TAG)/STAGE
EARLY=json.loads((OUT/'rack05os7/early-state.json').read_text())['own']
x=SshBackend();x.connect()
try:
 result=x.python(HOSTS[0],'archive_'+TAG,f'''import pathlib,json,time,os,hashlib,re,tarfile
run=pathlib.Path({RUN!r});markers=[];saved=[]
for p in run.glob('.node0-*/identity.json'):
 r=json.loads(p.read_text());pid=r['pid'];wanted=str(r.get('start_ticks',r.get('process_start_ticks')));proc=pathlib.Path('/proc')/str(pid)
 try:now=(proc/'stat').read_text().rsplit(')',1)[1].split()[19];same=now==wanted
 except FileNotFoundError:now=None;same=False
 markers.append({{'source':str(p),'identity':r,'current_start':now,'same_PID_start_alive':same}})
saved=[r['identity']['pid'] for r in markers]
early_checks=[]
for row in {EARLY!r}:
 p=pathlib.Path('/proc')/str(row['pid'])
 try:now=(p/'stat').read_text().rsplit(')',1)[1].split()[19];same=now==row['start']
 except FileNotFoundError:now=None;same=False
 early_checks.append({{'pid':row['pid'],'expected_start':row['start'],'pgid':row['pgid'],'argv':row['argv'],'current_start':now,'same_PID_start_alive':same}})
raw=[]
for name in ('ascend-log','shmem-debug'):
 root=run/name
 if root.exists():
  for p in root.rglob('*'):
   if p.is_file():raw.append({{'relative_path':str(p.relative_to(run)),'bytes':p.stat().st_size}})
files=[]
for p in run.rglob('*'):
 if not p.is_file() or p.name=='compact-evidence.tar.gz' or any(x in p.parts for x in ('ascend-log','shmem-debug')):continue
 if p.stat().st_size>16<<20:continue
 files.append(p)
archive=run/'compact-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(run)),recursive=False)
with archive.open('rb') as f:archive_sha=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps({{'time_ns':time.time_ns(),'recorded_marker_PID_start_checks':markers,'all_recorded_markers_gone':all(not v['same_PID_start_alive'] for v in markers),'sampled_early_own_PID_start_checks':early_checks,'all_sampled_early_own_gone':all(not v['same_PID_start_alive'] for v in early_checks),'raw_logs_preserved_remote':str(run),'raw_log_inventory':raw,'raw_log_bytes':sum(v['bytes'] for v in raw),'compact_archive':str(archive),'compact_archive_sha256':archive_sha,'scope':'No NPU query or process signal. Only this existing run tree and exact saved marker/early own PID/start.'}}))''')
 (OUT/'release-marker-and-log-preservation.json').write_text(json.dumps(result,indent=2))
 assert result['all_recorded_markers_gone']
 with x.clients[HOSTS[0]].open_sftp() as ftp:ftp.get(result['compact_archive'],str(OUT/'rack05os7-compact-evidence.tar.gz'))
 print(json.dumps({'markers':len(result['recorded_marker_PID_start_checks']),'all_markers_gone':True,'raw_log_bytes':result['raw_log_bytes'],'archive_sha256':result['compact_archive_sha256']}))
finally:x.close()
