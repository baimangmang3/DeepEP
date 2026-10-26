"""CPU/read-only archive and PID-start release checks, never a device launch."""
import pathlib,json,hashlib,shlex
from training_native_round import SshBackend,T,parallel,HOSTS
TAG='20261026_0153_p1_main_routed_once';STAGE='p1-host-v2-main-bf16-balanced'
RUN=T+'/runs/'+TAG+'/'+STAGE;R=T+'/validation-p1-main-routed-20261026'
OUT=pathlib.Path('outputs/training_results_'+TAG)/STAGE
x=SshBackend();x.connect()
try:
 def one(h):
  m='rack05os7' if h==HOSTS[0] else 'rack05os8';dest=OUT/m
  early=json.loads((dest/'early-state.json').read_text());ledger={str(r['pid']):r['stat'].rsplit(')',1)[1].split()[19] for r in early['processes']}
  code=f'''import pathlib,json,hashlib,os,time,subprocess,tarfile
os.nice(10)
run=pathlib.Path({RUN!r});ledger={ledger!r};checks=[]
for pid,start in ledger.items():
 p=pathlib.Path('/proc')/pid
 try:
  s=(p/'stat').read_text().rsplit(')',1)[1].split();checks.append({{'pid':int(pid),'expected_start':start,'current_start':s[19],'state':s[0],'own_gone':s[19]!=start}})
 except FileNotFoundError:checks.append({{'pid':int(pid),'expected_start':start,'own_gone':True}})
release={{'time_ns':time.time_ns(),'processes':checks,'all_own_released':all(v['own_gone'] for v in checks)}}
(run/'own-release-check.json').write_text(json.dumps(release,indent=2))
assert release['all_own_released'],'Own PID still alive; archive cannot claim release'
manifest=[]
for relative in ('ascend-log','shmem-debug'):
 for p in sorted((run/relative).rglob('*')):
  if not p.is_file():continue
  st=p.stat()
  with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  manifest.append({{'path':str(p.relative_to(run)),'bytes':st.st_size,'sha256':sha}})
raw={{'scope':'all raw files in this run process-local ASCEND and SHMEM log roots; retained remote unchanged','files':manifest,'total_bytes':sum(v['bytes'] for v in manifest)}}
(run/'raw-log-manifest.json').write_text(json.dumps(raw,indent=2))
verify=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',{R!r}+'/scripts/verify_deployment.py',{R!r}+'/diagnostic-deployment-manifest.json','03eca19be3194a5720e0f4aaf01fb181a2db211bcd40782e216e14ced0bbb95d'],capture_output=True,text=True,timeout=45)
assert verify.returncode==0,verify.stdout+verify.stderr
archive=run/'diagnostic-small-evidence.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for p in sorted(run.rglob('*')):
  if not p.is_file() or p==archive:continue
  rel=str(p.relative_to(run))
  if rel.startswith(('ascend-log/','shmem-debug/')):continue
  tar.add(p,arcname=rel,recursive=False)
with archive.open('rb') as f:archive_sha=hashlib.file_digest(f,'sha256').hexdigest()
report={{'release':release,'raw_log_files':len(manifest),'raw_log_bytes':raw['total_bytes'],'small_archive':str(archive),'small_archive_sha256':archive_sha,'small_archive_bytes':archive.stat().st_size,'post_run_pinned_source_native_check':json.loads(verify.stdout),'all_raw_log_bytes_downloaded':False,'all_raw_logs_preserved_remote':True}}
(run/'archive-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))'''
  r=x.python(h,TAG,code,timeout=140)
  with x.clients[h].open_sftp() as s:
   for name in ('own-release-check.json','raw-log-manifest.json','archive-summary.json','diagnostic-small-evidence.tar.gz'):s.get(RUN+'/'+name,str(dest/name))
  local=dest/'diagnostic-small-evidence.tar.gz';assert hashlib.sha256(local.read_bytes()).hexdigest()==r['small_archive_sha256']
  return r
 rows=parallel(one);(OUT/'archive-summary.json').write_text(json.dumps(rows,indent=2))
 print(json.dumps({h:{k:r[k] for k in ('raw_log_files','raw_log_bytes','small_archive_bytes','small_archive_sha256')}|{'own_released':r['release']['all_own_released']} for h,r in rows.items()}))
finally:x.close()
