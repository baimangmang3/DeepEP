"""Reviewed finite isolated controls deployment plus actual CPU help/smoke only."""
import pathlib,json,hashlib,shlex
from training_native_round import SshBackend,T,HOSTS,parallel
SRC=pathlib.Path('work/training_diagnostics/p2_ep2_serial_hostname_control');OUT=pathlib.Path('outputs/p2_ep2_serial_hostname_preparation')
R=T+'/validation-p2-ep2-serial-hostname-20261026';API='/root/miniconda3/envs/zzy/lib/python3.12/site-packages/torch/distributed/elastic/agent/server/api.py'
sha=lambda b:hashlib.sha256(b).hexdigest()
x=SshBackend();x.connect()
try:
 def deploy(host):
  machine='rack05os7' if host==HOSTS[0] else 'rack05os8';d=OUT/machine;d.mkdir(exist_ok=False)
  with x.clients[host].open_sftp() as ftp:
   with ftp.open(API,'rb') as f:raw=f.read()
  assert sha(raw)=='8184afa022897c622de9ff1b76504e2210550fa772ccbc2fba44279c913bc1fe'
  old=json.loads(pathlib.Path(f'outputs/p2_ep2_serial_child_first_preparation/{machine}-diagnostic-deployment-manifest.json').read_text())
  m=json.loads(json.dumps(old));m['schema']='p2-ep2-serial-hostname-controls-CPU-3';m['device_authorized']=False
  m['source_files'][API]={'bytes':len(raw),'sha256':sha(raw)}
  m['files']={f'scripts/{p.name}':{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in SRC.iterdir() if p.is_file() and p.name!='controller.py'}
  m['files']['contract.json']=old['files']['contract.json']
  m['files']['scripts/probe_wrapper_cpu.py']={'bytes':pathlib.Path('work/p2_wrapper_cpu_probe.py').stat().st_size,'sha256':sha(pathlib.Path('work/p2_wrapper_cpu_probe.py').read_bytes())}
  m['control_revision'].update(scope='Isolated CPU deployment; device not authorized',DNS_wrapper='Exact7e15cc8 wrapper changes only per-RUN elastic event hostname helper; actual8184 API source pin; original torchrun arguments/rendezvous/worker unchanged',wrapper_exact_path=R+'/scripts/launcher.py')
  manifest=OUT/f'{machine}-diagnostic-deployment-manifest.json';manifest.write_text(json.dumps(m,indent=2));mh=sha(manifest.read_bytes())
  x.python(host,'p2_cpu_hostname_deploy',f'''import pathlib,json
r=pathlib.Path({R!r});assert not r.exists(),'Refuse overwrite'
for part in ('scripts','evidence'):r.joinpath(part).mkdir(parents=True,exist_ok=True)
print(json.dumps({{'created':str(r)}}))''')
  with x.clients[host].open_sftp() as ftp:
   for p in SRC.iterdir():
    if p.is_file() and p.name!='controller.py':ftp.put(str(p),R+'/scripts/'+p.name)
   ftp.put('work/p2_wrapper_cpu_probe.py',R+'/scripts/probe_wrapper_cpu.py')
   ftp.put('work/p2_ep2_bf16_device_contract.json',R+'/contract.json');ftp.put(str(manifest),R+'/diagnostic-deployment-manifest.json')
  verify=x.python(host,'p2_cpu_hostname_verify',f'''import subprocess,json
p=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',{R!r}+'/scripts/verify_deployment.py',{R!r}+'/diagnostic-deployment-manifest.json',{mh!r}],capture_output=True,text=True,timeout=45)
assert p.returncode==0,p.stdout+p.stderr
q=subprocess.run(['bash','-n',{R!r}+'/scripts/run_dispatch_host_trace_checked.sh'],capture_output=True,text=True,timeout=10);assert q.returncode==0,q.stderr
print(json.dumps({{'verify':json.loads(p.stdout),'bash_n_rc':q.returncode}}))''')
  cmd=f'source {T}/activate-qualified-p0.sh\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\ntimeout 120 python '+shlex.join([R+'/scripts/probe_wrapper_cpu.py','--wrapper',R+'/scripts/launcher.py','--output',R+'/evidence/wrapper-help.json'])+f' > {R}/evidence/help.stdout 2> {R}/evidence/help.stderr\n'
  _,o,e=x.clients[host].exec_command('bash -lc '+shlex.quote(cmd),timeout=145);stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status()
  (d/'verify.json').write_text(json.dumps(verify,indent=2));(d/'help-command-result.json').write_text(json.dumps({'rc':rc,'command':cmd,'stdout':stdout,'stderr':stderr},indent=2))
  with x.clients[host].open_sftp() as ftp:
   for name in ('wrapper-help.json','help.stdout','help.stderr'):
    try:ftp.get(R+'/evidence/'+name,str(d/name))
    except FileNotFoundError:pass
  assert rc==0,(machine,rc)
  report=json.loads((d/'wrapper-help.json').read_text());assert report['passed'] and not report['npu_initialized'] and not report['PG_initialized']
  return {'passed':True,'rc':rc,'manifest_sha256':mh,'api_sha256':sha(raw),'api_bytes':len(raw),'help':report}
 rows=parallel(deploy)
 assert rows[HOSTS[0]]['manifest_sha256']==rows[HOSTS[1]]['manifest_sha256']
 (OUT/'deployment-cpu-help-summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps({'help_both_passed':True,'manifest':rows[HOSTS[0]]['manifest_sha256']}))
finally:x.close()
