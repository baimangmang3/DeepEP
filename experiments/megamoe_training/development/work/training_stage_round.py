"""One-shot dual-host gate and immediate checked torchrun stage launch."""
import argparse,json,re,shlex,time,hashlib
from pathlib import Path
from training_native_round import SshBackend,HOSTS,T,parallel,EXPECTED_IDENTITY


def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--run-tag',required=True);p.add_argument('--stage',required=True);p.add_argument('--nnodes',type=int,choices=(1,2),default=2);p.add_argument('--single-host',choices=HOSTS,default=HOSTS[0]);p.add_argument('--nproc',type=int,default=8);p.add_argument('--limit',type=int,default=900);p.add_argument('--script',default='tests/validate_training_npu.py');p.add_argument('--launcher',default=T+'/scripts/run_training_review_checked.sh');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
 if not a.execute:p.error('--execute required')
 if not all(re.fullmatch('[A-Za-z0-9_-]+',x) for x in (a.run_tag,a.stage)):p.error('invalid tag/stage')
 arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
 if Path(a.script).name=='hccl_paired_smoke.py' and '--allgather' not in arguments:p.error('paired smoke requires --allgather; no launch performed')
 out=Path('outputs')/('training_round_'+a.run_tag)/(a.stage+'-controller');out.mkdir(parents=True,exist_ok=True)
 backend=SshBackend();report={'run_tag':a.run_tag,'stage':a.stage,'status':'not_started','requested_script':a.script,'script_arguments':arguments,'nnodes':a.nnodes,'nproc_per_node':a.nproc,'limit_seconds':a.limit,'scope':'paired_MIN_SUM_ALLGATHER' if Path(a.script).name=='hccl_paired_smoke.py' else 'requested_script_scope'}
 local_script=Path(__file__).with_name(Path(a.script).name)
 if a.script.startswith(T+'/scripts/') and local_script.is_file():backend.helper_hashes[a.script]=hashlib.sha256(local_script.read_bytes()).hexdigest()
 try:
  backend.connect();checks=parallel(lambda host:backend.verify(host,a.run_tag))
  if not all(r.get('passed') is True and r.get('cached_content_digest')==EXPECTED_IDENTITY for r in checks.values()):raise ValueError('Frozen review identity mismatch')
  (out/'quick-identities.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
  gates=parallel(lambda host:backend.gate(host,a.run_tag));finish=time.monotonic()
  (out/'gates.json').write_text(json.dumps(gates,indent=2),encoding='utf-8')
  blocked=any(not g['observation'].get('all_eight_free') or g['observation'].get('npu_smi_rc')!=0 or g['owners']['processes'] or any(x.get('own') for x in g['observation'].get('processes',[])) for g in gates.values())
  if blocked:report['status']='occupied_or_uncertain_deferred'
  else:
   def launch(host):
    rank=HOSTS.index(host) if a.nnodes==2 else 0;stage=T+'/runs/'+a.run_tag+'/'+a.stage
    argv=['bash',a.launcher,str(rank),a.stage,a.script,*arguments]
    env={'RUN_TAG':a.run_tag,'NNODES':str(a.nnodes),'NPROC_PER_NODE':str(a.nproc),'LIMIT_SECONDS':str(a.limit),'TRAIN_MASTER_ADDR':'127.0.0.1' if a.nnodes==1 else HOSTS[0]}
    command='set -e\nmkdir -p '+shlex.quote(stage)+'\n'
    command+='if ! mkdir '+shlex.quote(stage+'/launch-marker')+' 2>/dev/null; then printf \'%s\\n\' \'{"status":"already_claimed"}\'; exit 0; fi\n'
    command+=' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' nohup timeout --signal=TERM --kill-after=30 '+str(a.limit+90)+' '+shlex.join(argv)+' > '+shlex.quote(stage+'/supervisor.log')+' 2>&1 < /dev/null &\n'
    command+='pid=$!\nprintf \'%s\\n\' "$pid" > '+shlex.quote(stage+'/supervisor.pid')+'\nprintf \'{"status":"supervisor_started","pid":%s}\\n\' "$pid"\n'
    _,stdout,stderr=backend.clients[host].exec_command(command,timeout=20);data=stdout.read().decode();error=stderr.read().decode();rc=stdout.channel.recv_exit_status()
    if rc:raise RuntimeError((host,rc,error))
    result=json.loads(data);result['launcher_argv']=argv;result['launcher_environment']=env
    return result
   report['gate_to_launch_ms']=1000*(time.monotonic()-finish)
   report['launches']=parallel(launch) if a.nnodes==2 else {a.single_host:launch(a.single_host)}
   report['status']='supervisors_started'
  report['gate_summary']={host:{'time_utc':g['observation']['time_utc'],'free_cards':g['observation']['free_cards'],'own_pids':[r['pid'] for r in g['owners']['own_device_processes']],'foreign_pids':[r['pid'] for r in g['owners']['foreign_device_processes']]} for host,g in gates.items()}
 except Exception as error:report.update(status='control_error',error=repr(error))
 finally:
  backend.close();(out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
  progress_path=Path('outputs/training_device_progress.json')
  if progress_path.exists() and report['status']=='supervisors_started' and all(x.get('status')=='supervisor_started' for x in report.get('launches',{}).values()):
   progress=json.loads(progress_path.read_text(encoding='utf-8'));progress.update(latest_occupancy={'source':'current launch dual gate','gate_summary':report['gate_summary']},device_tasks_currently_running=True,active={'run_tag':a.run_tag,'stage':a.stage,'watchdog_seconds':a.limit},active_stage={'run_tag':a.run_tag,'stage':a.stage,'supervisors':{h:r['pid'] for h,r in report['launches'].items()},'watchdog_seconds':a.limit},next_stage='await active '+a.stage+' result',wait_reason=None);progress_path.write_text(json.dumps(progress,indent=2),encoding='utf-8')
 print(json.dumps(report));return 2 if report['status']=='control_error' else 0


if __name__=='__main__':raise SystemExit(main())
