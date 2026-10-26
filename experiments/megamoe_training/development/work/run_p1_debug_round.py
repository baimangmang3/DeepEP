"""Same frozen host-v2 precision, only process-local log observations added."""
import hashlib,json,pathlib
from run_p1_host_v2_round import P1Backend,T,CP
from training_native_round import parallel
import training_stage_round

class LoggingBackend(P1Backend):
 def __init__(self):
  super().__init__();self.helper_hashes[T+'/scripts/run_training_p1_host_v2_debug_checked.sh']=hashlib.sha256(pathlib.Path('work/run_training_p1_host_v2_debug_checked.sh').read_bytes()).hexdigest()
 def connect(self):
  super().connect()
  def prepare(h):
   with self.clients[h].open_sftp() as s:s.put('work/run_training_p1_host_v2_debug_checked.sh',T+'/scripts/run_training_p1_host_v2_debug_checked.sh')
   return self.python(h,'p1_logging_cpu_preflight',f'''import pathlib,shutil,subprocess,json
root=pathlib.Path({T!r});u=shutil.disk_usage(root)
assert u.free>=20*(1<<30),'Insufficient log disk headroom'
r=subprocess.run(['bash','-n',str(root/'scripts/run_training_p1_host_v2_debug_checked.sh')],capture_output=True,text=True);assert r.returncode==0,r.stderr
print(json.dumps({{'disk_total_bytes':u.total,'disk_free_bytes':u.free,'required_log_headroom_bytes':20*(1<<30),'bash_n_rc':0,'only_environment_delta':{{'ASCEND_GLOBAL_LOG_LEVEL':'0','ASCEND_PROCESS_LOG_PATH':'<this run>/ascend-log'}}}}))''')
  rows=parallel(prepare);out=pathlib.Path('outputs/p1_debug_recovery_plan');out.mkdir(exist_ok=True);(out/'latest-cpu-log-headroom.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
 training_stage_round.SshBackend=LoggingBackend
 raise SystemExit(training_stage_round.main())
