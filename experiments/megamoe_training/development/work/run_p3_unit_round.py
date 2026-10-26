"""Deploy exact standalone P3 unit inputs; dual-host gate -> one card0 child."""
import hashlib,json,pathlib
from training_native_round import SshBackend,T,HOSTS,parallel
import training_stage_round
SRC=pathlib.Path('work/training_candidates/p3_up_swiglu');D=T+'/candidate-p3-up-swiglu-20261025'
SO=pathlib.Path('outputs/p3_up_swiglu_build/libmegamoe_up_swiglu.so')
SO_SHA='4623d9abf673210f0e3e07473f9f02c541256d699e47ce0a2a2dd7229c8c856b'
assert hashlib.sha256(SO.read_bytes()).hexdigest()==SO_SHA
M=json.loads((SRC/'manifest.json').read_text());FILES=M['source_sha256']
assert FILES['native/up_swiglu.cpp']=='66f60053da1919d4c094ab6a973b060f93556ed5c8ce82cc37140dab700cc4ef'
assert FILES['test_up_swiglu_npu.py']=='813e098133422833d158915d08b830b55ef53d8f4ce262eb8e2262806d404809'
for n,h in FILES.items():assert hashlib.sha256((SRC/n).read_bytes()).hexdigest()==h,n
FILES=dict(FILES,**{'manifest.json':hashlib.sha256((SRC/'manifest.json').read_bytes()).hexdigest()})
OUT=pathlib.Path('outputs/p3_up_swiglu_unit_deployment');OUT.mkdir(exist_ok=True)

class UnitBackend(SshBackend):
 def __init__(self):
  super().__init__();self.helper_hashes.update({D+'/'+n:h for n,h in FILES.items()});self.helper_hashes[D+'/build/'+SO.name]=SO_SHA
  for n in ('p3_unit_supervisor.py','run_p3_unit_checked.sh'):self.helper_hashes[T+'/scripts/'+n]=hashlib.sha256(pathlib.Path('work',n).read_bytes()).hexdigest()
 def connect(self):
  super().connect()
  def deploy(h):
   self.python(h,'p3_unit_cpu_deploy',f'''import pathlib,json,hashlib
p=pathlib.Path({D!r});(p/'native').mkdir(parents=True,exist_ok=True);(p/'build').mkdir(exist_ok=True)
if (p/'build/libmegamoe_up_swiglu.so').exists():assert hashlib.sha256((p/'build/libmegamoe_up_swiglu.so').read_bytes()).hexdigest()=={SO_SHA!r}
print(json.dumps({{'directory':str(p)}}))''')
   with self.clients[h].open_sftp() as s:
    for n in FILES:s.put(str(SRC/n),D+'/'+n+'.tmp');s.posix_rename(D+'/'+n+'.tmp',D+'/'+n)
    if h!=HOSTS[0]:s.put(str(SO),D+'/build/'+SO.name)
    for n in ('p3_unit_supervisor.py','run_p3_unit_checked.sh'):s.put('work/'+n,T+'/scripts/'+n)
   return self.python(h,'p3_unit_cpu_plan',f'''import subprocess,py_compile,pathlib,json
p=pathlib.Path({D!r})
for f in p.glob('*.py'):py_compile.compile(str(f),doraise=True)
py_compile.compile({T!r}+'/scripts/p3_unit_supervisor.py',doraise=True)
c=subprocess.run(['bash','-n',{T!r}+'/scripts/run_p3_unit_checked.sh'],capture_output=True,text=True);assert c.returncode==0,c.stderr
r=subprocess.run(['/root/miniconda3/envs/zzy/bin/python',str(p/'test_up_swiglu_npu.py')],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr
print(json.dumps({{'bash_n_rc':0,'py_compile':True,'default_plan':json.loads(r.stdout)}}))''')
  r=parallel(deploy);(OUT/'cpu-plans.json').write_text(json.dumps(r,indent=2));(OUT/'source-manifest.json').write_text(json.dumps(FILES,indent=2))
 def verify(self,h,tag):
  r=super().verify(h,tag)
  duplicate=self.python(h,tag,f'''import pathlib,json
rows=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=[v.decode(errors='replace') for v in p.joinpath('cmdline').read_bytes().split(b'\\0') if v]
  if '{D}/test_up_swiglu_npu.py' in args or '{T}/scripts/p3_unit_supervisor.py' in args:rows.append({{'pid':int(p.name),'argv':args}})
 except OSError:pass
print(json.dumps(rows))''')
  assert not duplicate,duplicate
  return r

if __name__=='__main__':
 training_stage_round.SshBackend=UnitBackend
 raise SystemExit(training_stage_round.main())
