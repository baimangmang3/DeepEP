"""Only EP16 worker predicate and actual cleanup-return-field regressions."""
import ast,copy,importlib.util,json,pathlib
WS=pathlib.Path(__file__).resolve().parent.parent
DIR=WS/'work/training_diagnostics/p2_ep16_serial_hostname_control'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=load('lifecycle_fields',DIR/'dual_host_lifecycle.py');P=load('actual_cleanup_policy',DIR/'cleanup_policy.py')
tree=ast.parse((DIR/'capture_live_workers.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='matches_worker')
ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_capture_predicate','exec'),ns);match=ns['matches_worker']
run=pathlib.PurePosixPath('/training/runs/RUN_EP16/stage');script='/candidate/validate_training_p2.py';argv=['python',script,'--output',str(run)]
env=dict(RUN_TAG='RUN_EP16',WORLD_SIZE='16',RANK='0',LOCAL_RANK='0');checks=[]
for rank in (0,7,8,15):
    e=dict(env,RANK=str(rank),LOCAL_RANK=str(rank%8));assert match(argv,e,run,script);checks.append('accept global/local rank'+str(rank))
for label,change in (('world2',{'WORLD_SIZE':'2'}),('negative rank',{'RANK':'-1'}),('rank16',{'RANK':'16'}),('negative local',{'LOCAL_RANK':'-1'}),('local8',{'LOCAL_RANK':'8'}),('local mismatch',{'RANK':'8','LOCAL_RANK':'1'}),('wrong RUN',{'RUN_TAG':'OTHER'}),('missing rank',{'RANK':None}),('missing local',{'LOCAL_RANK':None})):
    assert not match(argv,dict(env,**change),run,script);checks.append('reject '+label)
for label,a in (('wrong validator',['python','/other/validate_training_p2.py','--output',str(run)]),('wrong output',['python',script,'--output','/other']),('missing output value',['python',script,'--output'])):
    assert not match(a,env,run,script);checks.append('reject '+label)

class IO:
    def __init__(self,mode=None):self.mode=mode
    def wait_gone(self,*a):return True
    def harvest_rc(self,**k):return dict(node_rc=None if self.mode=='missing_rc' else 0,supervisor_rc=0)
    def recorders_done(self):return self.mode!='missing_rc'
    def save_summary(self,report):
        if self.mode=='summary_before':raise OSError('save/scan before remaining')
        report['remaining']=[]
        if self.mode=='summary_after':raise OSError('write after remaining')
for mode in (None,'missing_rc','summary_before','summary_after'):
    # Executes original cleanup_policy.cleanup with empty ledger and injected IO;
    # no signal function is called, and all actual returned field names retained.
    result=P.cleanup(IO(mode),[],'RUN_EP16')
    assert M.cleanup_is_uncertain({'cleanup':{'host27':result}})==(mode is not None)
    checks.append('actual cleanup return '+str(mode))
result=P.cleanup(IO(),[],'RUN_EP16');result['cleanup_errors']=['failed'];assert M.cleanup_is_uncertain({'cleanup':{'host27':result}});checks.append('actual cleanup_errors keeps uncertain')
result=P.cleanup(IO(),[],'RUN_EP16');result['remaining']=[{'pid':123}];assert M.cleanup_is_uncertain({'cleanup':{'host27':result}});checks.append('actual remaining keeps uncertain')
assert M.cleanup_is_uncertain({'cleanup':{'host27':{'node_rc':0,'supervisor_rc':0,'cleanup_errors':[]}}});checks.append('missing required remaining observation refuses inactive')
other=DIR.with_name('p2_ep16_overlap_hostname_control')
for file in ('capture_live_workers.py','dual_host_lifecycle.py'):assert (DIR/file).read_bytes()==(other/file).read_bytes()
assert 'if not matches_worker(argv,env,run,script):continue' in (DIR/'capture_live_workers.py').read_text();checks.append('both variants exact same tested predicate/lifecycle and actual callsite')
out=dict(passed=True,checks=len(checks),items=checks,scope='Only offline EP16 filter and cleanup return-field checks; no proc scan, remote, Torch, device, original strategy matrix replay.')
(WS/'outputs/p2_ep16_capture_cleanup_cpu_review.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out,indent=2))
