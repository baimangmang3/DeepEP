"""Bounded new single-worker fault models only. No SSH/process signal/NPU."""
import ast,contextlib,copy,importlib.util,json,pathlib,subprocess,sys,tempfile,types
from unittest.mock import patch
HERE=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import process_control as C
import controller as K
import resource_gate as G
checks=[]
def child(values):
    class Child:
        pid=123
        def __init__(self):self.actions=[];self.values=list(values);self.returncode=None
        def wait(self,timeout):
            v=self.values.pop(0)
            if isinstance(v,BaseException):raise v
            self.returncode=v;return v
        def poll(self):return self.returncode
        def terminate(self):self.actions.append('TERM')
        def kill(self):self.actions.append('KILL')
    return Child()
real=child([-6]);r=C.wait_worker(real,lambda:None);assert r['worker_rc']==-6 and not r['passed'];checks.append('actual child nonzero rc retained')
timeout=subprocess.TimeoutExpired('worker',180)
r=C.wait_worker(child([timeout,timeout]),lambda:(_ for _ in ()).throw(RuntimeError('cleanup secondary')))
assert r['worker_rc'] is None and r['first_error']=='worker_timeout' and r['cleanup_errors'] and not r['passed'];checks.append('timeout+cleanup first error, unknown rc never synthesized')
r=C.wait_worker(child([timeout,0]),lambda:{'gone':True});assert r['worker_rc']==0 and r['timeout'] and not r['passed'];checks.append('late rc0 cannot erase timeout')
with tempfile.TemporaryDirectory(dir=HERE/'tests') as tmp:
    run=pathlib.Path(tmp);argv=['python','worker.py'];env=dict(RUN_TAG='UNIT',P4_ROLE='worker',ASCEND_RT_VISIBLE_DEVICES='0')
    for mode in ('proc_failure','assert_failure','marker_failure'):
        c=child([-15]);writes={};original=LookupError('first proc failure')
        def read(pid):
            if mode=='proc_failure':raise original
            return dict(pid=123,start='1',uid=0,argv=['wrong'] if mode=='assert_failure' else argv,run_tag='UNIT',visible='0',role='worker')
        def write(path,value):
            if mode=='marker_failure' and path.name=='worker.identity.json':raise OSError('first marker failure')
            writes[path.name]=copy.deepcopy(value)
        with patch.object(C.os,'kill',side_effect=AssertionError('no guessed PID signal')):
            try:C.spawn_owned(argv,env,None,run,'UNIT',popen=lambda *a,**k:c,read_identity=read,write=write)
            except BaseException as exc:
                if mode=='proc_failure':assert exc is original
            else:raise AssertionError('startup failure accepted')
        proof=writes['startup-failure.json'];assert proof['owned_child_pid']==123 and proof['owned_child_reaped'] is True and proof['worker_rc']==-15 and c.actions==['TERM']
        checks.append(mode+' uses owned Popen only, bounded reaping and first error')
    worker=dict(pid=11,start='1',argv=['worker'],role='worker',visible='0',run_tag='UNIT',uid=0)
    for mode in ('live_worker','stop_raises','missing_marker','contradictory_stale_proof'):
        for p in run.iterdir():p.unlink()
        if mode!='missing_marker':C.save(run/'worker.identity.json',worker)
        if mode=='contradictory_stale_proof':C.save(run/'startup-failure.json',dict(run_tag='UNIT',owned_child_created=True,owned_child_reaped=True,worker_rc=0))
        C.save(run/'recorder.identity.json',dict(worker,pid=22,role='recorder'));C.save(run/'supervisor.identity.json',dict(worker,pid=33,role='supervisor'))
        called=[]
        def stop(saved):
            called.append(saved['pid'])
            if mode=='stop_raises':raise RuntimeError('worker identity unknown')
            return {'gone':False}
        with patch.object(C,'run_path',lambda tag:run),patch.object(C,'stop_exact',stop):r=C.cleanup_worker('UNIT',parents=True)
        assert r['worker_exit_or_nonstart_confirmed'] is False and 'parents_preserved' in r and all(pid==11 for pid in called)
        checks.append(mode+' retains recording parents')
    for p in run.iterdir():p.unlink()
    C.save(run/'startup-failure.json',dict(run_tag='UNIT',owned_child_created=False));C.save(run/'supervisor.rc',0)
    C.save(run/'recorder.identity.json',dict(worker,pid=22,role='recorder'));C.save(run/'supervisor.identity.json',dict(worker,pid=33,role='supervisor'));called=[]
    with patch.object(C,'run_path',lambda tag:run),patch.object(C,'stop_exact',lambda saved:called.append(saved['pid']) or {'gone':True}):r=C.cleanup_worker('UNIT',parents=True)
    assert r['worker_exit_or_nonstart_confirmed'] and called==[22,33];checks.append('proven no Popen child allows child-first recorder retirement')

report={};first=RuntimeError('lost response')
try:K.launch_recorded(lambda:(_ for _ in ()).throw(first),lambda:{'pid':123,'start':'1'},report)
except RuntimeError as exc:assert exc is first
assert report['first_error']==repr(first) and report['launch_recovered']['pid']==123;checks.append('launch response lost: exact recovery records, no restart or hidden success')
base=dict(recorder_rc=0,supervisor_rc=0,worker_status=dict(worker_rc=0,timeout=False,passed=True,cleanup_errors=[]))
assert K.real_rcs_pass(base)
for key in ('recorder_rc','supervisor_rc'):
    z=copy.deepcopy(base);z[key]=None;assert not K.real_rcs_pass(z)
z=copy.deepcopy(base);z['worker_status']['worker_rc']=None;assert not K.real_rcs_pass(z);checks.append('all three real rc required; missing child/recorders reject')
text='| NPU ID | Process id | Process name | Process memory(MB) | Process id in container |\n| 0 | 100 | python | 10 | 123 |\n'+''.join('| No running processes found in NPU '+str(i)+' |\n' for i in range(1,8))
assert G.npu_rows(text)[0]['pid']==123
worker=dict(pid=123,start='7',role='worker',visible='0',run_tag='UNIT',uid=0,argv=['worker'])
with patch.object(G,'launchers',lambda *a:{'controllers':[],'errors':[],'busy_or_uncertain':False}),patch.object(G.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0,stdout=text,stderr='')),patch.object(G,'proc',side_effect=FileNotFoundError('raced exit')):sample=G.sample('UNIT',worker)
assert not sample['passed'] and not sample['allowed_owned_rows'] and not sample['conflicts'] and sample['unknown_rows'][0]['proc_missing'];checks.append('missing proc NPU row is saved stale/unknown, never verified own/foreign/free')
history=[dict(observations={'host':sample})];state=dict(base,worker_identity=worker)
assert not K.resolve_unknowns(history,state)
state['worker_identity']=dict(worker,start='8');assert K.resolve_unknowns(history,state);checks.append('unknown sample resolves only to same saved worker birth plus real waitpid exit')

born=dict(worker,argv=C.worker_argv('UNIT'))
with tempfile.TemporaryDirectory(dir=HERE/'tests') as tmp:
    p=pathlib.Path(tmp)/'123';p.mkdir();(p/'cmdline').write_bytes(b'\0'.join(v.encode() for v in born['argv']))
    with patch.object(G.pathlib.Path,'glob',lambda *a:[p]),patch.object(G,'proc',lambda pid:born):race=G.launchers('UNIT')
    assert not race['controllers'] and len(race['unknown_candidates'])==1 and race['busy_or_uncertain']
    assert K.runtime_decision({'host':dict(npu_smi_rc=0,all_cards_accounted=True,launchers=race)}, {})=='unresolved_observation'
    history=[dict(observations={'host':dict(launchers=race)})]
    assert not K.resolve_unknowns(history,dict(worker_identity=born))
    assert K.resolve_unknowns(history,dict(worker_identity=dict(born,start='other')))
    checks.append('worker born between state/sample stays unknown until exact later marker, never foreign')
    for wrong in (dict(born,run_tag='EXTERNAL'),dict(born,argv=born['argv']+['--unexpected'])):
        with patch.object(G.pathlib.Path,'glob',lambda *a:[p]),patch.object(G,'proc',lambda pid:wrong):foreign=G.launchers('UNIT')
        assert foreign['controllers']==[wrong] and not foreign['unknown_candidates']
    checks.append('external same-name or same-RUN wrong-argv worker remains rejected')
with patch.object(G,'launchers',lambda *a:dict(controllers=[],errors=[],busy_or_uncertain=False)),patch.object(G.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0,stdout=text,stderr='')),patch.object(G,'proc',lambda pid:dict(born,run_tag='EXTERNAL',ppid=77)):
    foreign_sample=G.sample('UNIT')
assert foreign_sample['conflicts'][0]['current_proc']['ppid']==77
assert foreign_sample['npu_smi_before']['monotonic_ns']<=foreign_sample['npu_smi_after']['monotonic_ns']
checks.append('foreign exact proc birth/argv/PPID retained with npu-smi local time window')
expected=C.T+'/runs/UNIT/'+C.STAGE+'/component-results.json'
assert C.worker_argv('UNIT')[-1]==expected and '\\' not in C.worker_argv('UNIT')[-1]
assert (pathlib.PureWindowsPath(C.T)/'runs'/'UNIT'/C.STAGE/'component-results.json').as_posix()==expected
assert 'run_path(a.run_tag).as_posix()' in (HERE/'controller.py').read_text()
checks.append('Windows local path serialized as exact POSIX remote worker/report path')
m=json.loads((HERE/'manifest.json').read_bytes());assert K.local_backend_identity(m)['sha256']==m['local_backend']['sha256']
bad=copy.deepcopy(m);bad['local_backend']['sha256']='0'*64
try:K.local_backend_identity(bad)
except ValueError:pass
else:raise AssertionError('backend change accepted')
checks.append('actual imported local SSH backend bytes pinned and mismatch rejected')
commands=[]
class Stream:
    channel=types.SimpleNamespace(recv_exit_status=lambda:0)
    def __init__(self,data):self.data=data
    def read(self):return self.data
def remote(command,timeout):
    commands.append((command,timeout));return None,Stream(b'{"gone":true}'),Stream(b'')
backend=types.SimpleNamespace(clients={'host':types.SimpleNamespace(exec_command=remote)})
assert K.cleanup_call(backend,'host','UNIT','print(1)')=={'gone':True}
assert ' 200 ' in commands[0][0] and commands[0][1]==215 and '125' not in commands[0][0]
checks.append('cleanup uses independent bounded200s call rather than inherited125s cutoff; no SSH executed')
post=dict(npu_smi_rc=0,all_cards_accounted=True,all_eight_free=False,conflicts=[{'pid':999}],launchers=dict(controllers=[{'pid':999}],errors=[],unknown_candidates=[]))
released=dict(own_remaining=[],own_identity_unknown=[])
assert K.release_decision({'host':post},released)=='released_post_busy'
assert K.release_decision({'host':post},dict(released,own_identity_unknown=[{'pid':123}]))=='own_release_unconfirmed'
unknown=copy.deepcopy(post);unknown['unknown_rows']=[{'pid':123,'proc_missing':True}]
assert K.release_decision({'host':unknown},released)=='own_release_unconfirmed'
# Decision is pure: the caller immediately breaks on released_post_busy; no poll/signal.
body=(HERE/'controller.py').read_text();assert "if release in ('released_free','released_post_busy'):" in body
checks.append('post busy after exact own release stops polling, without signaling or invalidating component; own unknown rejects')
assert K.runtime_decision({'host':dict(npu_smi_rc=0,all_cards_accounted=False,launchers={})},{})=='resource_sample_error'
checks.append('inflight seven-card incomplete table rejects as resource_sample_error')
# Execute the actual main monitoring-loop AST, not a mirror of its decisions.
tree=ast.parse((HERE/'controller.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
loop=next(n for n in ast.walk(main) if isinstance(n,ast.While) and isinstance(n.test,ast.Compare) and any(isinstance(v,ast.Name) and v.id=='deadline' for v in ast.walk(n.test)))
state=copy.deepcopy(base);state.update(own_remaining=[],own_identity_unknown=[],worker_identity=born)
counts={'sample':0,'state':0}
def get_state():counts['state']+=1;return state
def get_sample(*args):counts['sample']+=1;return post
def forbidden(*args):raise AssertionError('post-completion sample entered inflight decision/sleep/signal')
with tempfile.TemporaryDirectory(dir=HERE/'tests') as tmp:
    env=dict(vars(K),current=None,state=get_state,sample=get_sample,parallel=lambda f:{'host':f('host')},HOSTS=['host'],runtime_decision=forbidden,
        time=types.SimpleNamespace(monotonic=lambda:0,sleep=forbidden),deadline=360,report={'samples':[]},a=types.SimpleNamespace(run_tag='UNIT'),
        call=lambda *args:{'worker':{'pid':born['pid'],'start_ticks':born['start']}},dest=pathlib.Path(tmp),audit=lambda *args:{'passed':True},m={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'actual-main-loop','exec'),env)
    assert env['report']['passed'] and env['report']['post_busy'] and env['report']['release_decision']=='released_post_busy'
    assert counts=={'sample':1,'state':1}
checks.append('actual main loop real-completion before first foreign sample: one state/sample, audit success, no retry/signal/inflight decision')

program="import runpy,sys;sys.argv=["+repr(str(HERE/'controller.py'))+",'--run-tag','CPU_ONLY_PLAN'];runpy.run_path("+repr(str(HERE/'controller.py'))+",run_name='__main__');assert 'torch' not in sys.modules and 'paramiko' not in sys.modules"
r=subprocess.run([sys.executable,'-I','-c',program],capture_output=True,text=True);assert r.returncode==0,r.stderr;plan=json.loads(r.stdout);assert plan['mode']=='CPU_plan';checks.append('actual controller default plan no Torch/SSH import or execute')
for name in ('worker_recorder.py','supervisor.py'):
    r=subprocess.run([sys.executable,'-I',str(HERE/name)],capture_output=True,text=True);assert r.returncode==0 and json.loads(r.stdout)['action']=='plan only'
checks.append('both recording actors no-execute default plans')
assert not any(n=='torch' or n.startswith(('torch.','torch_npu','deep_ep')) for n in sys.modules)
out=dict(passed=True,checks=len(checks),items=checks,scope='Only new single-worker lifecycle/resource-race fault models and default CPU plans; no subprocess worker/native, signals, SSH, NPU or old comprehensive regression.')
(HERE/'tests/cpu_results.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out,indent=2))
