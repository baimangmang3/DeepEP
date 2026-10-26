"""Only coverage union/used-source/CPU lifecycle regressions. No actual tool run."""
import copy,json,pathlib,signal,struct,subprocess,tempfile,types
from unittest.mock import patch
import compile_objects as C
from cpu_recorder import run_recorded,LinuxOps
HERE=pathlib.Path(__file__).resolve().parent;checks=[]
def reject(f):
    try:f()
    except (ValueError,AssertionError):return
    raise AssertionError('bad evidence accepted')
archive=HERE.parents[2]/'outputs/p1_1431_offline_pc_review'
# HERE.parents[2] is workspace root, independent of source snapshots.
inventory=json.loads((archive/'embedded-elf-inventory.json').read_bytes());dis=json.loads((archive/'disassembly-results.json').read_bytes())
for name in ('embedded-938d8.elf','embedded-22c900.elf'):
    p=archive/'actual-wheel-artifacts'/name;b=p.read_bytes();e=next(v for v in inventory if pathlib.PureWindowsPath(v['path']).name==name)
    assert C.pin(p)['sha256']==e['sha256'] and struct.unpack_from('<H',b,18)[0]==e['machine']==4137
    text=next(c['text'] for r in dis['results'] if r['elf']==name for c in r['commands'] if c['argv'][0]=='readelf')
    assert C.defined_functions(text)=={'dispatch_kernel'}
checks.append('two actual archived ELF hashes/machine4137 and original readelf FUNC evidence verified offline')
required=['dispatch_kernel','dispatch_profile_kernel']
elfs=[dict(machine=4137,readelf_rc=0,defined_required_symbols=[required[0]]),dict(machine=4137,readelf_rc=0,defined_required_symbols=[required[1]])]
assert C.device_symbol_sources(elfs,4137,required)=={required[0]:[0],required[1]:[1]};checks.append('separate matching device ELFs satisfy union with per-symbol provenance')
bad=copy.deepcopy(elfs);bad[1]['machine']=183;reject(lambda:C.device_symbol_sources(bad,4137,required))
text='  1: 000000 0 FUNC GLOBAL DEFAULT UND dispatch_kernel\n  2: 00000 8 OBJECT GLOBAL DEFAULT 2 dispatch_profile_kernel\nrandom dispatch_kernel text\n'
assert not C.defined_functions(text);checks.append('host ELF, undefined FUNC, metadata OBJECT and substring cannot supply device symbol')
with tempfile.TemporaryDirectory(dir=HERE) as tmp:
    tmp=pathlib.Path(tmp);root=tmp/'snapshots/original';root.mkdir(parents=True);src=root/'a.h';src.write_bytes(b'bound');extra=root/'extra.h';extra.write_bytes(b'unbound')
    m={'snapshot_files':{'original':{'a.h':C.pin(src)}}};bindings={'external_dependencies':{}}
    with patch.object(C,'HERE',tmp):
        used=C.verify_dependencies([str(src)],m,'original',bindings);assert used[str(src.resolve())]==C.pin(src)
        reject(lambda:C.verify_dependencies([str(extra)],m,'original',bindings))
        src.write_bytes(b'replaced');reject(lambda:C.verify_dependencies([str(src)],m,'original',bindings))
    checks.append('new.d root-contained unbound file rejects; bound actual source mutation rejects')

class FakeOps:
    def __init__(self,mode):self.mode=mode;self.now=0.;self.driver=mode not in ('ok','nonzero','exited_child');self.child=self.driver or mode=='exited_child';self.events=[];self.poll_failed=False
    def clock(self):return self.now
    def sleep(self,n):self.now+=max(1,n)
    def spawn(self,*args):return self
    pid=101
    def poll(self):
        if self.mode=='body_exception' and not self.poll_failed:self.poll_failed=True;raise RuntimeError('first body poll error')
        return None if self.driver else (-7 if self.mode=='nonzero' else getattr(self,'rc',0))
    def wait(self,timeout):
        self.events.append(('wait',101))
        if self.driver:self.now+=timeout;raise subprocess.TimeoutExpired('tool',timeout)
        return self.poll()
    def row(self,pid):return dict(pid=pid,start=str(pid)+'birth',uid=7,ppid=101 if pid==102 else 99,session=101,state='R',argv=['CPUtool'])
    def identity(self,pid):
        if self.mode=='startup':raise OSError('first identity error')
        return self.row(pid)
    def scan(self,*args):
        if self.mode=='scan_exception':raise OSError('first scan error')
        return ([self.row(p) for p,live in ((101,self.driver),(102,self.child)) if live],[])
    def current(self,saved):
        if self.mode=='unknown' and saved['pid']==102:raise OSError('child state unknown')
        return self.row(saved['pid']) if (self.driver if saved['pid']==101 else self.child) else None
    def signal(self,saved,sig):
        self.events.append((sig,saved['pid']))
        if self.mode=='kill' and sig==signal.SIGTERM:return True
        if saved['pid']==102:self.child=False
        else:self.driver=False;self.rc=-int(sig)
        return True
    def terminate(self):self.events.append(('direct_TERMINATE',101));self.driver=False;self.rc=-15
    def kill(self):self.driver=False;self.rc=-9
for mode in ('ok','nonzero','timeout','kill','unknown','startup'):
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        p=pathlib.Path(tmp);ops=FakeOps(mode)
        r=run_recorded(['readelf' if mode=='timeout' else 'CPUtool'],p,p/'stdout',p/'stderr',p/'status',limit=1,term_seconds=1,kill_seconds=1,reap_seconds=1,ops=ops)
        assert r==json.loads((p/'status').read_bytes())
        if mode=='ok':assert r['passed'] and r['returncode']==0 and r['released']
        elif mode=='nonzero':assert not r['passed'] and r['returncode']==-7
        elif mode in ('timeout','kill'):
            assert r['first_error']=='tool_timeout' and r['timeout'] and r['returncode']==(-15 if mode=='timeout' else -9) and r['released'] and not r['passed']
            signals=[e for e in ops.events if isinstance(e[0],int)]
            assert signals[0][1]==102 and next(i for i,e in enumerate(signals) if e[1]==101)>0
        elif mode=='unknown':assert 'child state unknown' in r['first_error'] and r['returncode'] is None and not r['released'] and r['cleanup_errors']
        else:assert 'first identity error' in r['first_error'] and r['returncode'] is None and not r['released'] and not r['signals']
        for v in r['signals']:assert v['start']==str(v['pid'])+'birth' and type(v['signal']) is int and v['time_ns']>0 and 'monotonic' in v
    checks.append(mode+' recorder: actual wait rc, child-first cleanup, timeout first cause, unknown release fail-closed')
for mode in ('body_exception','scan_exception','save_exception','exited_child'):
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        p=pathlib.Path(tmp);ops=FakeOps(mode);original=pathlib.Path.write_bytes;failed=[]
        def write(path,content):
            if mode=='save_exception' and path==p/'status' and not failed:failed.append(True);raise OSError('first save error')
            return original(path,content)
        with patch.object(pathlib.Path,'write_bytes',write):r=run_recorded(['CPUtool'],p,p/'stdout',p/'stderr',p/'status',limit=1,term_seconds=1,kill_seconds=1,reap_seconds=1,ops=ops)
        assert not r['passed']
        if mode=='scan_exception':assert 'first scan error' in r['first_error'] and not r['released'] and not r['signals'] and r['returncode'] is None
        elif mode=='exited_child':assert r['first_error']=='driver_exited_with_live_descendants' and r['returncode']==0 and r['released'] and all(s['pid']==102 for s in r['signals'])
        else:
            assert ('first body poll error' if mode=='body_exception' else 'first save error') in r['first_error']
            assert [s['pid'] for s in r['signals']]==[102,101] and r['returncode']==-15 and r['released']
    checks.append(mode+' afterspawn: unified exact child-first cleanup or preserve unknown driver; first error/real rc retained')
old=dict(pid=101,start='original',uid=7,ppid=99,session=101,state='R',argv=['oldtool'])
reused=dict(old,start='replacement')
with patch.object(pathlib.Path,'glob',lambda *args:[types.SimpleNamespace(name='101')]),patch.object(LinuxOps,'identity',lambda self,pid:reused):
    found,errors=LinuxOps().scan(old,{(101,'original',7):old})
assert not found and errors;checks.append('reused driver PID/session without original birth stays unknown and is never adopted as owned')
out=dict(passed=True,checks=len(checks),items=checks,real_compiler_or_readelf_executed=False,native_or_NPU=False)
(HERE/'revision_cpu_results.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out,indent=2))
