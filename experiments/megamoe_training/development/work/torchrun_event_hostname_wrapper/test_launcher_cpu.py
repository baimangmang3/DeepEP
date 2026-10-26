"""No Torch/DNS/worker launch: pinned-source AST and isolated importer stubs."""
import ast,contextlib,copy,hashlib,importlib.util,io,json,os,pathlib,sys,tempfile,types
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parent;WS=ROOT.parents[1]
s=importlib.util.spec_from_file_location('hostname_launcher',ROOT/'launcher.py');L=importlib.util.module_from_spec(s);s.loader.exec_module(L)
source=WS/'work/torch_elastic_exit_dns_review/rack05os7/site-packages/torch/distributed/elastic/agent/server/api.py'
raw=source.read_bytes();lf=raw.replace(b'\r\n',b'\n')
assert hashlib.sha256(lf).hexdigest()==L.API_SHA256
checks=[]
def fail_dns(*a,**k):raise AssertionError('DNS call forbidden in this test')
with tempfile.TemporaryDirectory() as temp:
    path=pathlib.Path(temp)/'api.py';path.write_bytes(lf)
    proof=L.verify_source(path);assert proof==dict(helper_line=389,call_line=854,owner='_construct_event',metadata_key='hostname');checks.append('exact installed LF pin and only event metadata helper reference')
    for host in ('rack05os7','rack05os8'):
        p=WS/'work/torch_elastic_exit_dns_review'/host/'site-packages/torch/distributed/elastic/agent/server/api.py'
        assert hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==L.API_SHA256
    checks.append('both archived source copies LF normalize to installed raw SHA; runtime never normalizes')
    def fixture(action):
        namespace={'__name__':L.API,'socket':types.SimpleNamespace(getfqdn=fail_dns,gethostname=lambda:'unused')}
        fn=next(n for n in ast.parse(lf).body if isinstance(n,ast.FunctionDef) and n.name=='_get_fq_hostname')
        exec(compile(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[]),str(path),'exec'),namespace)
        api=types.SimpleNamespace(__file__=str(path),_get_fq_hostname=namespace['_get_fq_hostname']);original=api._get_fq_hostname
        order=[];events=[];hostcalls=[]
        def fake_main():
            assert sys.argv is argv and os.environ==env
            events.extend([dict(hostname=api._get_fq_hostname(),state=state) for state in ('START','SUCCEEDED')])
            return action()
        def importer(name):
            order.append(name)
            if name==L.API:return api
            if name=='torch.distributed.run':return types.SimpleNamespace(main=fake_main)
            raise AssertionError(name)
        def hostname():hostcalls.append('gethostname');return 'rack05os7'
        return api,original,order,events,hostcalls,importer,hostname
    argv=['launcher.py','--nnodes=2','--nproc-per-node=8','--master-addr=177.22.27.1','--master-port=29511','worker.py','--shared'];env=dict(os.environ)
    with patch.object(sys,'argv',argv),patch('socket.getfqdn',fail_dns),patch('socket.gethostbyaddr',fail_dns),contextlib.redirect_stderr(io.StringIO()) as diagnostic:
        for expected in (None,17):
            api,original,order,events,hostcalls,importer,hostname=fixture(lambda:expected)
            assert L.main(importer=importer,gethostname=hostname)==expected
            assert api._get_fq_hostname is original and hostcalls==['gethostname']
            assert order==[L.API,'torch.distributed.run'] and len(events)==2 and all(v['hostname']=='rack05os7' for v in events)
        checks.append('no DNS; one cached hostname; original argv/env and main return unchanged; events retained; restore')
        for error in (SystemExit(9),RuntimeError('original launcher failure')):
            def action():raise error
            api,original,order,events,hostcalls,importer,hostname=fixture(action)
            try:L.main(importer=importer,gethostname=hostname)
            except BaseException as got:assert got is error
            else:raise AssertionError('launcher error swallowed')
            assert api._get_fq_hostname is original
        checks.append('SystemExit code and original launcher exception identity propagate')
        api,original,order,events,hostcalls,importer,hostname=fixture(lambda:None)
        path.write_bytes(lf+b'\n')
        try:L.main(importer=importer,gethostname=hostname)
        except RuntimeError:pass
        else:raise AssertionError('hash mismatch accepted')
        assert order==[L.API] and not events and api._get_fq_hostname is original
        checks.append('raw hash mismatch stops before run import/event/worker path')
        path.write_bytes(lf)
        api._get_fq_hostname=lambda:'preexisting-hook'
        try:L.main(importer=importer,gethostname=hostname)
        except RuntimeError:pass
        else:raise AssertionError('external hook overwritten')
        checks.append('preexisting function override rejected')
        L._ACTIVE=True
        try:
            try:L.main(importer=importer,gethostname=hostname)
            except RuntimeError:pass
            else:raise AssertionError('reentry accepted')
        finally:L._ACTIVE=False
        checks.append('reentrant active override rejected')
    assert 'elastic-agent-event-hostname=gethostname-cached' in diagnostic.getvalue()
assert not any(k=='torch' or k.startswith(('torch.','torch_npu','deep_ep')) for k in sys.modules)
result=dict(passed=True,checks=len(checks),items=checks,wrapper_sha256=hashlib.sha256((ROOT/'launcher.py').read_bytes()).hexdigest(),scope='Offline pinned source AST and importer stubs only; no actual Torch import, DNS, worker/process-group/device or deployment')
(ROOT/'cpu_results.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf8'));print(json.dumps(result,indent=2))
