"""Only new two-host submission/failure cases; no SSH/native/device calls."""
import hashlib,importlib.util,json,pathlib,subprocess,sys
WS=pathlib.Path(__file__).resolve().parent.parent
DIR=WS/'work/training_diagnostics/p2_ep16_serial_hostname_control'
s=importlib.util.spec_from_file_location('dual_lifecycle',DIR/'dual_host_lifecycle.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
checks=[];hosts=('host27','host28')
def scenario(fail=None,recovered=False,pre=False,cleanup_fail=False):
    report={};called=[];stopped=[];first=RuntimeError('first launch failure')
    def launch(host):
        called.append(host)
        if host==fail:raise first
        return {'node':hosts.index(host),'pid':100+hosts.index(host),'start':'123'}
    def recover(host):return {'node':hosts.index(host),'pid':101,'start':'123'} if recovered else None
    def prelaunch(started):
        if pre and started:raise first
    def stop(host):
        stopped.append(host)
        if cleanup_fail and host==hosts[0]:raise OSError('cleanup secondary failure')
        return {'remaining':[],'cleanup_errors':[],'node_rc':0,'supervisor_rc':0}
    try:M.launch_hosts_once(hosts,launch,recover,prelaunch,report)
    except RuntimeError as exc:assert exc is first
    else:assert fail is None and not pre
    M.cleanup_started(stop,report);M.cleanup_started(stop,report)
    assert len(stopped)==len(set(stopped)) and set(stopped)==set(report.get('launches',{}))
    if fail or pre:assert report['first_error']==repr(first)
    return report,called,stopped
r,c,k=scenario(fail=hosts[0]);assert c==[hosts[0]] and k==[] and hosts[0] in r['unconfirmed_launches'] and M.cleanup_is_uncertain(r);checks.append('first host failed/no exact marker: no second host or guessed cleanup')
r,c,k=scenario(fail=hosts[1]);assert c==list(hosts) and k==[hosts[0]];checks.append('only first host confirmed: cleanup only that host')
r,c,k=scenario(fail=hosts[1],recovered=True);assert k==list(hosts);checks.append('second response lost after exact marker: recover both started hosts')
r,c,k=scenario(fail=hosts[1],recovered=True,cleanup_fail=True);assert k==list(hosts) and r['cleanup_errors'][hosts[0]] and r['first_error']=="RuntimeError('first launch failure')" and M.cleanup_is_uncertain(r);checks.append('cleanup first host fails: still cleanup peer, retain original launch error')
r,c,k=scenario(pre=True);assert c==[hosts[0]] and k==[hosts[0]];checks.append('known first-peer failure revokes unstarted host successor')
r,c,k=scenario();assert k==list(hosts) and not M.cleanup_is_uncertain(r);checks.append('both confirmed hosts get independent child-first cleanup, no duplicate cleanup')
for mode in ('serial','overlap'):
    d=DIR.with_name('p2_ep16_'+mode+'_hostname_control');out=WS/'outputs'/('p2_ep16_'+mode+'_hostname_preparation')
    assert (d/'dual_host_lifecycle.py').read_bytes()==(DIR/'dual_host_lifecycle.py').read_bytes()
    shell=(d/'run_dispatch_host_trace_checked.sh').read_text();cleanup=(d/'cleanup_own.py').read_text();controller=(d/'controller.py').read_text()
    for flag in ('--nnodes=2 --nproc-per-node=8','--node-rank="$NODE_RANK"','--master-addr=177.22.27.1','--world-size 16','--E 64 --K 8','--cases balanced,zero_gate,empty,skew','--backward-schedule '+mode,'ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7'):assert flag in shell
    assert "--node-rank '+str(node)" in controller and "'.node{node}-controller-once/identity.json'" in cleanup
    assert 'logs/node{self.node_rank}.rc' in cleanup and 'supervisor-node{self.node_rank}.rc' in cleanup and "outer['node']==node" in cleanup
    assert 'states={h:state(h) for h in launches}' in controller
    manifest=json.loads((out/'rack05os7-diagnostic-deployment-manifest.json').read_bytes());assert manifest['device_authorized'] is False
    assert hashlib.sha256((d/'launcher.py').read_bytes()).hexdigest()=='7e15cc8ebaf480b09ab7d53185b9ad8077990a3a163adac7b7281cb4bf299e4d'
    r=subprocess.run([sys.executable,'-I',str(d/'controller.py'),'--run-tag','CPU_EP16_'+mode+'_plan_only'],cwd=WS,capture_output=True,text=True);assert r.returncode==0,r.stderr
    plan=json.loads(r.stdout);assert plan['stage']=='p2-ep16-bf16-'+mode
    (out/'controller-plan.json').write_bytes((json.dumps(plan,indent=2)+'\n').encode())
    checks.append(mode+' default plan + two-node argv/own-node cleanup + unchanged wrapper static checks')
result=dict(passed=True,checks=len(checks),items=checks,scope='Only new dual-host failure models and local default/static checks. No remote/device/CPU strategy matrix rerun.')
(WS/'outputs/p2_ep16_controls_cpu_review.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result,indent=2))
