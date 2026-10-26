"""Read-only NPU/process samples. No Torch/native library imports."""
import argparse,json,pathlib,subprocess,time
from process_control import R,PACKAGE,proc,exact,worker_argv
def npu_rows(text):
    active=False;rows=[]
    for line in text.splitlines():
        if 'process id' in line.lower() and 'process name' in line.lower():active=True;continue
        if not active or not line.startswith('|') or 'No running processes' in line:continue
        fields=[v.strip() for v in line.split('|')[1:-1]]
        if len(fields)<4 or not fields[0].isdigit() or not fields[1].isdigit():raise ValueError('Unparsed NPU process row')
        host_pid=int(fields[1]);container_pid=int(fields[4]) if len(fields)>4 and fields[4].isdigit() else None
        rows.append(dict(card=int(fields[0]),host_pid=host_pid,pid=container_pid if container_pid is not None else host_pid,name=fields[2]))
    if not active:raise ValueError('NPU process table header missing')
    return rows
def launchers(tag,worker=None):
    rows=[];errors=[];unknown=[]
    names={'torchrun','bench_ep_gmm_fused.py','benchmark.py','validate_training_p1.py','validate_training_p2.py','validate_component.py','pure_hccl_probe.py'}
    for p in pathlib.Path('/proc').glob('[0-9]*'):
        candidate=False
        try:
            argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
            candidate=any(pathlib.PurePosixPath(v).name in names or v=='torch.distributed.run' or v.endswith('/scripts/launcher.py') for v in argv)
            if not candidate:continue
            row=proc(int(p.name))
            if worker is not None and exact(worker,row):continue
            if row['run_tag']==tag and row['role']=='worker' and row['visible']=='0' and row['argv']==worker_argv(tag):
                unknown.append(dict(pid=row['pid'],candidate_identity=row,reason='same RUN worker born before marker snapshot; identity alignment pending'))
            else:rows.append(row)
        except (OSError,ValueError) as exc:
            if candidate:errors.append(dict(pid=int(p.name),error=repr(exc),proc_missing=isinstance(exc,FileNotFoundError),
                saved_worker_identity=worker if worker and int(p.name)==worker['pid'] else None))
    return dict(controllers=rows,errors=errors,unknown_candidates=unknown,busy_or_uncertain=bool(rows or errors or unknown))
def sample(tag,worker=None):
    result=dict(time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),launchers=launchers(tag,worker),allowed_owned_rows=[],conflicts=[],unknown_rows=[])
    result['npu_smi_before']=dict(time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns())
    smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    result['npu_smi_after']=dict(time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns())
    result.update(npu_smi_rc=smi.returncode,npu_smi=smi.stdout+smi.stderr)
    try:
        rows=npu_rows(smi.stdout);result['npu_rows']=rows
        for row in rows:
            owned=False
            try:
                current=proc(row['pid'])
                if worker is not None and row['card']==0 and row['pid']==worker['pid']:
                    owned=exact(worker,current)
                    if not owned:
                        result['unknown_rows'].append(dict(row,saved_worker_identity=worker,current_proc=current,reason='saved PID/start differs; stale/reuse unknown'));continue
                elif current['run_tag']==tag:
                    result['unknown_rows'].append(dict(row,current_proc=current,reason='RUN matches but no exact worker marker'));continue
            except OSError as exc:
                result['unknown_rows'].append(dict(row,saved_worker_identity=worker if worker and row['pid']==worker['pid'] else None,
                    proc_missing=isinstance(exc,FileNotFoundError),error=repr(exc),reason='NPU row not bound to current proc; stale/unknown, not verified own/foreign'));continue
            (result['allowed_owned_rows'] if owned else result['conflicts']).append(dict(row,current_proc=current))
        accounted=smi.stdout.count('No running processes')+len({v['card'] for v in rows})==8 and all(0<=v['card']<8 for v in rows)
        result['all_cards_accounted']=accounted
        result['passed']=smi.returncode==0 and accounted and not result['launchers']['busy_or_uncertain'] and not result['conflicts'] and not result['unknown_rows']
        result['all_eight_free']=result['passed'] and not rows and smi.stdout.count('No running processes')==8
    except Exception as exc:result.update(passed=False,error=repr(exc),all_eight_free=False)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-tag',required=True);p.add_argument('--worker-identity');a=p.parse_args()
    worker=json.loads(pathlib.Path(a.worker_identity).read_bytes()) if a.worker_identity and pathlib.Path(a.worker_identity).exists() else None
    print(json.dumps(sample(a.run_tag,worker)))
