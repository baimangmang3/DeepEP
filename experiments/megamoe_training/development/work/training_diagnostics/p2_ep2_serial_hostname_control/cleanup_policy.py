"""CPU policy prototype only. All process IO, signals and waits are injected.

Not imported by a deployed controller. Unknown/mixed groups are never signalled.
"""
INNER={'worker','torchrun'}
PARENT={'node_recorder','outer_recorder'}

def signal_checked(io,ledger,tag,roles,sig,events):
    for group in sorted({r['pgid'] for r in ledger if r['role'] in roles}):
        members=io.members(group)
        expected={r['pid']:r for r in ledger if r['pgid']==group}
        safe=bool(members) and all(
            p['pid'] in expected and p['start']==expected[p['pid']]['start']
            and p['run_tag']==tag and p['pgid']==group
            and expected[p['pid']]['role'] in roles for p in members)
        event=dict(pgid=group,signal=sig,safe=safe,members=members,
                   time_ns=io.time_ns(),monotonic_ns=io.monotonic_ns())
        if safe:
            try:io.signal_group(group,sig);event['sent']=True
            except Exception as exc:event['signal_error']=repr(exc)
        else:event['refused']='unknown, reused, foreign, mixed role or empty group'
        events.append(event)

def cleanup(io,ledger,tag,first_error=None):
    report=dict(first_error=first_error,events=[],cleanup_errors=[],node_rc=None,
                supervisor_rc=None,recorders_preserved_for_harvest=True)
    try:
        signal_checked(io,ledger,tag,INNER,15,report['events'])
        if not io.wait_gone(INNER,30):
            signal_checked(io,ledger,tag,INNER,9,report['events'])
            io.wait_gone(INNER,10)
        # No process returning None is mapped to a synthetic signal-derived rc.
        report.update(io.harvest_rc(deadline_seconds=40))
        if not io.recorders_done():
            # Innermost recorder first; outer recorder remains available to save rc.
            for role in ('node_recorder','outer_recorder'):
                signal_checked(io,ledger,tag,{role},15,report['events'])
                if not io.wait_gone({role},10):
                    signal_checked(io,ledger,tag,{role},9,report['events'])
                    io.wait_gone({role},5)
                report.update(io.harvest_rc(deadline_seconds=5))
    except Exception as exc:report['cleanup_errors'].append(repr(exc))
    finally:
        # Summary recorder is deliberately outside every signalled group.
        try:io.save_summary(report)
        except Exception as exc:report['summary_save_error']=repr(exc)
    return report

def external_launcher_evidence(rows,own_tag,known_wrappers=()):
    """Input is a bounded read-only process census, no NPU or env dumping.

    Adapter must retain PID/start/PPID/argv and only optional whitelisted visible
    keys: ASCEND_RT_VISIBLE_DEVICES, ASCEND_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES.
    Actual reads and command registry require separate deployment review.
    """
    result=[]
    for row in rows:
        argv=row.get('argv',[])
        recognized=any(a in known_wrappers or a in ('torchrun','torch.distributed.run','deep_ep_experimental.megamoe.benchmark') or a.endswith('/torchrun')
                       or a.endswith('/bench_ep_gmm_fused.py') or a.endswith('/benchmark.py') for a in argv)
        if recognized and row.get('run_tag')!=own_tag:
            result.append({k:row[k] for k in ('pid','start','ppid','argv','visible') if k in row})
    return dict(busy_or_uncertain=bool(result),controllers=result,
                inference='active controller presence, not proof of current/prior NPU execution')
