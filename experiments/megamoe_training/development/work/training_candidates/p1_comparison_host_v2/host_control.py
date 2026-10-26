"""Local host-only shutdown helpers, derived from approved host-v2 validator.
No dependency on validator paths at runtime; bound in comparison harness SHA.
"""
def phase(name, rank=None, **fields):
    """Best-effort flushed diagnostics; logging must not skip host convergence."""
    import json, os, time
    if rank is None:
        rank = os.environ.get('RANK', 'unknown')
    record = dict(phase=name, rank=rank, time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(), **fields)
    try:
        print(json.dumps(record), flush=True)
    except OSError:
        pass

def close_case_host(engine, host, report, case, prior_error=None):
    """Host-only error convergence, not a repair of SHMEM finalization."""
    close_error = None
    phase('case_close_before', host.rank(), case=case)
    try:
        if engine is not None:
            engine.close()
    except Exception as exc:
        close_error = exc
    phase('case_close_after', host.rank(), case=case, passed=close_error is None)
    case_index = len(report.get('case_cleanup', []))
    stage = 'comparison/case-close/' + str(case_index) + '/' + case
    local = {'rank': host.rank(), 'case': case, 'case_index': case_index, 'stage': stage, 'close_error': None if close_error is None else repr(close_error), 'body_error': None if prior_error is None else repr(prior_error)}
    item = dict(local)
    report.setdefault('case_cleanup', []).append(item)
    try:
        peers = host.all_gather(local, stage)
        if [value.get('rank') for value in peers] != list(range(host.size())) or any((value.get('stage') != stage or value.get('case') != case for value in peers)):
            raise RuntimeError('Host close status payload/stage mismatch')
        item['peer_status'] = peers
        item['all_ranks_closed'] = not any((value['close_error'] for value in peers))
        item['all_ranks_case_passed'] = not any((value['close_error'] or value['body_error'] for value in peers))
        phase('case_close_host_status', host.rank(), case=case, passed=item['all_ranks_case_passed'])
    except Exception as exc:
        report['passed'] = False
        item['host_exchange_error'] = repr(exc)
        host.abort('case close status exchange failed: ' + repr(exc))
        raise RuntimeError('Host close convergence failed; no next case permitted') from exc
    if not item['all_ranks_case_passed']:
        report['passed'] = False
        failure = close_error if close_error is not None else prior_error
        raise RuntimeError('A case body/close failed; no next case permitted: ' + (repr(failure) if failure is not None else 'peer failure; inspect peer_status')) from failure

def finish_host_report(host, report, dist, save, prior_error=None):
    """Keep host coordination alive until all ranks reach PG teardown."""
    errors = []
    if prior_error is not None:
        report['passed'] = False
    local = {'rank': host.rank(), 'error': None if prior_error is None else repr(prior_error)}
    try:
        peers = host.all_gather(local, 'comparison/before-PG-teardown')
        report['peer_before_PG_teardown'] = peers
        if any((peer['error'] for peer in peers)):
            report['passed'] = False
            errors.append('A peer reported an original pre-teardown error')
    except Exception as exc:
        errors.append('host pre-teardown: ' + repr(exc))
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:
        errors.append('PG teardown: ' + repr(exc))
    try:
        peers = host.all_gather({'rank': host.rank(), 'errors': errors}, 'comparison/after-PG-teardown')
        if any((peer['errors'] for peer in peers)):
            report['passed'] = False
            errors.append('A peer reported PG/host teardown failure')
        report['peer_exit_status'] = peers
    except Exception as exc:
        errors.append('host exit status: ' + repr(exc))
    try:
        host.close()
    except Exception as exc:
        errors.append('host close: ' + repr(exc))
    if errors:
        report['cleanup_errors'] = errors
        report['passed'] = False
    try:
        save()
    except Exception as exc:
        report['passed'] = False
        report['report_io_error'] = repr(exc)
        raise
    if errors:
        raise RuntimeError('Host/PG cleanup failed; inspect cleanup_errors')
