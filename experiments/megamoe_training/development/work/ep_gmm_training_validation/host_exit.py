"""Local copied/reviewed host exit helper; no external runtime import."""
def finish_host_report(host, report, dist, save, prior_error=None):
    """Keep host coordination alive until all ranks reach PG teardown."""
    errors = []
    if prior_error is not None:
        report['all_passed'] = False
    local = {'rank': host.rank(), 'error': None if prior_error is None else repr(prior_error)}
    try:
        peers = host.all_gather(local, 'EP-GMM/before-PG-teardown')
        report['peer_before_PG_teardown'] = peers
        if any((peer['error'] for peer in peers)):
            report['all_passed'] = False
            errors.append('A peer reported an original pre-teardown error')
    except Exception as exc:
        errors.append('host pre-teardown: ' + repr(exc))
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:
        errors.append('PG teardown: ' + repr(exc))
    try:
        peers = host.all_gather({'rank': host.rank(), 'errors': errors}, 'EP-GMM/after-PG-teardown')
        if any((peer['errors'] for peer in peers)):
            report['all_passed'] = False
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
        report['all_passed'] = False
    try:
        save()
    except Exception as exc:
        report['all_passed'] = False
        report['report_io_error'] = repr(exc)
        raise
    if errors:
        raise RuntimeError('Host/PG cleanup failed; inspect cleanup_errors')
