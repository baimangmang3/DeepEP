"""Stdlib injected two-host submission bookkeeping; no process/device actions."""
def launch_hosts_once(hosts,launch,recover,prelaunch,report):
    started=report.setdefault('launches',{})
    for host in hosts:
        # Check already-started peer before issuing the next launch.
        try:
            prelaunch(started)
            report.setdefault('launch_attempts',[]).append(host)
            started[host]=launch(host)
        except Exception as exc:
            report.setdefault('first_error',repr(exc))
            # A response can be lost after remote Popen. Recover only exact
            # marker identity; absence/unknown is never authority to signal.
            if host not in started and host in report.get('launch_attempts',[]):
                try:
                    identity=recover(host)
                    if identity is not None:started[host]=identity
                    else:report.setdefault('unconfirmed_launches',{})[host]='Attempt failed, no exact identity recovered; do not signal guessed processes'
                except Exception as recovery:
                    report.setdefault('launch_recovery_errors',{})[host]=repr(recovery)
            raise
    return started


def cleanup_started(stop,report):
    for host in tuple(report.get('launches',{})):
        if host in report.get('cleanup',{}) or host in report.get('cleanup_errors',{}):continue
        try:report.setdefault('cleanup',{})[host]=stop(host)
        except Exception as exc:report.setdefault('cleanup_errors',{})[host]=repr(exc)


def cleanup_is_uncertain(report):
    return bool(report.get('unconfirmed_launches') or report.get('launch_recovery_errors') or
        report.get('cleanup_errors') or any(
        not isinstance(v,dict) or not isinstance(v.get('remaining'),list) or v['remaining'] or
        not isinstance(v.get('cleanup_errors'),list) or v['cleanup_errors'] or v.get('summary_save_error') or
        type(v.get('node_rc')) is not int or type(v.get('supervisor_rc')) is not int
        for v in report.get('cleanup',{}).values()))
