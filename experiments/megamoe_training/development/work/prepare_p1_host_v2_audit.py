from pathlib import Path
p=Path('work/validate_p1_training_stage.py').read_text().replace('outputs/p1_training_deployment/manifest.json','outputs/p1_host_v2_deployment/manifest.json')
p=p.replace("assert [c['case'] for c in r['cases']]==cases", """assert [c['case'] for c in r['cases']]==cases
    assert not r.get('cleanup_errors') and not r.get('report_io_error')
    assert [c['case'] for c in r['case_cleanup']]==cases
    for index,c in enumerate(r['case_cleanup']):
        assert c['case_index']==index and c['all_ranks_closed'] is True and c['all_ranks_case_passed'] is True
        assert c['close_error'] is None and c['body_error'] is None and not c.get('host_exchange_error')
        peers=c['peer_status'];assert [v['rank'] for v in peers]==list(range(world))
        assert all(v['case']==c['case'] and v['case_index']==index and not v['close_error'] and not v['body_error'] for v in peers)
    assert [v['rank'] for v in r['peer_before_PG_teardown']]==list(range(world))
    assert all(v['error'] is None for v in r['peer_before_PG_teardown'])
    assert [v['rank'] for v in r['peer_exit_status']]==list(range(world)) and all(not v['errors'] for v in r['peer_exit_status'])""")
Path('work/validate_p1_host_v2_stage.py').write_text(p)
p=Path('work/record_completed_p1_accuracy.py').read_text().replace('completed_p1_accuracy_stages','completed_p1_host_v2_accuracy_stages');Path('work/record_completed_p1_host_v2.py').write_text(p)
