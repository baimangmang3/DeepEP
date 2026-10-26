"""Local-only reuse of reviewed controls/audit tools. Does not run any command."""
import pathlib,ast
tag='20261026_0425_p2_overlap_once';stage='p2-ep2-bf16-overlap'
starter=pathlib.Path('work/start_p2_overlap_once.py').read_text().replace('20261026_0354_p2_overlap_once',tag)
pathlib.Path('work/start_p2_overlap_0425_once.py').write_text(starter)
release=pathlib.Path('work/finalize_p2_serial_release_remote.py').read_text().replace('20261026_0348_p2_serial_once',tag).replace('p2-ep2-bf16-serial',stage)
pathlib.Path('work/finalize_p2_overlap_release_remote.py').write_text(release)
audit=pathlib.Path('work/audit_p2_serial_result.py').read_text().replace('20261026_0348_p2_serial_once',tag).replace('p2-ep2-bf16-serial',stage).replace('p2_ep2_serial_hostname_preparation','p2_ep2_overlap_hostname_preparation').replace('p2_ep2_serial_hostname_control','p2_ep2_overlap_hostname_control').replace("'serial'","'overlap'").replace('P2 serial','P2 overlap').replace('BF16 serial','BF16 overlap').replace('routed serial accuracy','routed overlap accuracy').replace('p2_ep2_serial_passed_no_successor','p2_ep2_overlap_passed_no_successor')
pathlib.Path('work/audit_p2_overlap_result.py').write_text(audit)
for file in ('start_p2_overlap_0425_once.py','finalize_p2_overlap_release_remote.py','audit_p2_overlap_result.py'):ast.parse(pathlib.Path('work',file).read_text())
print('Local reuse prepared; no SSH or process launched.')
