"""Local-only new-RUN reuse, no remote/query/execution."""
import pathlib,ast
for source,target in [('start_p2_overlap_0455_once.py','start_p2_overlap_0526_once.py'),('finalize_p2_overlap_0455_release_remote.py','finalize_p2_overlap_0526_release_remote.py'),('audit_p2_overlap_0455_result.py','audit_p2_overlap_0526_result.py')]:
 s=pathlib.Path('work',source).read_text().replace('20261026_0455_p2_overlap_once','20261026_0526_p2_overlap_once')
 ast.parse(s);pathlib.Path('work',target).write_text(s)
s=pathlib.Path('work/start_p2_ep16_serial_0456_conditional_once.py').read_text().replace('20261026_0455_p2_overlap_once','20261026_0526_p2_overlap_once').replace('20261026_0456_p2_ep16_serial_once','20261026_0527_p2_ep16_serial_once')
ast.parse(s);pathlib.Path('work/start_p2_ep16_serial_0527_conditional_once.py').write_text(s)
print('Prepared fresh unique RUN tools locally; nothing executed.')
