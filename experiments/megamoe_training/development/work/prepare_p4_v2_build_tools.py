"""Local-only exact v2 manifest/path adaptation of accepted v1 CPU build tools."""
import pathlib,ast
for source,target in [('deploy_build_p4_cpu_once.py','deploy_build_p4_v2_cpu_once.py'),('collect_p4_cpu_build.py','collect_p4_v2_cpu_build.py')]:
 s=pathlib.Path('work',source).read_text().replace('work/training_candidates/p4_wave_compute','work/training_candidates/p4_wave_compute_v2').replace('candidate-p4-compute-wave-build-20261026','candidate-p4-compute-wave-v2-build-20261026').replace('outputs/p4_compute_wave_build','outputs/p4_compute_wave_v2_build').replace("'manifest.json'","'source_manifest.json'").replace('deec8aea2e29c9a365fbbc6ddbc51fe3a64328db4173f5b9b72f97ead3e3f9d9','81d9290752168ba3e2bf6b625fec9a0bfa14e2e386166e179ce83f5c6e328949').replace("s['cpu_p4_build']","s['cpu_p4_v2_build']")
 ast.parse(s);pathlib.Path('work',target).write_text(s)
print('v2 build tools prepared; no remote action')
