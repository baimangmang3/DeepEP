"""Local derived v2 audit with explicit warning/source/toolchain comparison."""
import pathlib,json,hashlib,collections,runpy
src=pathlib.Path('work/finalize_p4_cpu_build_report.py').read_text()
src=src.replace('outputs/p4_compute_wave_build','outputs/p4_compute_wave_v2_build').replace('candidate-p4-compute-wave-build-20261026','candidate-p4-compute-wave-v2-build-20261026').replace('deec8aea2e29c9a365fbbc6ddbc51fe3a64328db4173f5b9b72f97ead3e3f9d9','81d9290752168ba3e2bf6b625fec9a0bfa14e2e386166e179ce83f5c6e328949').replace('fd605673edab709f39e4d8399af3c157f42688ccfba04275d27b1b2a21974a61','80a4c88e937b5b580faae3414922f2371846cfc216d04b6704ad37460c913fb4').replace("'vector_loop_uint16':1","'vector_loop_uint16':0").replace("s['cpu_p4_build']","s['cpu_p4_v2_build']").replace('PID4035691/start23659991','PID4045253/start23701312').replace('All14 compiler warning lines','All13 remaining compiler warning lines').replace('1 vector-loop condition uint16_t compatibility warning','0 vector-loop condition compatibility warnings').replace('cpp204/205','cpp205/206').replace('cpp193','cpp194')
exec(compile(src,'v2_local_audit','exec'))
OUT=pathlib.Path('outputs/p4_compute_wave_v2_build');OLD=pathlib.Path('outputs/p4_compute_wave_build')
load=lambda p:json.loads(pathlib.Path(p).read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
old=load(OLD/'preflight.json');new=load(OUT/'preflight.json')
assert old['compiler']==new['compiler'] and old['catlass_head']==new['catlass_head']
assert len(new['sources'])==6
unchanged=[name for name,pin in new['sources'].items() if pin==old['sources'][name]]
assert len(unchanged)==5 and 'native/wave_compute.cpp' not in unchanged
before=load(OLD/'compiler-diagnostics.json')['warnings'];after=load(OUT/'compiler-diagnostics.json')['warnings']
expected=collections.Counter(v['text'].split('warning:',1)[1].strip() for v in before if '-Wcce-compat' not in v['text'])
actual=collections.Counter(v['text'].split('warning:',1)[1].strip() for v in after)
assert len(before)==14 and len(after)==13 and actual==expected and not any('-Wcce-compat' in v['text'] for v in after)
comparison={'passed':True,'parent_v1_untouched_source_manifest_sha256':'deec8aea2e29c9a365fbbc6ddbc51fe3a64328db4173f5b9b72f97ead3e3f9d9','v2_manifest_sha256':'81d9290752168ba3e2bf6b625fec9a0bfa14e2e386166e179ce83f5c6e328949','unchanged5sources':unchanged,'actual_compiler_path_version_SHA_equal':True,'CATLASS_equal':True,'warning_lines_v1':14,'warning_lines_v2':13,'vector_condition_Wcce_compat_v1':1,'vector_condition_Wcce_compat_v2':0,'remaining_warning_semantics_multiset_exactly_equal':True,'new_warning_count':0,'comparison_normalization':'warning message after warning:; ignore variant directory and shifted source line location, retain text/category and multiplicity','native_loaded':False,'device_qualified':False}
(OUT/'v1-v2-source-toolchain-warning-comparison.json').write_text(json.dumps(comparison,indent=2))
p=OUT/'README.md';p.write_text(p.read_text()+'''\nV2 comparison: actual compiler path/version/SHA and CATLASS match v1 exactly. Other5 source files are byte-identical. The one vector-condition -Wcce-compat warning is absent. All13 remaining warning messages/categories/multiplicities exactly match v1 after ignoring source directory/shifted line location; no new warning. V1 source/build/report files were not overwritten. CPU compilation is not device qualification.\n''')
(OUT/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'},indent=2))
print(json.dumps({'comparison_passed':True,'warning_count':13,'Wcce_compat':0,'new_warnings':0,'audit_sha256':sha(OUT/'CPU-build-audit.json')}))
