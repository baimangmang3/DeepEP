"""Local derived-report correction only; preserves all raw inputs."""
import pathlib,json,hashlib
out=pathlib.Path('outputs/p4_compute_wave_build');p=out/'CPU-build-audit.json'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
raw_before={str(f.relative_to(out)):sha(f) for f in out.iterdir() if f.is_file() and f.name not in ('CPU-build-audit.json','README.md','artifact-sha256.json','derived-parser-correction.json')}
text=(out/'readelf.txt').read_text();names=set();matched=0
for line in text.splitlines():
 fields=line.split()
 if len(fields)<8 or not fields[0].rstrip(':').isdigit() or fields[4]!='GLOBAL' or fields[6]!='UND':continue
 name=fields[7];assert not name.startswith('('),line
 names.add(name);matched+=1
assert names and any('@GLIBC_' in name for name in names)
audit=json.loads(p.read_text());old=audit['ELF_strong_undefined_import_names'];audit['ELF_strong_undefined_import_names']=sorted(names)
audit['ELF_import_name_parser']='readelf -W symbol-table fixed columns: Num Value Size Type Bind Vis Ndx Name. Name=field7, retain@VERSION; trailing(parenthesized version index) is separate. Exact names deduplicated across saved symbol tables.'
p.write_text(json.dumps(audit,indent=2))
readme=out/'README.md';readme.write_text(readme.read_text()+'\nLocal derived-report correction: ELF UND names are parsed from the symbol-table Name column, preserving @VERSION and excluding trailing readelf version-index annotations. Raw readelf/build logs and binaries unchanged; no remote command, compilation or occupancy operation.\n')
assert all(sha(out/name)==h for name,h in raw_before.items())
report={'scope':'Local derived report only','old_names':old,'new_names':sorted(names),'matched_symbol_table_rows':matched,'raw_inputs_sha256_unchanged':raw_before,'CPU_build_audit_sha256':sha(p),'remote_commands':False,'recompiled':False,'occupancy_queried':False}
(out/'derived-parser-correction.json').write_text(json.dumps(report,indent=2))
(out/'artifact-sha256.json').write_text(json.dumps({str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='artifact-sha256.json'},indent=2))
print(json.dumps({'fixed_import_names':len(names),'versioned_examples':sorted(n for n in names if '@' in n)[:4],'audit_sha256':sha(p),'raw_unchanged':True}))
