"""Only local copy/hash prep; never invokes compiler."""
import hashlib,json,pathlib,re,shlex,shutil
HERE=pathlib.Path(__file__).resolve().parent;WORK=HERE.parents[1];SOURCE=WORK/'DeepEP';P5=HERE.parent/'p5_dispatch_logical_roles';RECORDS=WORK/'shmem_runtime_header_review_20261026'
def pin(p):
    b=p.read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
    flags=(RECORDS/'build-records/device-flags.make').read_text();old=json.loads((P5/'manifest.json').read_bytes())
    compiler=re.search(r'# compile CXX with (.+)',flags)[1].strip()
    values={k:shlex.split(re.search(r'^'+k+r' = (.*)$',flags,re.M)[1]) for k in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS')}
    snapshots={}
    for flavor in ('original','identity64'):
        target=HERE/'snapshots'/flavor
        if target.exists():raise FileExistsError('snapshot already exists; never overwrite')
        shutil.copytree(SOURCE/'csrc',target/'csrc')
        for rel,v in old['files'].items():
            if pin(target/rel)!=v['before']:raise ValueError('originalP5 before bytes drifted')
            if flavor=='identity64':shutil.copyfile(P5/'candidate'/rel,target/rel)
        snapshots[flavor]={p.relative_to(target).as_posix():pin(p) for p in sorted(target.rglob('*')) if p.is_file()}
    evidence=HERE/'evidence';evidence.mkdir()
    for name in ('build-records/device-flags.make','build-records/dispatch.cpp.o.d','build-records/device-DependInfo.cmake','installed-deepep-build-info.json'):
        dst=evidence/pathlib.Path(name).name;shutil.copyfile(RECORDS/name,dst)
    dep=(evidence/'dispatch.cpp.o.d').read_text().replace('\\\n',' ')
    dependencies=shlex.split(dep.split(':',1)[1]);sourceprefix='/home/zzy/deploy/megamoe-20261024/DeepEP/csrc/'
    external=sorted({str(pathlib.PurePosixPath(n)) for n in dependencies if not n.startswith(sourceprefix)})
    build=json.loads((evidence/'installed-deepep-build-info.json').read_bytes())
    m=dict(schema='P5-paired-Dispatch-object-plan-1',compiler=compiler,recorded_device_compiler_sha256=build['device_compiler_sha256'],defines=values['CXX_DEFINES'],includes=values['CXX_INCLUDES'],flags=values['CXX_FLAGS'],snapshot_files=snapshots,external_dependency_paths=external,
        evidence={p.name:pin(p) for p in evidence.iterdir()},p5_patch=old['patch'],p5_manifest=pin(P5/'manifest.json'),
        missing_before_execute=['A reviewed perRUN bindings.json with compiler/readelf exact path+bytes+SHA and every listed external dependency path->bytes/SHA. SDK/system files are not present locally; cannot assert present/matching.','Explicit device_elf_machine from independently established matching toolchain artifact plus expected device_symbols=[dispatch_kernel,dispatch_profile_kernel]. Do not guess machine ID or label hostonly -c success as devicecoverage.','Parent CPU-only build/deploy authorization, fresh output directory and exact Linux/aarch64 toolchain. This package has no implicit authorization.'],
        binding_schema=dict(compiler='path/bytes/sha256, SHA must equal recorded devicecompiler',readelf='path/bytes/sha256',external_dependencies='exact external_dependency_paths mapping each to bytes/sha256',device_elf_machine='reviewed integer non0/non183 devicearch',device_symbols=['dispatch_kernel','dispatch_profile_kernel']),
        limitations=['Existing .d does NOT name most __CCE_AICORE__ device helperheaders; it is not a full device include closure proof. Full csrc snapshots close all project-relative source paths. External dependency pins only cover recorded closure, not unreported compiler-internal device headers.','No generated build.make or full original driver invocation was retrieved. Reconstruct object command from exact CXX flags/includes/defines plus standard -c/-o and identical -MD/-MF accounting; no CMake/relink.','Both source roots map to same . via replacement of original source file-prefix-map; same SDKpaths/defines/includeorder/O3+O2flags and same outputbasename. Same commands on each variant; sourcecontent is only intended independentvariable.','Embedded ELF scanner handles section-backed little-endianELF64 only. If compiler uses compressed/proprietary offload container, preserve rawobject/log and fail coverageUnknown rather than invent extraction flags.','No code loaded, linked or installed; objectbytes may differ legitimately with parameterized source. No performance/numerical inference from hashes.'],
        device_authorized=False,compile_executed=False)
    (HERE/'manifest.json').write_bytes((json.dumps(m,indent=2)+'\n').encode());print(json.dumps(dict(source_files={f:len(v) for f,v in snapshots.items()},external_dependencies=len(external),manifest=pin(HERE/'manifest.json'))))
if __name__=='__main__':main()
