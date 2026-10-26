"""Isolated CPU -xasc object compilation, DEFAULT PLAN. No imports of Torch.

No SSH/deployment/linking/library loading/device operations. Execution requires
separate reviewed tool/dependency/embedded-ELF contract; not supplied by default.
"""
import argparse,hashlib,json,pathlib,platform,shlex,struct,subprocess,time,traceback
from cpu_recorder import run_recorded
HERE=pathlib.Path(__file__).resolve().parent
OLD_ROOT='/home/zzy/deploy/megamoe-20261024/DeepEP'
def pin(p):
    b=pathlib.Path(p).read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,value):pathlib.Path(p).write_bytes((json.dumps(value,indent=2)+'\n').encode())
def command(m,flavor,output,package_root=None):
    source_root=pathlib.PurePosixPath(package_root)/'snapshots'/flavor if package_root else (HERE/'snapshots'/flavor).resolve()
    flags=[v.replace('-ffile-prefix-map='+OLD_ROOT+'=.', '-ffile-prefix-map='+source_root.as_posix()+'=.') for v in m['flags']]
    return [m['compiler'],*m['defines'],*m['includes'],*flags,'-MD','-MT','dispatch.o','-MF',(output/'dispatch.o.d').as_posix(),'-o',(output/'dispatch.o').as_posix(),'-c',(source_root/'csrc/kernels/dispatch/dispatch.cpp').as_posix()]
def validate_bindings(m,b):
    required={'compiler','readelf','external_dependencies','device_elf_machine','device_symbols'}
    if set(b)!=required or b['compiler'].get('path')!=m['compiler']:raise ValueError('incomplete exact tool/coverage contract')
    if b['compiler']['sha256']!=m['recorded_device_compiler_sha256']:raise ValueError('compiler differs from recorded build')
    if type(b['device_elf_machine']) is not int or b['device_elf_machine']!=m['device_elf_machine']:raise ValueError('recorded device ELF machine required')
    if b['device_symbols']!=['dispatch_kernel','dispatch_profile_kernel']:raise ValueError('both Profile branch kernels required')
    if set(b['external_dependencies'])!=set(m['external_dependency_paths']):raise ValueError('all historical external dependency pins required')
    for item in (b['compiler'],b['readelf']):
        actual=pin(item['path'])
        if actual!={k:item[k] for k in ('bytes','sha256')}:raise ValueError('tool bytes changed')
    for path,expected in b['external_dependencies'].items():
        if pin(path)!=expected:raise ValueError('SDK/system dependency changed: '+path)

def embedded_elfs(blob):
    """Bounded section-backed ELF extents; compressed/offload containers unsupported."""
    out=[];at=0
    while True:
        at=blob.find(b'\x7fELF',at+1)
        if at<0:break
        try:
            if blob[at+4:at+6]!=b'\x02\x01':continue # explicit ELF64 little endian
            machine=struct.unpack_from('<H',blob,at+18)[0]
            shoff=struct.unpack_from('<Q',blob,at+40)[0];entsize,count=struct.unpack_from('<HH',blob,at+58)
            if entsize!=64 or not 0<count<65536 or shoff+count*64>len(blob)-at:continue
            end=shoff+count*64
            for i in range(count):
                h=at+shoff+i*64;kind=struct.unpack_from('<I',blob,h+4)[0];offset,size=struct.unpack_from('<QQ',blob,h+24)
                if kind!=8:end=max(end,offset+size) # NOBITS is not in file
            if end>len(blob)-at:continue
            payload=blob[at:at+end];out.append(dict(offset=at,bytes=end,machine=machine,sha256=hashlib.sha256(payload).hexdigest()))
        except (IndexError,struct.error):continue
    return out

def defined_functions(text):
    return {line.split()[-1] for line in text.splitlines() if len(line.split())>=8 and line.split()[3]=='FUNC' and line.split()[-2].isdigit() and int(line.split()[-2])>0}

def device_symbol_sources(elfs,machine,required):
    # Each -xasc kernel may have a separate ELF. Host stubs, undefinedFUNC,
    # metadataOBJECTs and substrings never contribute to this union.
    found={name:[] for name in required}
    for index,e in enumerate(elfs):
        if e['machine']==machine and e.get('readelf_rc')==0:
            for name in set(e.get('defined_required_symbols',[])).intersection(required):found[name].append(index)
    if any(not sources for sources in found.values()):raise ValueError('device ELF defined-FUNC union lacks required kernel')
    return found

def verify_dependencies(paths,m,flavor,bindings):
    root=(HERE/'snapshots'/flavor).resolve();used={}
    known={str(pathlib.Path(p).resolve()):expected for p,expected in bindings['external_dependencies'].items()}
    for path in paths:
        actual=pathlib.Path(path).resolve()
        if actual.is_relative_to(root):
            rel=actual.relative_to(root).as_posix()
            if rel not in m['snapshot_files'][flavor]:raise ValueError('unbound source-root dependency: '+rel)
            expected=m['snapshot_files'][flavor][rel]
        else:
            if str(actual) not in known:raise ValueError('unbound external dependency: '+str(actual))
            expected=known[str(actual)]
        if pin(actual)!=expected:raise ValueError('actual used dependency changed: '+str(actual))
        used[str(actual)]=expected
    return used

def main():
    p=argparse.ArgumentParser();p.add_argument('--compile-objects',action='store_true');p.add_argument('--bindings',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);p.add_argument('--plan-root');a=p.parse_args()
    m=json.loads((HERE/'manifest.json').read_bytes())
    report=dict(mode='isolated_CPU_object_compile' if a.compile_objects else 'CPU_plan',manifest=pin(HERE/'manifest.json'),link=False,NPU=False,torch_import=False,
        commands={f:command(m,f,pathlib.PurePosixPath('/NEW_EMPTY_OUTPUT')/f,a.plan_root) for f in ('original','identity64')},
        missing_contract=m['missing_before_execute'])
    if not a.compile_objects:print(json.dumps(report,indent=2));return
    if a.plan_root:p.error('--plan-root is illustration only; execute always uses actual package location')
    if a.bindings is None or a.output is None:p.error('explicit reviewed --bindings and fresh --output required')
    if not pathlib.Path('/proc/self').exists() or platform.machine() not in ('aarch64','arm64'):raise RuntimeError('prepared only for original Linux/aarch64 CPU toolchain')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False);report['runs']=[]
    try:
        for name,expected in m['scripts'].items():
            if pin(HERE/name)!=expected:raise ValueError('compile recorder/plan source changed')
        b=json.loads(a.bindings.read_bytes());report['bindings']=pin(a.bindings);validate_bindings(m,b)
        for flavor,inventory in m['snapshot_files'].items():
            for rel,expected in inventory.items():
                if pin(HERE/'snapshots'/flavor/rel)!=expected:raise ValueError('source snapshot changed')
        for flavor in ('original','identity64'):
            dest=output/flavor;dest.mkdir();argv=command(m,flavor,dest);item=dict(flavor=flavor,argv=argv,start_ns=time.time_ns());report['runs'].append(item)
            process=run_recorded(argv,dest,dest/'stdout.log',dest/'stderr.log',dest/'process-status.json',limit=300)
            item.update(process=process,returncode=process['returncode'],end_ns=time.time_ns(),stdout=pin(dest/'stdout.log'),stderr=pin(dest/'stderr.log'))
            save(output/'report.json',report)
            if not process.get('passed'):raise RuntimeError(flavor+' compile failed: '+str(process.get('first_error',process['returncode']))+'; successor not executed')
            deps=shlex.split((dest/'dispatch.o.d').read_text().replace('\\\n',' ').split(':',1)[1])
            item['dependency_file']=pin(dest/'dispatch.o.d');item['actual_used_dependencies']=verify_dependencies(deps,m,flavor,b)
            item['object']=pin(dest/'dispatch.o');raw=(dest/'dispatch.o').read_bytes()
            item['embedded_ELFs']=embedded_elfs(raw)
            for i,e in enumerate(item['embedded_ELFs']):
                payload=dest/('embedded-'+str(i)+'.elf');payload.write_bytes(raw[e['offset']:e['offset']+e['bytes']])
                prefix=dest/('embedded-'+str(i)+'-readelf')
                r=run_recorded([b['readelf']['path'],'--wide','-h','-S','-s',str(payload)],dest,prefix.with_suffix('.stdout.log'),prefix.with_suffix('.stderr.log'),prefix.with_suffix('.status.json'),limit=30)
                e['readelf_process']=r;e['readelf_rc']=r['returncode'];e['readelf_stdout']=pin(prefix.with_suffix('.stdout.log'));e['readelf_stderr']=pin(prefix.with_suffix('.stderr.log'))
                if not r.get('passed'):raise RuntimeError('readelf failed: '+str(r.get('first_error',r['returncode'])))
                text=prefix.with_suffix('.stdout.log').read_text(errors='replace')
                defined=defined_functions(text)
                e['defined_required_symbols']=sorted(defined.intersection(b['device_symbols']))
            item['device_symbol_sources']=device_symbol_sources(item['embedded_ELFs'],b['device_elf_machine'],b['device_symbols'])
            for path,expected in item['actual_used_dependencies'].items():
                if pin(path)!=expected:raise ValueError('used source/dependency mutated after compile: '+path)
            validate_bindings(m,b) # same actual files across the pair, not only version text
        report['passed']=True
    except BaseException as exc:
        report.update(passed=False,first_error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        report['scope']='Object compile/device-payload evidence only, not loaded kernel or device accuracy';save(output/'report.json',report)
if __name__=='__main__':main()
