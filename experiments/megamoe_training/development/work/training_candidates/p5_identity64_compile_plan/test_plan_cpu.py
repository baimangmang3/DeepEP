"""Plan/source consistency only. Never run toolchain, native or network."""
import json,pathlib,subprocess,sys,struct
import compile_objects as C
HERE=pathlib.Path(__file__).resolve().parent;m=json.loads((HERE/'manifest.json').read_bytes());checks=[]
for flavor,inventory in m['snapshot_files'].items():
    for rel,expected in inventory.items():assert C.pin(HERE/'snapshots'/flavor/rel)==expected
assert len(m['snapshot_files']['original'])==len(m['snapshot_files']['identity64'])==88
different=[rel for rel,pin in m['snapshot_files']['original'].items() if pin!=m['snapshot_files']['identity64'][rel]]
assert len(different)==7;checks.append('88file snapshots byte-pinned with exactly7 overlay differences')
args={f:C.command(m,f,pathlib.PurePosixPath('/OUT')/f,'/PKG') for f in ('original','identity64')}
normalized={f:[a.replace('/PKG/snapshots/'+f,'/SOURCE').replace('/OUT/'+f,'/OUTPUT') for a in v] for f,v in args.items()}
assert normalized['original']==normalized['identity64']
assert '-xasc' in args['original'] and '--npu-arch=dav-3510' in args['original'] and '-c' in args['original'] and '-shared' not in args['original']
checks.append('same flags/includeorder/defines/driver under source+output path normalization; objectonly xasc')
assert m['recorded_device_compiler_sha256']=='fb635ef2090ff89f81c7962399b561d617c5a86417260c662d9bc6f84016f3c9'
try:C.validate_bindings(m,{})
except ValueError:pass
else:raise AssertionError('missing SDK/tool/coverage contract accepted')
checks.append('recorded devicecompiler bound; incomplete actualexternalpins/coverage contract rejects before compile')
blob=bytearray(64+128);at=64;blob[at:at+6]=b'\x7fELF\x02\x01';struct.pack_into('<H',blob,at+18,77);struct.pack_into('<Q',blob,at+40,64);struct.pack_into('<HH',blob,at+58,64,1)
e=C.embedded_elfs(bytes(blob));assert len(e)==1 and e[0]['offset']==64 and e[0]['bytes']==128 and e[0]['machine']==77
assert not C.embedded_elfs(b'host object only') and not C.embedded_elfs(bytes(blob[:-1]));checks.append('bounded innerELF extent extraction vs absent/truncated payload; not an architecture claim')
program="import runpy,sys;sys.path.insert(0,"+repr(str(HERE))+");sys.argv=["+repr(str(HERE/'compile_objects.py'))+",'--plan-root','/PARENT_REVIEWED_PACKAGE'];runpy.run_path("+repr(str(HERE/'compile_objects.py'))+",run_name='__main__');assert 'torch' not in sys.modules"
p=subprocess.run([sys.executable,'-I','-c',program],capture_output=True,text=True);assert p.returncode==0,p.stderr
plan=json.loads(p.stdout);assert plan['mode']=='CPU_plan' and plan['commands']['original'][-1].startswith('/PARENT_REVIEWED_PACKAGE/')
(HERE/'default_plan.json').write_bytes((json.dumps(plan,indent=2)+'\n').encode());checks.append('actual defaultplan imports noTorch and emits explicit portableLinux sourcepaths without execution')
result=dict(passed=True,checks=len(checks),items=checks,toolchain_invoked=False,native_compiled=False,NPU=False)
(HERE/'cpu_results.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result,indent=2))
