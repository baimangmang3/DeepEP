"""CPU-only exact P3 package deployment. Never imports Torch or calls a device API."""
import hashlib,json,pathlib
from training_native_round import SshBackend,T,parallel,HOSTS

LOCAL=pathlib.Path('work/training_candidates');OUT=pathlib.Path('outputs/p3_training_deployment');OUT.mkdir(exist_ok=True)
D=T+'/candidate-p3-training';P0=T+'/qualified-p0-20261025/experiments/megamoe_training';P1=T+'/candidate-p1-host-v2/p1_pointwise'
UP_SO=T+'/candidate-p3-up-swiglu-20261025/build/libmegamoe_up_swiglu.so'
UP_SHA='4623d9abf673210f0e3e07473f9f02c541256d699e47ce0a2a2dd7229c8c856b'
TRAIN_MANIFEST_SHA='0d7cd18912b239554039f22dee88fd46c1534cb054c5f40b424a1c67e69d2f2c'
assert hashlib.sha256((LOCAL/'p3_training/manifest.json').read_bytes()).hexdigest()==TRAIN_MANIFEST_SHA
files={};manifests={}
for name,key in [('p3_training','files_sha256'),('p3_up_swiglu','source_sha256')]:
 p=LOCAL/name/'manifest.json';m=json.loads(p.read_text());manifests[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 for n,h in m[key].items():
  q=LOCAL/name/n;assert hashlib.sha256(q.read_bytes()).hexdigest()==h,n;files[name+'/'+n]={'source':q,'sha256':h}
 files[name+'/manifest.json']={'source':p,'sha256':manifests[name]}
assert files['p3_training/validate_training_p3.py']['sha256']=='4ee40fec5ee643ee1e449e416d9f029c3caa0eb5b37c1aec9172661a2aae6487'
assert files['p3_training/candidate_provenance.py']['sha256']=='2d18392c3f8f114ca8637b5425529ddf417e29e8678c9b784fc12a1766e7eb50'
evidence={
 'up-swiglu-results.json':pathlib.Path('outputs/training_results_20261025_1413/p3-up-swiglu-unit/rack05os7/up-swiglu-results.json'),
 'unit-audit.json':pathlib.Path('outputs/training_results_20261025_1413/p3-up-swiglu-unit/unit-audit.json')}
for n,p in evidence.items():files['evidence/'+n]={'source':p,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
hashes={n:v['sha256'] for n,v in files.items()}
SO_EXPECTED={P0+'/build/libmegamoe_training.so':'f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32',T+'/candidate-p1/build/libmegamoe_pointwise.so':'e189f39a8448df078e082ba90b3147e593962bbcc374aa388d69344cfc125ca1',UP_SO:UP_SHA}

runner=f'''import pathlib,json,hashlib,sys,runpy,io,contextlib,py_compile
D=pathlib.Path({D!r});frozen=pathlib.Path({P0!r});p1=pathlib.Path({P1!r});root=D/'p3_training';up=D/'p3_up_swiglu'
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
source={{}}
for n,h in {hashes!r}.items():
 p=D/n;actual=sha(p);assert actual==h,n;source[n]={{'sha256':actual,'bytes':p.stat().st_size}}
 if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
libraries={{}}
for n,h in {SO_EXPECTED!r}.items():
 p=pathlib.Path(n);actual=sha(p);assert actual==h,n;libraries[n]={{'sha256':actual,'bytes':p.stat().st_size,'evidence_kind':'disk file hash; NOT current process mapping'}}
sys.path.insert(0,str(root))
import candidate_provenance as cp
assert pathlib.Path(cp.__file__).resolve()==(root/'candidate_provenance.py').resolve()
cp.check_source_layout(frozen,root,p1,up)
unit=D/'evidence/up-swiglu-results.json';audit=D/'evidence/unit-audit.json';raw=json.loads(unit.read_text());a=json.loads(audit.read_text())
release=cp.require_standalone_audit(a,unit,{UP_SHA!r})
configs=[dict(label='ep2-small',world=2,shape=dict(B=32,H=128,I=128,E=8,K=4),cases=['balanced','zero_gate','empty','skew']),dict(label='ep16-small',world=16,shape=dict(B=32,H=128,I=128,E=64,K=8),cases=['balanced','zero_gate','empty','skew']),dict(label='ep16-target',world=16,shape=dict(B=8192,H=6144,I=2048,E=64,K=8),cases=['balanced'])]
plans=[]
for v in configs:
 execution=dict(cp.FIXED,world=v['world'],shape=v['shape'],seed=4317,weight_dtype='torch.bfloat16',shared=False,cases=v['cases'],oracle_chunk=1024,max_reference_gib=12.,limits=cp.LIMITS)
 cp.check_execution(execution)
 qualification=cp.require_standalone(raw,up,{UP_SHA!r},'f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32',execution)
 argv=[str(root/'validate_training_p3.py'),'--training-source-root',str(frozen),'--candidate-source-root',str(root),'--pointwise-library',{T!r}+'/candidate-p1/build/libmegamoe_pointwise.so','--p1-source-root',str(p1),'--component-source-root',str(up),'--up-swiglu-library',{UP_SO!r},'--standalone-report',str(unit),'--standalone-audit',str(audit),'--world-size',str(v['world']),'--cases',','.join(v['cases']),'--weight-dtype','bf16','--seed','4317','--oracle-chunk','1024','--max-reference-gib','12','--output',str(D/'plans'/v['label'])]
 for k,n in v['shape'].items():argv.extend(['--'+k,str(n)])
 sys.argv=argv;capture=io.StringIO()
 with contextlib.redirect_stdout(capture):runpy.run_path(argv[0],run_name='__main__')
 plan=json.loads(capture.getvalue());assert plan['action']=='plan only, no Torch/NPU import'
 plans.append(dict(configuration=v,argv=argv,default_plan=plan,source_layout_passed=True,standalone_contract=qualification,standalone_process_audit=release))
forbidden=[n for n in sys.modules if n=='torch' or n.startswith(('torch.','torch_npu','deep_ep','p3_training','p3_up_swiglu','p1_pointwise'))]
assert forbidden==[],forbidden
candidate_maps=[v for v in pathlib.Path('/proc/self/maps').read_text().splitlines() if any(n in v for n in ('libmegamoe','libshmem','libtorch_npu'))]
assert not candidate_maps,candidate_maps
result=dict(passed=True,scope='stdlib CPU preparation only; NO device/full-training qualification',source_manifests={manifests!r},files=source,libraries_disk_only=libraries,evidence_raw_sha256={{'report':sha(unit),'audit':sha(audit)}},configurations=plans,forbidden_runtime_modules=forbidden,candidate_native_mappings=candidate_maps,check_imports_called=False,device_calls=False,torch_imported=False)
(D/'cpu-preflight-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
(OUT/'preflight-cpu.py').write_text(runner)
(OUT/'source-manifest.json').write_text(json.dumps(hashes,indent=2));(OUT/'raw-evidence-sha256.json').write_text(json.dumps({n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in evidence.items()},indent=2))
x=SshBackend()
try:
 x.connect()
 # Existing standalone binary is reused. If the peer lacks it, copy exact verified host27 bytes only.
 presence=parallel(lambda h:x.python(h,'p3_cpu_native_presence',f'''import pathlib,json,hashlib
p=pathlib.Path({UP_SO!r});print(json.dumps({{'exists':p.is_file(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}}))'''))
 assert presence[HOSTS[0]]=={'exists':True,'sha256':UP_SHA}
 for h,r in presence.items():
  if r['exists']:assert r['sha256']==UP_SHA
  else:
   assert h==HOSTS[1]
   local=OUT/'libmegamoe_up_swiglu-copy.so'
   with x.clients[HOSTS[0]].open_sftp() as s:s.get(UP_SO,str(local))
   assert hashlib.sha256(local.read_bytes()).hexdigest()==UP_SHA
   x.python(h,'p3_cpu_native_copy',f'import pathlib,json\npathlib.Path({UP_SO!r}).parent.mkdir(parents=True,exist_ok=True)\nprint(json.dumps({{"directory_only":True}}))')
   with x.clients[h].open_sftp() as s:s.put(str(local),UP_SO)
 def deploy(h):
  x.python(h,'p3_cpu_package_prepare',f'''import pathlib,json,hashlib
p=pathlib.Path({D!r});expected={hashes!r}
if p.exists():
 for n,h in expected.items():
  q=p/n
  if q.exists():assert hashlib.sha256(q.read_bytes()).hexdigest()==h,'Existing changed file: '+str(q)
for n in expected:(p/n).parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({{'new_namespace':str(p)}}))''')
  with x.clients[h].open_sftp() as s:
   for n,v in files.items():s.put(str(v['source']),D+'/'+n)
   s.put(str(OUT/'preflight-cpu.py'),D+'/preflight-cpu.py')
  return x.python(h,'p3_cpu_preflight',f'import runpy\nrunpy.run_path({D!r}+"/preflight-cpu.py",run_name="__main__")')
 results=parallel(deploy);(OUT/'cpu-preflight-results.json').write_text(json.dumps(results,indent=2));(OUT/'deployment.json').write_text(json.dumps({'remote_root':D,'manifests':manifests,'preflight_script_sha256':hashlib.sha256((OUT/'preflight-cpu.py').read_bytes()).hexdigest(),'SO_presence_before':presence,'device_stage_started':False,'NPU_qualification':False},indent=2));print(json.dumps({h:{'passed':r['passed'],'configurations':len(r['configurations']),'torch_imported':r['torch_imported'],'forbidden_modules':r['forbidden_runtime_modules'],'raw_evidence_sha256':r['evidence_raw_sha256']} for h,r in results.items()}))
finally:x.close()
