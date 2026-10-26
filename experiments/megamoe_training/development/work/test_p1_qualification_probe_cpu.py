"""Probe-only offline tests; archived Linux paths use an explicit path model."""
import ast,copy,hashlib,importlib.util,json,pathlib,py_compile,runpy,subprocess,sys,types
from unittest.mock import patch
p=pathlib.Path(__file__).with_name('p1_qualification_cpu_probe.py').resolve();m=runpy.run_path(str(p));py_compile.compile(str(p),doraise=True);checks=[]
plans=m['plans']();assert len(plans)==8 and sum(v['shared'] for v in plans)==5 and all(v['cases']=='balanced' for v in plans if v['B']==8192);checks.append('eight legal plans/main balanced')
program='import runpy,sys;sys.argv=['+repr(str(p))+'];runpy.run_path('+repr(str(p))+',run_name="__main__");assert not any(k=="torch" or k.startswith(("torch.","torch_npu","deep_ep")) for k in sys.modules)'
r=subprocess.run([sys.executable,'-I','-c',program],capture_output=True,text=True);assert r.returncode==0,r.stderr;checks.append('actual default no framework import')
r=subprocess.run([sys.executable,'-I',str(p),'--run-cpu'],capture_output=True,text=True);assert r.returncode==2;checks.append('missing explicit paths rejected')
class Body(Exception):pass
class State(Exception):pass
def bad():raise State('state failure')
report={}
try:
 first=None
 try:raise Body('body first')
 except BaseException as e:first=e;raise
 finally:m['finish_state'](report,bad,first)
except Body:pass
else:raise AssertionError('body replaced')
assert 'State' in report['after_state_error'] and 'state failure' in report['after_state_traceback'];checks.append('body and state first error preserved')
try:m['finish_state']({},bad,None)
except State:pass
else:raise AssertionError('state swallowed')
checks.append('state-only primary error')
report={};m['finish_state'](report,lambda:{'false':False},None);assert report['after']=={'false':False};checks.append('successful final state retained')
tree=ast.parse(p.read_text());p3=ast.parse(p.with_name('p3_qualification_cpu_probe.py').read_text())
for name in ('plans','snapshot','finish_state'):
 f=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(f(tree))==ast.dump(f(p3))
checks.append('plans/state/first-error AST match reviewed P3')
forbidden={'tensor','empty','zeros','set_device','init_process_group','PointwiseTrainingMoE','Session','ElasticBuffer'}
assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id in forbidden) or (isinstance(n.func,ast.Attribute) and n.func.attr in forbidden))];checks.append('no engine tensor PG device calls')
ws=p.parents[1];root=ws/'work/training_candidates/p1_qualification_review';s=importlib.util.spec_from_file_location('offline_fd_p1',root/'p1_pointwise/candidate_provenance.py');cp=importlib.util.module_from_spec(s);s.loader.exec_module(cp)
a=json.loads((ws/'outputs/p2_qualification_deployment/rack05os7/collision-cpu.json').read_bytes());raw=a['raw_mapping_result'];coll=next(iter(a['collision_groups'].values()));roots={pathlib.Path(v['path']).parent.name:pathlib.Path(v['path']).parent for v in coll};actual=cp.checked_maps(raw,roots);pw=next(v for v in actual.values() if pathlib.Path(v['path']).name=='libmegamoe_pointwise.so')
i={'content':{'mapped_libraries':cp.canonical_maps(actual),'candidate_files':[dict(role='native.pointwise',sha256=pw['sha256'],bytes=pw['bytes'])]},'actual_maps':list(actual.values()),'diagnostic_paths':{'native.pointwise':pw['path']}}
base=types.SimpleNamespace(collect_runtime_identity=lambda:a['base_identity']);pr=types.SimpleNamespace(validate_identity=lambda *args:None,bind_selected_maps=cp.bind_selected_maps)
class ArchivePath(pathlib.PurePosixPath):
 def resolve(self,strict=False):return self
with patch.dict(m['inspect_identity'].__globals__,Path=ArchivePath):
 e=m['inspect_identity'](i,pr,base);assert len(e['dual_C_roles'])==2 and len(e['selected_native_and_dependency_FD'])>6;checks.append('archived dual_C and exact native/dependency FD inspector model')
 for change in ('missing_core','missing_C_role'):
  z=copy.deepcopy(i)
  if change=='missing_core':z['actual_maps']=[v for v in z['actual_maps'] if v['path']!=a['base_identity']['diagnostic_paths']['native.deepep_core']]
  else:z['content']['mapped_libraries'].pop(next(k for k in z['content']['mapped_libraries'] if k.startswith('package/torch_npu/') and '/_C.' in k))
  try:m['inspect_identity'](z,pr,base)
  except (ValueError,KeyError):pass
  else:raise AssertionError(change)
  checks.append(change+' rejected')
prefix='/home/zzy/deploy/megamoe-20261024/training';new=prefix+'/candidate-p1-qualification-schema2-20261026'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(passed=True,checks=len(checks),items=checks,probe_sha256=sha(p),plans=plans,explicit_proposed_paths=dict(training_source_root=prefix+'/qualified-p0-20261025/experiments/megamoe_training',candidate_source_root=new+'/p1_pointwise',pointwise_library=prefix+'/candidate-p1/build/libmegamoe_pointwise.so',output=new+'/evidence/cpu-import-identity.json'),isolated_gate=dict(source=str(root/'comparison/gate.py'),proposed_copy=new+'/comparison/gate.py',sha256=sha(root/'comparison/gate.py'),replaces_active_gate=False),scope='Default/static/archived FD inspector models and fault checks only. Linux archive path model is explicit, not live FD verification. No51 rerun, deployment/device/current collector.',collector_contract='Each8 plans: prepare, initial collect, base current exact-FD recheck, repeat collect and final transition+stable digest; before/after NPU/PG false/_OWNER none. No runtime/session/tensor creation.',production_manifest_sha256=sha(root/'manifest.json'))
out=p.with_name('p1_qualification_cpu_probe_preparation.json');out.write_bytes((json.dumps(result,indent=2)+'\n').encode('utf8'));print(json.dumps(dict(checks=len(checks),probe=sha(p),preparation=sha(out))))
