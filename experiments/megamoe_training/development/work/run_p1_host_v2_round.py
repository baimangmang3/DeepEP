"""Verified8-source P1 package, frozen-P0 dependency and one-shot dual-host gate."""
import hashlib,json
from pathlib import Path
from training_native_round import SshBackend,parallel,T
import training_stage_round

SRC=Path('work/training_candidates/p1_host_v2/p1_pointwise')
MANIFEST=json.loads((SRC/'training_manifest.json').read_text())
FILES=MANIFEST['sources_sha256']
assert len(FILES)==8 and FILES['validate_training_p1.py']=='841f663ee336bb3c342c83de9bc4d19bf5b41cc685329bef141dcf5fbc0ac93c'
assert FILES['candidate_provenance.py']=='f1bfd202a641a064eeb8a298d58119f646b1bc4e5bec0ffa6def6bfce0eab6d5'
for name,sha in FILES.items():assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha,name
CP=T+'/candidate-p1-host-v2/p1_pointwise'

class P1Backend(SshBackend):
    def __init__(self):
        super().__init__()
        self.helper_hashes.update({CP+'/'+name:sha for name,sha in FILES.items()})
        self.helper_hashes[T+'/candidate-p1/build/libmegamoe_pointwise.so']='e189f39a8448df078e082ba90b3147e593962bbcc374aa388d69344cfc125ca1'
        frozen=T+'/qualified-p0-20261025/experiments/megamoe_training/'
        base=json.loads(Path('outputs/training_review_shared_main_source_manifest.json').read_text())
        self.helper_hashes.update({frozen+e['path']:e['sha256'] for e in base['files']})
        self.helper_hashes[frozen+'build/libmegamoe_training.so']='f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32'
        self.helper_hashes[T+'/activate-qualified-p0.sh']='90b1bc782cf83b9d331dd743fb09a865e163a9ed70604291d5b32d2669ad3d3a'
        for name in ('run_training_p1_host_v2_checked.sh','training_occupancy_gate.py'):
            self.helper_hashes[T+'/scripts/'+name]=hashlib.sha256(Path('work',name).read_bytes()).hexdigest()
    def connect(self):
        super().connect()
        def deploy(host):
            self.python(host,'p1_training_cpu_deployment',f'''
import pathlib,json,hashlib
p=pathlib.Path({CP!r});expected={FILES!r}
if p.exists() and any(p.iterdir()) and not ((p/'training_manifest.json').exists() or (p/'cpu-staging.json').exists()):raise RuntimeError('Unowned P1 package path; do not overwrite')
for name,sha in expected.items():
 q=p/name
 if q.exists():
  got=hashlib.sha256(q.read_bytes()).hexdigest()
  if name=='validate_training_p1.py' and got=='4020cc4f0a7ee3145be315168f07bd0a27c6172cfd6e95c27a0f766e2814bd54':
   backup=p.parent/'staged-validator-before-host-v2.py'
   if not backup.exists():backup.write_bytes(q.read_bytes())
  else:assert got==sha,'Modified existing candidate: '+name
(p/'native').mkdir(parents=True,exist_ok=True)
print(json.dumps({{'package_root':str(p)}}))
''')
            with self.clients[host].open_sftp() as s:
                for name in FILES:
                    target=CP+'/'+name;s.put(str(SRC/name),target+'.tmp');s.posix_rename(target+'.tmp',target)
                with s.open(CP+'/training_manifest.json','w') as f:f.write(json.dumps(MANIFEST,indent=2))
                for name in ('run_training_p1_host_v2_checked.sh','training_occupancy_gate.py'):
                    target=T+'/scripts/'+name;s.put(str(Path('work',name)),target+'.tmp');s.posix_rename(target+'.tmp',target)
            return self.python(host,'p1_training_cpu_deployment',f'''import pathlib,json,subprocess,py_compile
p=pathlib.Path({CP!r})
for name in {list(FILES)!r}:
 if name.endswith('.py'):py_compile.compile(str(p/name),doraise=True)
r=subprocess.run(['bash','-n',{T!r}+'/scripts/run_training_p1_host_v2_checked.sh'],capture_output=True,text=True);assert r.returncode==0,r.stderr
print(json.dumps({{'py_compile_passed':True,'bash_n_rc':0}}))''')
        deployed=parallel(deploy)
        out=Path('outputs/p1_host_v2_deployment');out.mkdir(exist_ok=True)
        (out/'manifest.json').write_text(json.dumps(MANIFEST,indent=2));(out/'cpu-deployment.json').write_text(json.dumps(deployed,indent=2))

if __name__=='__main__':
    training_stage_round.SshBackend=P1Backend
    raise SystemExit(training_stage_round.main())
