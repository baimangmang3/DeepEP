import pathlib,json,hashlib,os,socket
t=pathlib.Path('/home/zzy/deploy/megamoe-20261024/training');r=t/'review-20261024';source=r/'experiments/megamoe_training'
m=json.loads((r/'source-manifest.json').read_text());prior=json.loads((r/'cpu-validation.json').read_text());identity=prior['identity']
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
for row in m['files']:assert sha(source/row['path'])==row['sha256'],row['path']
for row in identity['content']['files']:assert sha(identity['diagnostic_paths'][row['role']])==row['sha256'],row['role']
assert sha(source/'build/libmegamoe_training.so')==m['native_library_sha256']
result=dict(host=socket.gethostname(),source_files_verified=len(m['files']),cached_identity_files_rehashed=len(identity['content']['files']),cached_content_digest=identity['content_digest'],live_identity_recollected=False,unchanged_cpu_qualification_reused=True,npu_imported=False,passed=True)
out=t/'runs'/os.environ['RUN_TAG']/'preflight';out.mkdir(parents=True,exist_ok=True);(out/(socket.gethostname()+'-quick-identity.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
