"""Stdlib-only pinned original source/native deployment verification."""
import pathlib,sys,json,hashlib
p=pathlib.Path(sys.argv[1]);assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2],'Manifest SHA drift'
m=json.loads(p.read_text());R=p.parent
def check(p,v):
 with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 assert sha==v['sha256'] and p.stat().st_size==v['bytes'],str(p)
for n,v in m['files'].items():check(R/n,v)
for n,v in m['source_files'].items():check(pathlib.Path(n),v)
for n,v in m['external_native'].items():check(pathlib.Path(n),v)
print(json.dumps({'passed':True,'manifest_sha256':sys.argv[2],'original_source_and_native_pins':True,'Torch_imported':False}))
