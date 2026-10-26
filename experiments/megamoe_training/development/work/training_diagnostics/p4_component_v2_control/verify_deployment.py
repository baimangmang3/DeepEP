"""Read raw disk pins only; no Torch or native loads."""
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]
m=json.loads(p.read_bytes())
for name,item in m['files'].items():
    file=p.parent/name;assert file.stat().st_size==item['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256'],name
if '--worker-source' in sys.argv:
    for name,item in m['worker_source_files'].items():
        file=pathlib.Path(name);assert file.stat().st_size==item['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256'],name
print(json.dumps(dict(passed=True,manifest_sha256=sys.argv[2],worker_source_verified='--worker-source' in sys.argv)))
