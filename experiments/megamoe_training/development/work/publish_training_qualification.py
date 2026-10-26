"""Publish an audited complete16-rank qualification; stdlib/SSH file operations only."""
import hashlib
import json
from pathlib import Path
import sys
from training_native_round import SshBackend, parallel, T

tag,stage=sys.argv[1:3]
audit=json.loads(Path('outputs','training_results_'+tag,'stage-accuracy-audit.json').read_text())
assert audit['complete'] is True and audit['weight_dtype']=='torch.bfloat16'
assert json.loads(Path('outputs/training_device_progress.json').read_text())['device_tasks_currently_running'] is False
name=tag+'_'+stage
local=Path('outputs/training_qualifications')/name
local.mkdir(parents=True,exist_ok=True)
manifest=dict(shape=audit['shape'],backend=audit['backend'],shared=audit['shared'],weight_dtype=audit['weight_dtype'],identity=audit['identities'][0],source_stage=tag+'/'+stage,files={})
for rank in range(16):
    host='rack05os7' if rank<8 else 'rack05os8'
    data=Path('outputs','training_results_'+tag,stage,host,f'rank{rank}.json').read_bytes()
    (local/f'rank{rank}.json').write_bytes(data)
    manifest['files'][f'rank{rank}.json']=hashlib.sha256(data).hexdigest()
(local/'qualification-manifest.json').write_text(json.dumps(manifest,indent=2))
backend=SshBackend()
try:
    backend.connect()
    remote=T+'/qualification/'+name
    def publish(host):
        backend.python(host,tag,f'import pathlib,json\np=pathlib.Path({remote!r});p.mkdir(parents=True,exist_ok=True)\nprint(json.dumps({{"path":str(p)}}))')
        with backend.clients[host].open_sftp() as s:
            for source in local.iterdir():
                target=remote+'/'+source.name
                s.put(str(source),target+'.tmp');s.posix_rename(target+'.tmp',target)
        checked=backend.python(host,tag,f'''
import pathlib,hashlib,json
p=pathlib.Path({remote!r});m=json.loads((p/'qualification-manifest.json').read_text())
assert len(m['files'])==16
for name,sha in m['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
print(json.dumps({{'path':str(p),'all16_file_hashes_passed':True,'identity':m['identity']}}))
''')
        return checked
    results=parallel(publish)
    (local/'remote-publication.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results))
finally:
    backend.close()
