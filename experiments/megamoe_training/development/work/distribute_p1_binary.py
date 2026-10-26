import hashlib,json
from pathlib import Path
from training_native_round import SshBackend,parallel,T,HOSTS

D=T+'/candidate-p1';out=Path('outputs/p1_pointwise_build');library=D+'/build/libmegamoe_pointwise.so'
b=SshBackend()
try:
    b.connect()
    with b.clients[HOSTS[0]].open_sftp() as s:
        with s.open(D+'/logs/build-attempt3.rc','rb') as f:assert f.read().strip()==b'0'
        s.get(library,str(out/'libmegamoe_pointwise.so'))
        s.get(D+'/logs/build-attempt3.log',str(out/'build-attempt3.log'))
    data=(out/'libmegamoe_pointwise.so').read_bytes();sha=hashlib.sha256(data).hexdigest()
    with b.clients[HOSTS[1]].open_sftp() as s:
        try:s.mkdir(D+'/build')
        except OSError:pass
        s.put(str(out/'libmegamoe_pointwise.so'),library+'.tmp');s.posix_rename(library+'.tmp',library)
    def verify(host):
        return b.python(host,'p1_binary_distribution',f'''
import pathlib,json,hashlib
p=pathlib.Path({D!r});m=json.loads((p/'source-manifest.json').read_text())
for name,expected in m['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
lib=p/'build/libmegamoe_pointwise.so';assert hashlib.sha256(lib.read_bytes()).hexdigest()=={sha!r}
m['compile_fix']='Pure AIV entry core_ratio(0,1), plus missing register API aggregate header; no math/events/tile/block changes'
(p/'source-manifest.json').write_text(json.dumps(m,indent=2))
manifest={{'source_files':m['files'],'library':str(lib),'sha256':{sha!r},'bytes':lib.stat().st_size,'build_host':'177.22.27.1','build_rc':0,'candidate_scope':'standalone pointwise only; no training adapter'}}
(p/'binary-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
''')
    result=parallel(verify);(out/'binary-distribution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:b.close()
