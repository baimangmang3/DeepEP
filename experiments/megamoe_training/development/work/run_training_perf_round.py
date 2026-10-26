"""Pinned external harness, isolated qualified-P0 launcher and one-shot dual-host gate."""
import hashlib
import json
from pathlib import Path
from training_native_round import SshBackend, parallel, T
import training_stage_round

APPROVED={'probe.py':'8cd2d92b7f1b085fe80218cec34d38df149aa353cbc912f236463ae95ad1cd2c',
          'gate.py':'4755516f347d78afdc5719e0c0000afa69d43e2222c8689616ca5f7a3c71fe8f'}
for name,sha in APPROVED.items():
    assert hashlib.sha256(Path('work/training_perf_probe',name).read_bytes()).hexdigest()==sha,name


class PerfBackend(SshBackend):
    def __init__(self):
        super().__init__()
        for name,sha in APPROVED.items():self.helper_hashes[T+'/scripts/training_perf_probe/'+name]=sha
        for name in ('run_training_perf_checked.sh','training_occupancy_gate.py'):
            self.helper_hashes[T+'/scripts/'+name]=hashlib.sha256(Path('work',name).read_bytes()).hexdigest()
        manifest=json.loads(Path('outputs/training_review_shared_main_source_manifest.json').read_text())
        prefix=T+'/qualified-p0-20261025/experiments/megamoe_training/'
        self.helper_hashes.update({prefix+e['path']:e['sha256'] for e in manifest['files']})
        self.helper_hashes[prefix+'build/libmegamoe_training.so']='f26ddcb4ea287d10f9b45bdc7bc29b44d03c79304c530960abad46d750fecf32'
        self.helper_hashes[T+'/activate-qualified-p0.sh']='90b1bc782cf83b9d331dd743fb09a865e163a9ed70604291d5b32d2669ad3d3a'

    def connect(self):
        super().connect()
        def deploy(host):
            self.python(host,'training_perf_deployment',f'import pathlib,json\np=pathlib.Path({T!r})/"scripts/training_perf_probe";p.mkdir(parents=True,exist_ok=True)\nprint(json.dumps({{"directory":str(p)}}))')
            with self.clients[host].open_sftp() as s:
                sources={T+'/scripts/training_perf_probe/'+name:Path('work/training_perf_probe',name) for name in APPROVED}
                sources.update({T+'/scripts/'+name:Path('work',name) for name in ('run_training_perf_checked.sh','training_occupancy_gate.py')})
                for target,source in sources.items():s.put(str(source),target+'.tmp');s.posix_rename(target+'.tmp',target)
            return self.python(host,'training_perf_deployment',f'import subprocess,json\nr=subprocess.run(["bash","-n",{T!r}+"/scripts/run_training_perf_checked.sh"],capture_output=True,text=True);assert r.returncode==0,r.stderr\nprint(json.dumps({{"bash_n_rc":r.returncode}}))')
        parallel(deploy)


if __name__=='__main__':
    training_stage_round.SshBackend=PerfBackend
    raise SystemExit(training_stage_round.main())
