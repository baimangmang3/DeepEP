"""Offline wheel/build-record SHA binding; no import/install/native loading."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

REV='35fa7d91b321340c153b098bf166202645976a75'
PATCH='4dc5a3073a0dec8f45bd2f4c49d2a00ed6fea8d5f6c5fa49eb2535ba5747b839'
EXPECTED_EP16_SOURCES={'csrc/runtime/session.cpp': 'e81daf07fc7a0a6f0379619434ca2eb65ee97a0cc696530e1dc250dd45a4cf08', 'experiments/ep_gmm_fused/csrc/tiling_validation.hpp': '969a68c7d4cc1d6a92b0e051ed0f41fafe0c2db968aeba535bd352f4e1cd739f', 'experiments/ep_gmm_fused/csrc/all_to_allv_grouped_mat_mul_udma/ascend950/all_to_allv_grouped_mat_mul_udma.cpp': 'cf17a3e8721dde2b06679094c507b5f552f74275c6433d9d1adefcbbdbb83a85', 'experiments/ep_gmm_fused/csrc/all_to_allv_grouped_mat_mul_udma/ascend950/all_to_allv_grouped_mat_mul_udma_device.hpp': 'cdd3766409dc6a3a7e08469b5323df9e4dface66e74ab872d56bdb30a2ed3083', 'experiments/ep_gmm_fused/csrc/all_to_allv_grouped_mat_mul_udma/ascend950/udma_runtime.cpp': 'f940649faab79a278463fd2b7d57dc3cf373f9121a5f4ccb67ae531926cf70b6', 'experiments/ep_gmm_fused/csrc/catccos_overlay/include/catccos/dgemm/moe_route_metadata.hpp': 'c518858f3a0e528de89d7dc9cb24232736441ae2b2c99790b5ce425816cc6058', 'experiments/ep_gmm_fused/csrc/grouped_mat_mul_all_to_allv_udma/ascend950/grouped_mat_mul_all_to_allv_udma.cpp': '4750845f8bceb4076540e5e46e6952866e5089d7a705928b42345a218a0e632c', 'experiments/ep_gmm_fused/csrc/grouped_mat_mul_all_to_allv_udma/ascend950/grouped_mat_mul_all_to_allv_udma_device.hpp': '4d80898de8bdefef508292833592918f6926de1cbd768dc8d93ffc1364c2dc63', 'experiments/ep_gmm_fused/python/deep_ep_experimental/ep_gmm_fused/ops.py': '57344593f053e7247f971bc0eb68b5a7c71920029d00c9a57fca29a53da8a5c1', 'experiments/ep_gmm_fused/python/deep_ep_experimental/ep_gmm_fused/runtime.py': 'f0d13f1de5814c511bd4474aa806e52068685a138ecd712e96e19122f13259f2', 'experiments/ep_gmm_fused/python/deep_ep_experimental/ep_gmm_fused/_tiling.py': 'cd5eb5e02b991822859cb1c2a297de1f2e396acb6c30e89bc9a1d45856cdab75'}


def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make_contract(variant,state,wheel):
    if state.get('build_rc')!=0:raise ValueError('Successful audited build required')
    if variant=='ep16_adapted':
        if state.get('base_revision')!=REV or state.get('patch_sha256')!=PATCH or state.get('source_sha256')!=EXPECTED_EP16_SOURCES:
            raise ValueError('EP16 reviewed revision/patch/11-source record required')
        dependencies={k:{v:x for v,x in entry.items() if v!='path'} for k,entry in state.get('dependencies',{}).items()}
        sources=state['source_sha256']
    elif variant=='ep8_original':
        repos=state.get('repository_status',{})
        if repos.get('deepep')!={'head':REV,'status':''}:raise ValueError('Clean pinned EP8 revision required')
        dependencies=repos;sources={}
    else:raise ValueError('Unknown variant')
    wheel=Path(wheel);h=hashlib.sha256()
    with wheel.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    wheel_sha=h.hexdigest()
    if not any(v.get('sha256')==wheel_sha and v.get('bytes')==wheel.stat().st_size and v.get('relative','').endswith('.whl') for v in state.get('files',[])):
        raise ValueError('Wheel content does not match audited build record')
    native={};metadata={};python_files={}
    with zipfile.ZipFile(wheel) as archive:
        names=archive.namelist()
        for name in names:
            if name.endswith('.py') and (name.startswith('deep_ep/') or name.startswith('deep_ep_experimental/ep_gmm_fused/') or name=='deep_ep_experimental/__init__.py'):
                payload=archive.read(name);python_files[name]={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
        for role,predicate in {
            'native.ep_gmm':lambda n:n.startswith('deep_ep_experimental/ep_gmm_fused/_C') and n.endswith('.so'),
            'native.deep_core':lambda n:n.startswith('deep_ep/_C') and n.endswith('.so'),
            'native.kernels':lambda n:Path(n).name=='libdeepep_kernels.so',
        }.items():
            matches=[n for n in names if predicate(n)]
            if len(matches)!=1:raise ValueError('Missing/ambiguous wheel native role '+role)
            payload=archive.read(matches[0]);native[role]={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'wheel_member':matches[0]}
        for name in ('deep_ep/_build_info.json','deep_ep_experimental/ep_gmm_fused/_build_info.json','deep_ep_experimental/ep_gmm_fused/upstream.json'):
            payload=archive.read(name);metadata[name]={'sha256':hashlib.sha256(payload).hexdigest(),'metadata':json.loads(payload)}
        shmem=metadata['deep_ep/_build_info.json']['metadata']['shmem']['sha256']['lib/libshmem.so']
        native['native.shmem']={'sha256':shmem}
    content={'variant':variant,'base_revision':REV,'patch_sha256':PATCH if variant=='ep16_adapted' else None,
             'wheel_sha256':wheel_sha,'native':native,'source_sha256':sources,'dependencies':dependencies,'embedded_metadata':metadata,'wheel_python':python_files}
    return {'schema':'ep-gmm-build-contract-1','content':content,'content_digest':digest(content)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--variant',choices=['ep8_original','ep16_adapted'],required=True)
    ap.add_argument('--build-state',type=Path,required=True);ap.add_argument('--wheel',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();record=make_contract(a.variant,json.loads(a.build_state.read_text()),a.wheel)
    a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'content_digest':record['content_digest'],'variant':a.variant}))


if __name__=='__main__':main()
