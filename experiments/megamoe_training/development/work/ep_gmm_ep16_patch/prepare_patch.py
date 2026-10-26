"""Read pinned Git blobs, write only isolated candidate copies and unified diff."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent/'DeepEP'
REV='35fa7d91b321340c153b098bf166202645976a75'
E='experiments/ep_gmm_fused/'
PY=E+'python/deep_ep_experimental/ep_gmm_fused/'
UP=E+'csrc/all_to_allv_grouped_mat_mul_udma/ascend950/'
DOWN=E+'csrc/grouped_mat_mul_all_to_allv_udma/ascend950/'
META=E+'csrc/catccos_overlay/include/catccos/dgemm/moe_route_metadata.hpp'
PATHS=[PY+'runtime.py',PY+'ops.py',PY+'_tiling.py',UP+'all_to_allv_grouped_mat_mul_udma.cpp',
       UP+'udma_runtime.cpp',UP+'all_to_allv_grouped_mat_mul_udma_device.hpp',
       DOWN+'grouped_mat_mul_all_to_allv_udma.cpp',DOWN+'grouped_mat_mul_all_to_allv_udma_device.hpp',
       E+'csrc/tiling_validation.hpp',META,'csrc/runtime/session.cpp']


def original(path):
    return subprocess.check_output(['git','-C',str(REPO),'show',REV+':'+path]).decode()


def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old))
    return text.replace(old,new)


def transform(path,text):
    if path==PY+'runtime.py':return replace(text,'rank < world_size <= 8','rank < world_size <= 16')
    if path==PY+'ops.py':
        text=replace(text,'_MOE_ROUTE_MAX_RANKS = 8','_MOE_ROUTE_MAX_RANKS = 16')
        return replace(text,'route metadata supports at most 8 EP ranks','route metadata supports at most 16 EP ranks')
    if path==PY+'_tiling.py':
        text=replace(text,'world_size > 8','world_size > 16')
        text=replace(text,'world_size must not exceed 8','world_size must not exceed 16')
        text=replace(text,'values = [128, 256, 256, 64, 1280, 2, 32, 1, 0, 1]',
            'values = [128, 256, 256, 64, 1280, 2, 32, 1, 0, 1]\n        if case["world_size"] == 16:\n            values = [128, 256, 256, 64, 1024, 2, 32, 1, 0, 1]')
        return replace(text,'values = [128, 256, 256, 10, 32, 0, 2, 1]',
            'values = [128, 256, 256, 10, 32, 0, 2, 1]\n        if case["world_size"] == 16:\n            # Explicit physical staging; nominal copy rows32 retain the default\n            # UB cap below (N6144 -> effective10). Explicit overrides are not capped.\n            values = [128, 256, 256, 12, 32, 128, 2, 1]')
    if path==UP+'all_to_allv_grouped_mat_mul_udma.cpp':
        text=replace(text,'sizeof(Catccos::DGemm::MoeRouteMetadata) == 12880',
            'sizeof(Catccos::DGemm::MoeRouteMetadata) == 25232')
        text=replace(text,'worldSize > 0 && worldSize <= 8','worldSize > 0 && worldSize <= 16')
        key='constexpr TilingProfile kDefaultTiling{KernelVariant::kM0_128, 128, 256, 256, 10, 64, 64, 1, 16, 1280, 2, 32, 1, 0, 1};'
        text=replace(text,key,key+'\n// EP16-only profile: K6144 uses384MiB of the unchanged400MiB data reserve.\nconstexpr TilingProfile kEp16Tiling{KernelVariant::kM0_128, 128, 256, 256, 8, 64, 64, 1, 16, 1024, 2, 32, 1, 0, 1};')
        return replace(text,'    const bool isFourRank28ExpertCase =',
            '    if (worldSize == 16) {\n        return kEp16Tiling;\n    }\n    const bool isFourRank28ExpertCase =')
    if path==DOWN+'grouped_mat_mul_all_to_allv_udma.cpp':
        key='constexpr TilingProfile kDefaultTiling{KernelVariant::kM0_128, 128, 256, 256, 10, 64, 64, 1, 16, 0, 2, 32, 1};'
        text=replace(text,key,key+'\n// Explicit stageRows avoids deriving staging from32*AIC/EP and interval10.\n// Nominal copy rows32 keep the existing default UB cap: N6144 -> effective10.\nconstexpr TilingProfile kEp16Tiling{KernelVariant::kM0_128, 128, 256, 256, 12, 64, 64, 1, 16, 128, 2, 32, 1};')
        return replace(text,'    const bool isFourRank28ExpertCase =',
            '    if (worldSize == 16) {\n        return kEp16Tiling;\n    }\n    const bool isFourRank28ExpertCase =')
    if path==UP+'udma_runtime.cpp':
        text=replace(text,'worldSize <= 8','worldSize <= 16')
        return replace(text,'UDMA world size must be in [1, 8]','UDMA world size must be in [1, 16]')
    if path==UP+'all_to_allv_grouped_mat_mul_udma_device.hpp':return replace(text,'MoeConstraints<1, 8, 64>','MoeConstraints<1, 16, 64>')
    if path==DOWN+'grouped_mat_mul_all_to_allv_udma_device.hpp':return replace(text,'MoeConstraints<1, 8, 32>','MoeConstraints<1, 16, 32>')
    if path==E+'csrc/tiling_validation.hpp':
        text=replace(text,'worldSize > 8','worldSize > 16')
        return replace(text,'world_size in [1, 8]','world_size in [1, 16]')
    if path==META:
        text=replace(text,'#include <cstdint>','#include <cstdint>\n#include <cstddef>')
        text=replace(text,'MOE_ROUTE_MAX_RANKS = 8','MOE_ROUTE_MAX_RANKS = 16')
        asserts='''// EP16 candidate ABI: Python ops.py and this SO must be deployed together.
static_assert(sizeof(MoeRouteSegment) == 24, "route segment ABI");
static_assert(offsetof(MoeRouteSegment, sourceRowOffset) == 0, "segment source offset");
static_assert(offsetof(MoeRouteSegment, outputRowOffset) == 8, "segment output offset");
static_assert(offsetof(MoeRouteSegment, rows) == 16, "segment rows offset");
static_assert(offsetof(MoeRouteSegment, sourceRank) == 20, "segment rank offset");
static_assert(offsetof(MoeRouteSegment, localExpert) == 22, "segment expert offset");
static_assert(MOE_ROUTE_MAX_SEGMENTS == 1024, "route segment count");
static_assert(offsetof(MoeRouteMetadata, rankSize) == 0, "header rank offset");
static_assert(offsetof(MoeRouteMetadata, localExpertNum) == 4, "header expert offset");
static_assert(offsetof(MoeRouteMetadata, segmentCount) == 8, "header segment offset");
static_assert(offsetof(MoeRouteMetadata, reserved) == 12, "header reserved offset");
static_assert(offsetof(MoeRouteMetadata, peerRows) == 16, "peer rows ABI");
static_assert(offsetof(MoeRouteMetadata, expertRows) == 144, "expert rows ABI");
static_assert(offsetof(MoeRouteMetadata, segments) == 656, "route segments ABI");
static_assert(sizeof(MoeRouteMetadata) == 25232, "route metadata ABI");
'''
        return replace(text,'}  // namespace Catccos::DGemm',asserts+'\n}  // namespace Catccos::DGemm')
    if path=='csrc/runtime/session.cpp':
        return replace(text,'if (size < 1 || size > 8 || rank < 0 || rank >= size ||',
            'if (size < 1 || size > 16 || rank < 0 || rank >= size ||')
    raise AssertionError(path)


def main():
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD']).decode().strip()==REV
    manifest={'pinned_revision':REV,'scope':'Isolated EP16 candidate source only; not built, initialized or qualified',
              'files':[],'UDMA_only':'Session::create_udma guard only; general HT Session constructor unchanged',
              'ABI':'candidate Python and native must match; pinned originalEP8 remains separate and unchanged',
              'defaults':'EP<=8 profiles unchanged. EP16 up1024/2stage/384MiB; down128/2stage/48MiB, interval12, copy nominal32/effective10 at N6144. Explicit32 fails UB.',
              'unverified':['SHMEM16-peer connection/channel capacity and init/cleanup','CANN compile/stack/register/UB effects of expanded scheduler arrays','quiet/ready/recycle liveness with empty/skew/repeated/queued cases','full BF16/FP32-master/shared autograd qualification']}
    patch=[]
    for path in PATHS:
        old=original(path);new=transform(path,old);dest=ROOT/'candidate'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(new.encode())
        patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+path,tofile='b/'+path))
        manifest['files'].append({'path':path,'original_sha256':hashlib.sha256(old.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(new.encode()).hexdigest()})
    (ROOT/'ep16.patch').write_text(''.join(patch),newline='\n')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'revision':REV,'candidate_files':len(PATHS),'patch_lines':len(''.join(patch).splitlines())}))


if __name__=='__main__':main()
