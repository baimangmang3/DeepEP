"""Offline identity64 parameter threading. Never modifies original DeepEP."""
import difflib,hashlib,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'DeepEP'
changes={}
def add(name,old,new):changes.setdefault(name,[]).append((old,new))
add('dispatch.cpp','    const uint32_t core = GetBlockIdx(), cores = 64U;','    const uint32_t physical_role = GetBlockIdx();\n    const uint32_t logical_role = physical_role;  // identity64 only; launch/participants unchanged\n    const uint32_t core = logical_role, cores = 64U;')
add('dispatch.cpp','PublishPeerAddresses(workspace, t, copy, slots);','PublishPeerAddresses(workspace, t, logical_role, copy, slots);')
add('dispatch.cpp','    if (core >= 32U) {\n        if (weights)', '    if (core >= 32U) {\n        const uint32_t aux_owner = logical_role - 32U;\n        if (weights)')
add('dispatch.cpp','DispatchAuxPipeline::Run<Profile>(workspace, t, plan, slots, copy, wqe_buffer);','DispatchAuxPipeline::Run<Profile>(workspace, t, plan, logical_role, aux_owner, slots, copy, wqe_buffer);')
add('dispatch.cpp','PeerScales<Profile>(workspace, scales, destinations, t, copy, slots, wqe_buffer);','PeerScales<Profile>(workspace, scales, destinations, t, logical_role, aux_owner, copy, slots, wqe_buffer);')
add('dispatch.cpp','WaitDispatchIncoming(workspace, t, wqe_buffer);','WaitDispatchIncoming(workspace, t, logical_role, wqe_buffer);')
add('peer_address.hpp','__aicore__ inline void PublishPeerAddresses(GM_ADDR workspace, const DispatchTiling& t,','__aicore__ inline void PublishPeerAddresses(GM_ADDR workspace, const DispatchTiling& t, uint32_t logical_role,')
add('peer_address.hpp','CacheClosDirectories(t, slots);','CacheClosDirectories(t, logical_role, slots);')
add('peer_address.hpp','if (GetBlockIdx() == 0U)','if (logical_role == 0U)')
add('peer_transport.hpp','__aicore__ inline void CacheClosDirectories(const ascend_deepep::DispatchTiling& t,','__aicore__ inline void CacheClosDirectories(const ascend_deepep::DispatchTiling& t, uint32_t logical_role,')
add('peer_transport.hpp','if (t.world <= 8U || GetBlockIdx() < 8U || GetBlockIdx() >= 24U)','if (t.world <= 8U || logical_role < 8U || logical_role >= 24U)')
add('completion_wait.hpp','__aicore__ inline void WaitDispatchIncoming(GM_ADDR workspace, const DispatchTiling& t, Scratch& scratch)','__aicore__ inline void WaitDispatchIncoming(GM_ADDR workspace, const DispatchTiling& t, uint32_t logical_role, Scratch& scratch)')
add('completion_wait.hpp','if (GetBlockIdx() != 7U)','if (logical_role != 7U)')
add('aux_pipeline.hpp','__aicore__ inline void Run(GM_ADDR workspace, const DispatchTiling& t, const PeerAuxPlan& plan,','__aicore__ inline void Run(GM_ADDR workspace, const DispatchTiling& t, const PeerAuxPlan& plan,\n                           uint32_t logical_role, uint32_t owner,')
add('aux_pipeline.hpp','    const uint32_t owner = AscendC::GetBlockIdx() - 32U;\n','')
add('aux_pipeline.hpp','(t.profile) + (owner + 32U) * 32U;','(t.profile) + logical_role * 32U;')
add('peer_scale.hpp','__aicore__ inline void PeerScales(GM_ADDR workspace, GM_ADDR input, GM_ADDR destinations, const DispatchTiling& t,','__aicore__ inline void PeerScales(GM_ADDR workspace, GM_ADDR input, GM_ADDR destinations, const DispatchTiling& t,\n                                  uint32_t logical_role, uint32_t aux_owner,')
add('peer_scale.hpp','DispatchScalePipeline::Run<Profile>(workspace, t, plan, staging, metadata, wqe_scratch);','DispatchScalePipeline::Run<Profile>(workspace, t, plan, logical_role, aux_owner, staging, metadata, wqe_scratch);')
add('scale_pipeline.hpp','__aicore__ inline void Run(GM_ADDR workspace, const DispatchTiling& t, const PeerScalePlan& plan,','__aicore__ inline void Run(GM_ADDR workspace, const DispatchTiling& t, const PeerScalePlan& plan,\n                           uint32_t logical_role, uint32_t worker,')
add('scale_pipeline.hpp','    const uint32_t worker = AscendC::GetBlockIdx() - 32U;\n','')
add('scale_pipeline.hpp','t.profile + uint64_t(AscendC::GetBlockIdx()) * 256U','t.profile + uint64_t(logical_role) * 256U')
def pin(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def prepare():
    manifest=dict(schema='P5-identity64-logical-role-patch-1',scope='Offline source candidate only. Original vector-only64 identity mapping; not mixed role/32reuse/tile-ready/hardware qualification.',files={},changes=changes,device_authorized=False,native_compile_authorized=False)
    patches=[]
    for name,edits in changes.items():
        rel='csrc/kernels/dispatch/'+name;original=(ROOT/rel).read_bytes();text=original.decode('utf8')
        for old,new in edits:
            if text.count(old)!=1:raise ValueError('source anchor drift: '+name+' '+repr(old))
            text=text.replace(old,new,1)
        after=text.encode('utf8');path=HERE/'candidate'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(after)
        diff=''.join(difflib.unified_diff(original.decode().splitlines(keepends=True),text.splitlines(keepends=True),fromfile='a/'+rel,tofile='b/'+rel))
        dp=HERE/'diffs'/(name+'.patch');dp.parent.mkdir(parents=True,exist_ok=True);dp.write_bytes(diff.encode())
        patches.append(diff);manifest['files'][rel]=dict(before=pin(original),after=pin(after),diff=pin(diff.encode()))
    patch=''.join(patches).encode();(HERE/'identity64.patch').write_bytes(patch);manifest['patch']=pin(patch)
    manifest['remaining_physical']=['candidate dispatch.cpp: one GetBlockIdx at top-level physical_role->logical_role=physical_role identity64.','Original peer_plan.hpp: NotifyDispatchPrefix/NotifyDispatchDirectory two independent physical partition uses unchanged; no Notify role refactor claimed.','low_latency subtree is separate entry and remains unchanged.','SHMEM/SDK internal physical worker/engine ownership is unchanged and not parameterized by this patch.']
    manifest['callsites']='Entire csrc search found only definitions/calls changed here; no external forward declarations found for these helper names in current files. Signature/definition sites included; C extern dispatch launcher ABI untouched.'
    manifest['uncompiled_risks']=['AscendC/bisheng compile/link has not run. Header include order resolves CacheClosDirectories before peer_address in current entry; source checks do not prove device compiler compatibility.','No new launch guard or runtime role validation added; unchanged native launch64 remains precondition.','No changes to events/fences/quiet/SIMT/UDMA ordering intended; native identity64 accuracy still requires separate bounded hardware qualification.']
    manifest['harness']={name:pin((HERE/name).read_bytes()) for name in ('prepare_patch.py','test_patch_cpu.py')}
    (HERE/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode());return manifest
if __name__=='__main__':
    m=prepare();print(json.dumps(dict(files=len(m['files']),patch=m['patch'])))
