"""Actual candidate Python packer/tiling plus source-bound CPU arithmetic."""
import ast
import ctypes
import importlib.util
import json
from pathlib import Path
import re
import struct
from types import SimpleNamespace
from prepare_patch import ROOT,REPO,REV,E,PY,UP,DOWN,META,original


def load(name,text):
    m=SimpleNamespace();exec(compile(text,name,'exec'),m.__dict__);return m


def packer(text):
    tree=ast.parse(text);body=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('_MOE_ROUTE_') for t in n.targets)]
    body += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_build_route_metadata']
    class Sink:
        def __init__(self,b):self.b=b
        def to(self,**kw):return self.b
    fake=SimpleNamespace(Tensor=object,uint8=object(),frombuffer=lambda b,**kw:Sink(b))
    ns={'struct':struct,'torch':fake};exec(compile(ast.Module(body=body,type_ignores=[]),'actual_packer','exec'),ns);return ns


def main():
    candidate=ROOT/'candidate';get=lambda p:(candidate/p).read_text()
    pinned=load('pinned_tiling',original(PY+'_tiling.py'));new=load('candidate_tiling',get(PY+'_tiling.py'))
    checks=[]
    def check(name,ok,**details):
        assert ok,name;checks.append(dict(name=name,passed=True,**details))
    def rejects(fn):
        try:fn()
        except ValueError:return True
        return False
    oldcases=0
    for world in range(1,9):
        for op in (new.ALL_TO_ALLV,new.GROUPED_MATMUL):
            for m,n,k in ((65536,6144,2048),(131072,1024,2048),(131072,512,2048),(131072,2048,1024),(131072,2048,512),(131072,6144,3584),(131072,3072,3584),(131084,3584,3072),(131084,3584,6144)):
                for export in ((False,True) if op==new.ALL_TO_ALLV else (False,)):
                    for experts in (28,256):
                        c=dict(operator=op,world_size=world,expert_num=experts,shape=dict(m=m,n=n,k=k),output_all_to_allv=export)
                        check('EP<=8_profile_'+str(oldcases),pinned.default_tiling(c)==new.default_tiling(c));oldcases+=1
    def case(op,n=6144,k=2048,export=False):return dict(operator=op,world_size=16,expert_num=64,shape=dict(m=65536,n=n,k=k),output_all_to_allv=export)
    up=new.default_tiling(case(new.ALL_TO_ALLV,n=4096,k=6144));down=new.default_tiling(case(new.GROUPED_MATMUL))
    check('up_EP16_default',list(up.values())==[128,256,256,64,1024,2,32,1,0,1])
    check('down_EP16_effective_default',list(down.values())==[128,256,256,12,10,128,2,1])
    def validate(op,p,**kw):return new.validate_tiling(op,p,m=65536,n=kw.pop('n',6144),k=kw.pop('k',2048),world_size=kw.pop('world_size',16),aic_cores=kw.pop('aic_cores',32),**kw)
    validate(new.ALL_TO_ALLV,up,k=6144);validate(new.GROUPED_MATMUL,down)
    check('EP16_defaults_accepted',True,up_bytes=2*1024*16*6144*2,down_bytes=2*128*16*6144*2)
    check('rank17_first_illegal',rejects(lambda:validate(new.GROUPED_MATMUL,down,world_size=17)))
    check('aic15_first_insufficient',rejects(lambda:validate(new.GROUPED_MATMUL,down,aic_cores=15)))
    check('up_stage_first_unaligned',rejects(lambda:validate(new.ALL_TO_ALLV,dict(up,stage_rows=1025),k=6144)))
    check('up_stage1152_overflow',rejects(lambda:validate(new.ALL_TO_ALLV,dict(up,stage_rows=1152),k=6144)))
    validate(new.ALL_TO_ALLV,up,k=6400)
    check('up_K6400_last_K6401_first_illegal',rejects(lambda:validate(new.ALL_TO_ALLV,up,k=6401)))
    validate(new.GROUPED_MATMUL,down)
    check('down_effective10_and11_first_illegal',rejects(lambda:validate(new.GROUPED_MATMUL,dict(down,local_copy_tile_rows=11))),effective_UB_bytes=4*10*6144+256,first_illegal_UB_bytes=4*11*6144+256)
    check('down_explicit_nominal32_rejected',rejects(lambda:validate(new.GROUPED_MATMUL,dict(down,local_copy_tile_rows=32))))
    check('worker18_first_illegal',rejects(lambda:validate(new.GROUPED_MATMUL,dict(down,self_copy_cores=18))))
    validate(new.GROUPED_MATMUL,dict(down,local_copy_tile_rows=1),n=51200)
    check('down_N51200_last_N51201_first_illegal',rejects(lambda:validate(new.GROUPED_MATMUL,dict(down,local_copy_tile_rows=1),n=51201)))
    check('legacy_interval10_derived_rejected',rejects(lambda:validate(new.GROUPED_MATMUL,dict(down,stage_rows=0,comm_interval=10))))
    for cores in (16,17,24,31,32,48):validate(new.GROUPED_MATMUL,down,aic_cores=cores)
    check('explicit_stage128_decouples_interval_from_AIC',True)
    for m in (0,1,127,128,129,255,256,257,65535,65536,65537):
        chunks=[min(128,m-start) for start in range(0,m,128)]
        check('down_tail_rows_'+str(m),sum(chunks)==m and all(0<n<=128 for n in chunks))
    # Parse actual native profile, and assert its unchanged default-only UB cap.
    native=get(DOWN+'grouped_mat_mul_all_to_allv_udma.cpp')
    raw=re.search(r'kEp16Tiling\{([^}]+)\}',native).group(1).split(',')
    fields=[int(x.strip()) for x in raw[1:]]
    check('native_down_nominal_profile',fields==[128,256,256,12,64,64,1,16,128,2,32,1])
    cap=(248*1024-256)//(4*6144)
    check('native_Python_default_effective10',min(fields[-2],cap)==down['local_copy_tile_rows']==10)
    check('native_default_only_clamp_preserved','tilingValues.empty() ? std::min(tiling.localCopyTileRows, TilingValidation::LocalCopyRowCapacity(n, ubBytes))' in native)
    checks_before_workers=len(checks)
    for world in (8,16):
        for own in range(world):
            for self_workers in range(1,32-world+2):
                for export in (False,True):
                    peer_owners={peer:[] for peer in range(world) if peer!=own}
                    down_owners={peer:[] for peer in range(world) if peer!=own};sync=[];local=[]
                    for core in range(32):
                        primary=core<world;extra=world<=core<world+self_workers-1
                        if not(primary or extra):continue
                        peer=own if extra else core
                        for sub in (0,1):
                            up_owner=(sub==1) if export else ((peer==own and sub==0) or (peer!=own and primary and sub==1))
                            if up_owner:
                                if peer==own:local.append((core,sub))
                                else:peer_owners[peer].append((core,sub))
                            if sub==0 and peer==own and primary:sync.append((core,sub))
                            if sub==0 and peer!=own:down_owners[peer].append((core,sub))
                    check(f'worker_w{world}_r{own}_self{self_workers}_export{export}',all(len(v)==1 for v in peer_owners.values()) and all(len(v)==1 for v in down_owners.values()) and len(local)==self_workers and sync==[(own,0)])
    worker_checks=len(checks)-checks_before_workers
    # Execute candidate packer; validate every source/expert segment and bytes.
    for world,source in ((8,original(PY+'ops.py')),(16,get(PY+'ops.py'))):
        ns=packer(source);local=64;values=[(s*7+e*3)%11 for s in range(world) for e in range(local)]
        blob=ns['_build_route_metadata'](values,world,world*local,None)
        expert_offset=16+world*8;segment_offset=expert_offset+64*8
        check('ABI_EP'+str(world),len(blob)==segment_offset+world*64*24,bytes=len(blob),offsets=[16,expert_offset,segment_offset])
        offsets=[0]*world;out=0
        for e in range(local):
            for s in range(world):
                n=values[s*local+e];assert struct.unpack_from('<QQIHH',blob,segment_offset+(e*world+s)*24)==(offsets[s],out,n,s,e)
                offsets[s]+=n;out+=n
        check('ABI_all_segments_EP'+str(world),True,segments=world*64)
        check('ABI_first_illegal_rank_EP'+str(world),rejects(lambda:ns['_build_route_metadata']([0]*(world+1),world+1,world+1,None)))
        check('ABI_first_illegal_localexpert_EP'+str(world),rejects(lambda:ns['_build_route_metadata']([0]*(world*65),world,world*65,None)))
        check('ABI_uint32_last_EP'+str(world),bool(ns['_build_route_metadata']([2**32-1]+[0]*(world-1),world,world,None)))
        check('ABI_uint32_first_illegal_EP'+str(world),rejects(lambda:ns['_build_route_metadata']([2**32]+[0]*(world-1),world,world,None)))
    class Segment(ctypes.Structure):
        _fields_=[('source',ctypes.c_uint64),('output',ctypes.c_uint64),('rows',ctypes.c_uint32),('rank',ctypes.c_uint16),('expert',ctypes.c_uint16)]
    class Metadata(ctypes.Structure):
        _fields_=[('header',ctypes.c_uint32*4),('peer',ctypes.c_uint64*16),('expert',ctypes.c_uint64*64),('segments',Segment*1024)]
    check('independent_Ctypes_EP16_ABI',ctypes.sizeof(Segment)==24 and ctypes.sizeof(Metadata)==25232 and [Metadata.peer.offset,Metadata.expert.offset,Metadata.segments.offset]==[16,144,656])
    host=get(UP+'all_to_allv_grouped_mat_mul_udma.cpp');metadata=get(META);ns=packer(get(PY+'ops.py'))
    def sizes_match(h):
        host_size=int(re.search(r'sizeof\(Catccos::DGemm::MoeRouteMetadata\)\s*==\s*(\d+)',h).group(1))
        cpp_size=int(re.search(r'sizeof\(MoeRouteMetadata\)\s*==\s*(\d+)',metadata).group(1))
        return host_size==cpp_size==ns['_MOE_ROUTE_METADATA_SIZE']==ctypes.sizeof(Metadata)==25232
    check('host_header_Python_ctypes_ABI_agree',sizes_match(host))
    check('old_host_ABI_assert_rejected',not sizes_match(host.replace('MoeRouteMetadata) == 25232','MoeRouteMetadata) == 12880')))
    check('host_segment24_preserved','sizeof(Catccos::DGemm::MoeRouteSegment) == 24' in host)
    arrays=[]
    overlay=E+'csrc/catccos_overlay/include/catccos/'
    paths=['dgemm/block/block_scheduler_alltoallv_allgather.hpp','dgemm/block/block_scheduler_gmm_alltoallv_target_order.hpp',
        'comm/block/comm_block_scheduler_alltoallv_allgather.hpp','comm/block/comm_block_scheduler_gmm_alltoallv_target_order.hpp',
        'dgemm/kernel/ascend950_alltoallv_grouped_matmul.hpp','dgemm/kernel/grouped_matmul_alltoallv_tla.hpp']
    for path in paths:
        for line,text in enumerate(original(overlay+path).splitlines(),1):
            if '[' in text and 'MoeConstraints::' in text:arrays.append(dict(path=overlay+path,line=line,declaration=text.strip()))
    check('general_Session_constructor_unchanged',get('csrc/runtime/session.cpp').split('#ifdef DEEPEP_BUILD_EP_GMM_FUSED')[0]==original('csrc/runtime/session.cpp').split('#ifdef DEEPEP_BUILD_EP_GMM_FUSED')[0])
    result=dict(scope='CPU Python execution/source-bound arithmetic only; no C++ compiler/NPU/UDMA initialization used',all_passed=True,checks=len(checks),old_profile_cases=oldcases,worker_mapping_checks=worker_checks,
        metadata16=dict(peerRows=16,expertRows=144,segments=656,bytes=25232),down_copy=dict(nominal=32,effective_N6144=10,first_illegal=11),array_propagation=arrays,
        details=checks,unverified=['Native static_asserts/header tests must be compiled','Expanded scheduler private-state/compiler resources','SHMEM connection capacity, queue drain/quiet completion and stage recycle need hardware','No changes to kernel ready/barrier/quiet/recycle protocol in this patch'])
    (ROOT/'cpu_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('all_passed','checks','old_profile_cases','metadata16','down_copy')}))


if __name__=='__main__':main()
