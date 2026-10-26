"""Join persistent copied wire fragments, returning LogicalWriter only.

Requires sibling frozen p4_metadata_wire_decoder on Python import path.
No physical SQ alias assignment, dynamic seq, device I/O or Notify algorithm.
"""
from dataclasses import dataclass
from p4_metadata_wire_decoder import decoder as D

@dataclass(frozen=True)
class RankDump:
    rank: int
    local_counts: tuple
    original_stride: int
    topk_idx: bytes
    dst: bytes
    peer_plan: bytes
    forward_list: bytes
    forward_counts: bytes
    weight_return_meta: bytes

def local_first(world):return 7 if world==8 else 24

def primary_owner(total,route,world):
    """dispatch.cpp DispatchSplitRange: front-loaded remainder, NOT floor(total*w/n)."""
    workers=32-local_first(world);whole,remainder=divmod(total,workers)
    owners=[]
    for worker in range(workers):
        first=worker*whole+min(worker,remainder);end=first+whole+(worker<remainder)
        if first<=route<end:owners.append(local_first(world)+worker)
    if len(owners)!=1:raise ValueError('self primary range not uniquely owned')
    return owners[0]

def expansion_owner(source,total_routes,forward_count,record,world):
    """peer_order.hpp SplitLocalForwardRange: rotate by source*T*K global CSR base."""
    begin=source*total_routes;workers=32-local_first(world);whole,remainder=divmod(forward_count,workers);owners=[]
    for worker in range(workers):
        rotated=(worker+workers-begin%workers)%workers
        first=begin+rotated*whole+min(rotated,remainder);end=first+whole+(rotated<remainder)
        if first<=begin+record<end:owners.append(local_first(world)+worker)
    if len(owners)!=1:raise ValueError('local secondary range not uniquely owned')
    return owners[0]

def fm_role(sender,target):
    if sender//8!=target//8 or sender==target:raise ValueError('FM final writer must target another same-server rank')
    role=(target%8-sender%8-1)%8
    if role>=7:raise ValueError('invalid FM logical role')
    return role

def join(dumps,*,world,tokens,topk,num_experts):
    if world not in (8,16) or type(num_experts) is not int or num_experts<=0 or num_experts%world:raise ValueError('bounded reviewed world8/16 and integral localexperts required')
    D.dimensions(world,tokens,topk)
    if len(dumps)!=world or any(not isinstance(d,RankDump) or type(d.rank) is not int for d in dumps) or sorted(d.rank for d in dumps)!=list(range(world)):raise ValueError('exactly one dump for each rank required')
    raw={d.rank:d for d in dumps};local_experts=num_experts//world
    if any(len(d.local_counts)!=local_experts for d in dumps):raise ValueError('local count extent mismatch')
    rows=[sum(D.natural(n) for n in raw[r].local_counts) for r in range(world)];stride=max(1,max(rows));decoded={}
    for rank in range(world):
        d=raw[rank]
        if d.original_stride!=stride:raise ValueError('Notify original stride mismatch; Dispatch power2 stride is not accepted')
        ids=D.words(d.topk_idx,'q',tokens*topk)
        if any(v< -1 or v>=num_experts for v in ids):raise ValueError('unreviewed topk expert id')
        decoded[rank]=dict(ids=ids,dst=D.destinations(d.dst,world=world,tokens=tokens,topk=topk,stride=stride,rank_rows=rows),
            returns=D.return_metadata(d.weight_return_meta,world=world,tokens=tokens,topk=topk,local_counts=d.local_counts),
            pairs=D.peer_plan(d.peer_plan,rank=rank,world=world,tokens=tokens,topk=topk)['pairs'],
            forward=D.forwards(d.forward_list,d.forward_counts,rank=rank,world=world,tokens=tokens,topk=topk),
            forward_counts=D.words(d.forward_counts,'i',world))
    # Final row -> original route occurrence, validated against that source's
    # persistent signed destination and topk expert id. No forward col5 slot use.
    finals={};origins={}
    for rank in range(world):
        for e in decoded[rank]['returns']:
            src=e['source'];route=e['token']*topk+e['slot'];key=(rank,e['final_row']);origin=(src,route)
            sd=decoded[src]['dst'][route];expert=rank*local_experts+e['expert_local']
            if origin in origins or not sd['valid'] or (sd['destination_rank'],sd['final_row'])!=key or decoded[src]['ids'][route]!=expert:raise ValueError('original source/token/slot/expert/final-row inconsistent or duplicated')
            finals[key]=dict(e,rank=rank,expert=expert,primary=sd['primary']);origins[origin]=key
    for source in range(world):
        for route,idx in enumerate(decoded[source]['ids']):
            sd=decoded[source]['dst'][route]
            if (idx>=0)!=sd['valid'] or (idx>=0)!=((source,route) in origins):raise ValueError('valid route missing/extra final occurrence')
    pairs={};pair_used=set();secondary={};direct_seen=set();primary_has_secondary=set()
    for source in range(world):
        for p in decoded[source]['pairs']:
            key=(source,p['peer'],p['final_peer_row'],p['input_token'])
            if key in pairs:raise ValueError('duplicate primary pair')
            pairs[key]=p
    for gateway in range(world):
        for f in decoded[gateway]['forward']:
            primary=finals.get((gateway,f['primary_gateway_row']))
            if primary is None or not primary['primary'] or primary['source']!=f['source'] or primary['token']!=f['token']:raise ValueError('forward primary read source/row mismatch')
            if f['target_rank']<0:
                key=(gateway,f['source'],f['token'],f['primary_gateway_row'])
                if key in direct_seen:raise ValueError('duplicate direct forward record')
                direct_seen.add(key);continue
            target=f['target_rank'];key=(target,f['final_target_row']);expected=finals.get(key)
            if target//8!=gateway//8 or expected is None or expected['primary'] or expected['source']!=f['source'] or expected['token']!=f['token'] or key in secondary:raise ValueError('missing/duplicate/inconsistent secondary forward')
            secondary[key]=(gateway,f);primary_has_secondary.add((gateway,f['primary_gateway_row']))
    result=[]
    for key,e in sorted(finals.items()):
        target,row=key;source=e['source'];route=e['token']*topk+e['slot'];fields=dict(record_type='LogicalWriter',final_rank=target,expert=e['expert'],expert_local=e['expert_local'],expert_row=e['expert_row'],final_row=row,source=source,token=e['token'],slot=e['slot'],native_domain=None,seq=None)
        if e['primary']:
            direct_key=(target,source,e['token'],row)
            if (key in primary_has_secondary)==(direct_key in direct_seen):raise ValueError('primary must have either secondary references or one direct record')
            if source==target:
                writer=dict(kind='local_primary',writer_rank=source,logical_role=primary_owner(tokens*topk,route,world))
            else:
                pkey=(source,target,row,e['token'])
                if pkey not in pairs:raise ValueError('missing primary source plan pair')
                p=pairs[pkey];pair_used.add(pkey)
                expected_phase='relay' if key in primary_has_secondary else 'direct'
                if p['phase']!=expected_phase:raise ValueError('pair relay/direct phase inconsistent with persistent forward records')
                if source//8==target//8:
                    if p['qp']!=0:raise ValueError('FM primary cannot use logicalQP1')
                    writer=dict(kind='FM_primary',writer_rank=source,logical_role=fm_role(source,target),logical_qp=0,peer=target,phase=p['phase'])
                else:
                    owner=target%8;writer=dict(kind='Clos_primary',writer_rank=source,logical_role=8+owner+8*p['qp'],logical_owner=owner,logical_qp=p['qp'],peer=target,phase=p['phase'],cross_qp_index=2*owner+p['qp'])
        else:
            if key not in secondary:raise ValueError('secondary has no unique gateway forward record')
            gateway,f=secondary[key]
            if gateway==target:
                role=expansion_owner(source,tokens*topk,decoded[gateway]['forward_counts'][source],f['record'],world)
                writer=dict(kind='local_expand',writer_rank=gateway,logical_role=role,forward_record=f['record'])
            else:writer=dict(kind='FM_secondary',writer_rank=gateway,logical_role=fm_role(gateway,target),logical_qp=0,peer=target,forward_record=f['record'])
        result.append(dict(fields,**writer))
    if set(pairs)!=pair_used:raise ValueError('plan contains unmatched primary pairs')
    return result
