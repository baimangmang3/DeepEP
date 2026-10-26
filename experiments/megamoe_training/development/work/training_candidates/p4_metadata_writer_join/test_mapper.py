"""Coherent source-structured allrank bytes, no real NPU dump or Notify run."""
import dataclasses,json,pathlib,struct,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from mapper import RankDump,join,primary_owner,expansion_owner,fm_role
checks=[]
def pack(code,values):return struct.pack('<'+code*len(values),*values)
def reject(f):
    try:f()
    except ValueError:return
    raise AssertionError('incoherent metadata accepted')
W=16;T=2;K=4;F=T*K
def plan(rank,pairs):
    align=lambda n:(n+511)//512*512
    rows=align(64+W*32);scratch=rows+align(T*K*8);mailbox=scratch+align(W*((T+15)//16)*16)
    b=bytearray(mailbox+245760);h=[0]*16;h[4:8]=[3,W*4,mailbox//4,8];h[12]=1412;struct.pack_into('<16I',b,0,*h);at=0
    for phase in range(2):
      for peer in range(W):
       for qp in range(2):
        selected=[p for p in pairs if p[:3]==(phase,peer,qp)]
        struct.pack_into('<ii',b,64+((phase*W+peer)*2+qp)*8,at,len(selected))
        for _,_,_,token,row in selected:struct.pack_into('<ii',b,rows+8*at,token,row);at+=1
    return bytes(b)
def fixture():
    counts={0:(2,0),1:(2,0),8:(2,1),9:(1,0)}
    returns={0:[0,1],1:[2,7],8:[3,4,6],9:[5]}
    # (end, token, primarygatewayrow, finaltargetrank, finalrow, gatherrow)
    forward={0:[(2,0,0,0,1,-1),(2,0,0,1,0,0)],1:[(1,1,1,-1,-1,-1)],8:[(1,0,0,-1,-1,-1),(3,1,1,9,0,0),(3,1,1,8,2,-1)]}
    out=[]
    for rank in range(W):
        idx=[0,0,2,16,16,18,17,2] if rank==0 else [-1]*F
        dst=[0,~1,~3,24,25,~27,~26,4] if rank==0 else [-(1<<63)]*F
        pairs=[(0,8,0,1,1),(1,1,0,1,1),(1,8,0,0,0)] if rank==0 else []
        data=[987654]*(W*F*6);fc=[0]*W
        if rank in forward:
            fc[0]=len(forward[rank])
            for i,row in enumerate(forward[rank]):data[i*6:i*6+6]=row
        out.append(RankDump(rank,counts.get(rank,(0,0)),3,pack('q',idx),pack('q',dst),plan(rank,pairs),pack('i',data),pack('i',fc),pack('i',returns.get(rank,[]))))
    return out
dumps=fixture();result=join(dumps,world=W,tokens=T,topk=K,num_experts=32)
assert len(result)==8 and {x['kind'] for x in result}=={'local_primary','FM_primary','Clos_primary','local_expand','FM_secondary'}
assert all(x['seq'] is None and x['native_domain'] is None for x in result);checks.append('all5 final writer kinds across16rank joined from persistent copied arrays; logical only')
byorigin={(x['source'],x['token'],x['slot']):x for x in result}
assert byorigin[(0,0,0)]['expert']==byorigin[(0,0,1)]['expert']==0
assert byorigin[(0,0,0)]['final_row']!=byorigin[(0,0,1)]['final_row'];checks.append('duplicate expert/topk occurrences retain distinct slot/finalrow, never use forwardcol5 as slot')
assert byorigin[(0,1,2)]['kind']=='local_expand' and byorigin[(0,1,2)]['writer_rank']==8 and byorigin[(0,1,2)]['logical_role']==26
assert byorigin[(0,1,1)]['kind']=='FM_secondary' and byorigin[(0,1,1)]['writer_rank']==8;checks.append('crosshost primary then local/FM secondary retain originalsource and actualgateway')
assert primary_owner(10,0,16)==24 and primary_owner(10,1,16)==24 and primary_owner(10,2,16)==25
assert [expansion_owner(1,6,3,r,16) for r in range(3)]==[30,31,24]
assert fm_role(7,0)==0 and fm_role(0,7)==6;checks.append('exact front-loaded primary split, source*T*K rotated local split and wraparound FM inverse')
reject(lambda:join(dumps[:-1],world=W,tokens=T,topk=K,num_experts=32));reject(lambda:join(dumps+[dumps[0]],world=W,tokens=T,topk=K,num_experts=32));checks.append('missing or duplicate rank rejects')
bad=list(dumps);bad[0]=dataclasses.replace(bad[0],original_stride=4);reject(lambda:join(bad,world=W,tokens=T,topk=K,num_experts=32));checks.append('originalNotify stride3 cannot be replaced by Dispatch stride4')
for values in ([0,0],[0,7]):
    bad=list(dumps);bad[0]=dataclasses.replace(bad[0],weight_return_meta=pack('i',values));reject(lambda:join(bad,world=W,tokens=T,topk=K,num_experts=32))
checks.append('duplicate source occurrence or source-token-slot/finalrow mismatch rejects')
bad=list(dumps);bad[0]=dataclasses.replace(bad[0],peer_plan=plan(0,[(0,8,0,1,1),(1,1,0,1,1)]));reject(lambda:join(bad,world=W,tokens=T,topk=K,num_experts=32));checks.append('missing primary plan pair rejects')
bad=list(dumps);bad[0]=dataclasses.replace(bad[0],peer_plan=plan(0,[(0,8,0,1,1),(1,1,0,1,1),(1,8,0,0,0),(1,8,0,0,0)]));reject(lambda:join(bad,world=W,tokens=T,topk=K,num_experts=32));checks.append('duplicate primary plan pair rejects')
bad=list(dumps);blob=bytearray(bad[8].forward_list);struct.pack_into('<i',blob,6*4+2*4,0);bad[8]=dataclasses.replace(bad[8],forward_list=bytes(blob));reject(lambda:join(bad,world=W,tokens=T,topk=K,num_experts=32));checks.append('gateway primary row/token mismatches reject')
bad=list(dumps);blob=bytearray(bad[0].forward_counts);struct.pack_into('<i',blob,0,1);bad[0]=dataclasses.replace(bad[0],forward_counts=bytes(blob));reject(lambda:join(bad,world=W,tokens=T,topk=K,num_experts=32));checks.append('missing secondary forward record rejects')
bad=list(dumps);blob=bytearray(bad[0].forward_list);blob[6*4:12*4]=blob[:6*4];bad[0]=dataclasses.replace(bad[0],forward_list=bytes(blob));reject(lambda:join(bad,world=W,tokens=T,topk=K,num_experts=32));checks.append('duplicate secondary target occurrence rejects')
empty=[]
for d in fixture():empty.append(dataclasses.replace(d,local_counts=(0,0),original_stride=1,topk_idx=pack('q',[-1]*F),dst=pack('q',[-(1<<63)]*F),peer_plan=plan(d.rank,[]),forward_counts=pack('i',[0]*W),weight_return_meta=b''))
assert join(empty,world=W,tokens=T,topk=K,num_experts=32)==[];checks.append('all-empty and unwrittenpadding ignored; actualexpert tail rows only')
assert not any(n=='torch' or n.startswith(('torch.','torch_npu','deep_ep')) for n in sys.modules)
out=dict(passed=True,checks=len(checks),items=checks,scope='source-structured synthetic persistent-wire join only; no realNPUdump, physicaldomain or runtime sequence',logical_writers=result)
p=pathlib.Path(__file__).with_name('cpu_results.json');p.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(dict(passed=True,checks=len(checks),items=checks),indent=2))
