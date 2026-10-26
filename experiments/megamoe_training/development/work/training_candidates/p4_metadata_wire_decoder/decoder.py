"""Read copied little-endian wire arrays only; never runs or reconstructs Notify.

Not an Expected-builder: distributed writer ownership and runtime issue seq are
deliberately absent. No Torch/native/network/file I/O in decoder functions.
"""
import struct
LIMIT=128<<20
def natural(n,minimum=0):
    if type(n) is not int or n<minimum:raise ValueError('invalid extent')
    return n
def words(blob,code,count):
    natural(count);width=struct.calcsize('<'+code)
    if type(blob) is not bytes or len(blob)!=width*count or len(blob)>LIMIT:raise ValueError('wrong copied byte extent/cap')
    return [v[0] for v in struct.iter_unpack('<'+code,blob)]
def dimensions(world,tokens,topk):
    natural(world,1);natural(tokens);natural(topk,1)
    if world>512 or topk>16:raise ValueError('outside reviewed ABI')
def destinations(blob,*,world,tokens,topk,stride,rank_rows):
    dimensions(world,tokens,topk);natural(stride,1)
    if len(rank_rows)!=world or any(natural(n)>stride for n in rank_rows):raise ValueError('rank rows/encoding stride mismatch')
    out=[]
    for route,raw in enumerate(words(blob,'q',tokens*topk)):
        if raw==-(1<<63):out.append(dict(route=route,valid=False));continue
        flat=~raw if raw<0 else raw;peer,row=divmod(flat,stride)
        if peer>=world or row>=rank_rows[peer]:raise ValueError('encoded occurrence outside actual rows')
        out.append(dict(route=route,token=route//topk,slot=route%topk,valid=True,primary=raw>=0,destination_rank=peer,final_row=row))
    return out
def return_metadata(blob,*,world,tokens,topk,local_counts):
    dimensions(world,tokens,topk);counts=[natural(n) for n in local_counts]
    values=words(blob,'i',sum(counts));out=[];row=0
    for expert,n in enumerate(counts):
        for localrow in range(n):
            encoded=values[row]
            if encoded<0 or encoded>=world*tokens*topk:raise ValueError('invalid source route in actual expanded rows')
            source_token,slot=divmod(encoded,topk);source,token=divmod(source_token,tokens)
            out.append(dict(expert_local=expert,expert_row=localrow,final_row=row,source=source,token=token,slot=slot));row+=1
    return out
def forwards(blob,counts_blob,*,rank,world,tokens,topk,dedup_group_size=8):
    dimensions(world,tokens,topk);natural(rank)
    if rank>=world or dedup_group_size!=8:raise ValueError('this fragment parser only reviews server-dedup8 normal-token ABI')
    capacity=max(1,tokens*topk);counts=words(counts_blob,'i',world);data=words(blob,'i',world*capacity*6);out=[]
    for source,count in enumerate(counts):
        if count<0 or count>tokens*topk:raise ValueError('forward count exceeds actual source routes')
        for record in range(count):
            end,token,primary,target,final,gather=data[(source*capacity+record)*6:(source*capacity+record+1)*6]
            if not record<end<=count or not 0<=token<tokens or primary<0 or not -1<=target<world or gather< -1:raise ValueError('invalid valid forward record')
            if target<0 and final!=-1 or target>=0 and final<0:raise ValueError('invalid secondary destination')
            out.append(dict(source=source,record=record,token_group_record_end=end,token=token,primary_gateway_row=primary,target_rank=target,final_target_row=final,gather_row=gather))
    return out
def peer_plan(blob,*,rank,world,tokens,topk):
    dimensions(world,tokens,topk);natural(rank)
    if rank>=world:raise ValueError('rank outside world')
    align=lambda n:(n+511)//512*512
    rows_offset=align(64+world*32);scratch_offset=rows_offset+align(tokens*topk*8)
    mailbox_offset=scratch_offset+align(world*((tokens+15)//16)*16)
    mailbox_bytes=2*(8*(128+128*4+64*4)*8)+(16*64*16*8)
    if type(blob) is not bytes or len(blob)!=mailbox_offset+mailbox_bytes or len(blob)>LIMIT:raise ValueError('wrong plan byte extent/cap')
    header=words(blob[:64],'I',16)
    if header[4]!=3 or header[5]!=world*4 or header[6]*4!=mailbox_offset or header[7]!=8:raise ValueError('wrong reviewed plan ABI/mode/offset')
    directory=words(blob[64:64+world*32],'i',world*8);pairs=[];cursor=0
    for phase in range(2):
        for peer in range(world):
            for qp in range(2):
                at=((phase*world+peer)*2+qp)*2;begin,count=directory[at:at+2]
                if begin!=cursor or count<0 or begin+count>tokens*topk or peer==rank and count:raise ValueError('directory not a disjoint packed prefix')
                for i in range(begin,begin+count):
                    token,row=struct.unpack_from('<ii',blob,rows_offset+i*8)
                    if not 0<=token<tokens or row<0 or peer//8==rank//8 and qp:raise ValueError('invalid primary pair')
                    pairs.append(dict(source=rank,peer=peer,phase='relay' if phase==0 else 'direct',qp=qp,pair_index=i,input_token=token,final_peer_row=row))
                cursor+=count
    return dict(header_options=header[12],rows_offset=rows_offset,scratch_offset=scratch_offset,mailbox_offset=mailbox_offset,pairs=pairs,used_pairs=cursor)
