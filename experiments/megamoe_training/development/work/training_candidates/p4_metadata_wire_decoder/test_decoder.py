"""Source-structured byte samples, NOT a real NPU dump."""
import json,pathlib,struct,sys
import decoder as d
checks=[]
def pack(code,x):return struct.pack('<'+code*len(x),*x)
def reject(f):
    try:f()
    except ValueError:return
    raise AssertionError('bad wire accepted')
r=d.destinations(pack('q',[0,~5,-(1<<63),6]),world=2,tokens=2,topk=2,stride=4,rank_rows=[1,3])
assert r[0]['primary'] and not r[1]['primary'] and r[1]['destination_rank']==1 and r[1]['final_row']==1 and not r[2]['valid'];checks.append('signed primary/complement secondary and INT64_MIN sentinel')
reject(lambda:d.destinations(pack('q',[3]),world=2,tokens=1,topk=1,stride=4,rank_rows=[1,3]));checks.append('encoded padding/out-of-range actual row rejected')
r=d.return_metadata(pack('i',[0,11,4]),world=2,tokens=3,topk=2,local_counts=[2,0,1]);assert r[1]['source']==1 and r[1]['token']==2 and r[1]['slot']==1 and r[2]['expert_local']==2;checks.append('weight_return_meta actual expert segment to source/token/slot with empty expert')
reject(lambda:d.return_metadata(pack('i',[-1]),world=2,tokens=3,topk=2,local_counts=[1]));checks.append('unwritten invalid source metadata rejected')
r=d.forwards(pack('i',[1,0,3,1,7,0]+[999]*6),pack('i',[1,0]),rank=0,world=2,tokens=1,topk=1)
assert r[0]['source']==0 and r[0]['token_group_record_end']==1 and r[0]['final_target_row']==7;checks.append('forward source is partition not column0; ignore uninitialized padded records')
reject(lambda:d.forwards(pack('i',[1,0,3,1,7,0]+[999]*6),pack('i',[2,0]),rank=0,world=2,tokens=1,topk=1));checks.append('forward valid counts bound before padding access')
# Exact source allocation arithmetic plus source-constructed packed prefix.
world=16;tokens=1;topk=2;rows=1024;scratch=1536;mailbox=2048;blob=bytearray(mailbox+245760)
header=[0]*16;header[4:8]=[3,world*4,mailbox//4,8];header[12]=1412;struct.pack_into('<16I',blob,0,*header)
cursor=0
for phase in range(2):
 for peer in range(world):
  for qp in range(2):
   count=int((phase,peer,qp) in ((0,8,0),(1,8,1)))
   struct.pack_into('<ii',blob,64+((phase*world+peer)*2+qp)*8,cursor,count);cursor+=count
struct.pack_into('<4i',blob,rows,0,17,0,29)
r=d.peer_plan(bytes(blob),rank=0,world=16,tokens=1,topk=2)
assert r['rows_offset']==rows and r['used_pairs']==2 and [p['qp'] for p in r['pairs']]==[0,1];checks.append('peer plan exact header phase/peer/QP directory and pair offsets')
bad=bytearray(blob);struct.pack_into('<I',bad,28,9);reject(lambda:d.peer_plan(bytes(bad),rank=0,world=16,tokens=1,topk=2));checks.append('unreviewed ABI rejects')
reject(lambda:d.peer_plan(bytes(blob[:-1]),rank=0,world=16,tokens=1,topk=2));checks.append('truncated copied plan rejects')
assert not any(n=='torch' or n.startswith(('torch.','torch_npu','deep_ep')) for n in sys.modules)
out=dict(passed=True,checks=len(checks),items=checks,actual_NPU_dump=False,scope='stdlib partial wire fragment decoder; not Expected writer mapper/Notify implementation/runtime sequence extractor')
path=pathlib.Path(__file__).with_name('cpu_results.json');path.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out,indent=2))
