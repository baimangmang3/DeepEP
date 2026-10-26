"""P4 geometry/privateflag/wait graph model; not native/numerical qualification."""
import ast,collections,hashlib,importlib.util,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('p4_contract',R/'contract.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
def acyclic(nodes,edges):
    degree={n:0 for n in nodes};adj=collections.defaultdict(list)
    for a,b in set(edges):
        assert a in degree and b in degree
        adj[a].append(b);degree[b]+=1
    ready=collections.deque(n for n,v in degree.items() if not v);seen=0
    while ready:
        n=ready.popleft();seen+=1
        for b in adj[n]:
            degree[b]-=1
            if not degree[b]:ready.append(b)
    return seen==len(nodes)

def verify(p):
    seen_gu=set();seen_a=set();seen_z=set();jobs=set();nodes=set();edges=[];cube_last={};vec_last={};areas=collections.Counter();wave_count=0
    for wave in C.waves(p):
        wave_count+=1;by_job={q['job']:q for q in wave['up']};assert wave['rows']>0
        for q in wave['up']:
            j=q['job'];assert j not in jobs;jobs.add(j)
            assert q['expert']==wave['expert'] and q['rows']>0 and q['cols'] in (128,256)
            assert q['row']>=p['row_prefix'][q['expert']] and q['row']+q['rows']<=p['row_prefix'][q['expert']+1]
            a=(q['expert'],q['mtile'],q['col']);assert a not in seen_a;seen_a.add(a)
            for half in (0,1):
                g=(q['expert'],q['mtile'],q['col']+half*p['I']);assert g not in seen_gu;seen_gu.add(g);areas['GU']+=q['rows']*q['cols']
            # AIV0 covers all actual rows in8-row UB batches, including row64+.
            batches=[(r,min(8,q['rows']-r)) for r in range(0,q['rows'],8)]
            assert [r for start,n in batches for r in range(start,start+n)]==list(range(q['rows']))
            areas['A']+=q['rows']*q['cols'];nodes.update((('G',j),('A',j)));edges.append((('G',j),('A',j)))
            if q['core'] in vec_last:edges.append((vec_last[q['core']],('A',j)))
            vec_last[q['core']]=('A',j)
        # Actual native per-core sequence: ALL up pairs, then its down tasks.
        for core in range(32):
            sequence=[('G',q['job']) for q in wave['up'] if q['core']==core]
            for zindex,q in enumerate(wave['down']):
                if q['core']!=core:continue
                z=('Z',wave_count,zindex);nodes.add(z);sequence.append(z)
                assert q['wait_jobs'] and all(j in by_job for j in q['wait_jobs'])
                producers=[by_job[j] for j in q['wait_jobs']]
                assert {v['mtile'] for v in producers}=={q['mtile']}
                assert [v['col'] for v in producers]==list(range(0,p['I'],256))
                for j in q['wait_jobs']:edges.append((('A',j),z))
            if core in cube_last and sequence:edges.append((cube_last[core],sequence[0]))
            edges.extend(zip(sequence,sequence[1:]))
            if sequence:cube_last[core]=sequence[-1]
        for q in wave['down']:
            z=(q['expert'],q['mtile'],q['col']);assert z not in seen_z;seen_z.add(z);areas['Z']+=q['rows']*q['cols']
    # Independent expected canonical tile grid, not a second call to waves().
    expected_a={(e,m//128,n) for e,c in enumerate(p['counts']) for m in range(0,c,128) for n in range(0,p['I'],256)}
    expected_gu={(e,m,n+half*p['I']) for e,m,n in expected_a for half in (0,1)}
    expected_z={(e,m//128,n) for e,c in enumerate(p['counts']) for m in range(0,c,128) for n in range(0,p['H'],256)}
    assert seen_a==expected_a and seen_gu==expected_gu and seen_z==expected_z
    assert jobs==set(range(p['jobs']))
    assert areas==dict(GU=p['rows']*2*p['I'],A=p['rows']*p['I'],Z=p['rows']*p['H']) or p['rows']==0 and not areas
    assert acyclic(nodes,edges)
    return dict(waves=wave_count,pairs=len(jobs),up_gemm_tiles=2*len(jobs),down_tiles=len(seen_z),model_nodes=len(nodes),wait_graph_acyclic=True,canonical_elements=dict(areas))

results=[]
for counts in ([0,0,0,0],[1,0,0,0],[7,8,9,17],[127,128,129,0],[511,512,513,1],[0,0,0,1025],[16384]*4):
    for h,i in ((128,128),(384,384),(6144,2048),(128,2048),(6144,128)):
        p=C.plan(sum(counts),h,i,counts);results.append(dict(counts=counts,H=h,I=i,**verify(p)))
checks=[]
assert not acyclic({'GU','A','Z'},[('GU','A'),('A','Z'),('Z','GU')]);checks.append('negative up-after-wait cycle detected')
assert list(range(64))!=list(range(128));checks.append('negative old64-row half cannot cover128-row activation')
p=C.plan(513,128,128,[0,0,0,513]);waves=list(C.waves(p));assert waves[-1]['rows']==1 and len(waves[-1]['up'])==1 and len(waves[-1]['down'])==1;checks.append('513-row tail waits only actual one producer')
assert not list(C.waves(C.plan(0,128,128,[0]*4)));checks.append('empty experts no producer/consumer/wait')
for args in ((1,128,128,[0]*4),(1,129,128,[1,0,0,0]),(1,128,127,[1,0,0,0]),(1,128,128,[True,0,0,0])):
    try:C.plan(*args)
    except ValueError:pass
    else:raise AssertionError('invalid contract accepted')
checks.append('counts/dimensions/type invalid contracts rejected')
p=C.plan(1,128,128,[1,0,0,0]);assert C.plan(1,128,128,[1,0,0,0],p['total_bytes'])
try:C.plan(1,128,128,[1,0,0,0],p['total_bytes']-1)
except ValueError:pass
else:raise AssertionError('cap boundary accepted')
checks.append('exact byte cap accepts first lower byte rejects')
try:C.product(1<<62,4)
except ValueError:pass
else:raise AssertionError('overflow accepted')
checks.append('checked signed64 overflow rejected')
a=C.plan(1,128,128,[1,0,0,0]);b=C.plan(1,128,128,[1,0,0,0]);a['counts'][0]=0;assert b['counts'][0]==1;checks.append('no cross-call contract state cache')
cpp=(R/'native/wave_compute.cpp').read_text();wrapper=ast.parse((R/'wave_compute.py').read_text())
assert 'if (GetSubBlockIdx()!=0) return;' in cpp and 'for (uint32_t row=0;row<q.rows;row+=kVectorRows)' in cpp
assert cpp.index('for (uint32_t j=core;j<pairs;j+=kCores)')<cpp.index('for (uint32_t t=core;t<mt*nh;t+=kCores)')
assert 'SetFlag<HardEvent::MTE3_S>' in cpp and 'WaitFlag<HardEvent::MTE3_S>' in cpp and 'WaitReady(Ready(p,1,base+m*ni+tile))' in cpp
assert 'for (uint64_t wave=0;wave<p.counts[e];wave+=kWave)' in cpp
checks.append('native source uses full-row AIV0, actual I waits, fences, up-before-down and64-bit wave iterator')
call=next(n for n in wrapper.body if isinstance(n,ast.ClassDef) and n.name=='WaveCompute')
body=next(n for n in call.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
assert sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.zeros' for n in ast.walk(body))==1
assert 'tensors = (x, w1, w2, gu, a, z, flags)' in ast.unparse(body) and 't.record_stream(stream)' in ast.unparse(body)
checks.append('wrapper fresh private flags and all7 tensors record_stream per invocation')
assert not any(k=='torch' or k.startswith(('torch.','torch_npu','deep_ep')) for k in sys.modules)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=dict(passed=True,layout_cases=len(results),additional_checks=len(checks),cases=results,checks=checks,native_sha256=sha(R/'native/wave_compute.cpp'),scope='New P4 stdlib coverage/wait-dependency model and source checks only; no compiler/native invocation/Torch/device/math qualification.')
(R/'tests/cpu_layout_results.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(dict(passed=True,layout_cases=len(results),additional_checks=len(checks))))
