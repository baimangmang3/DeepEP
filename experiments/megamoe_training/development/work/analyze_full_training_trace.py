"""Per-device full-step trace observations; formal timings live in the benchmark reports."""
import hashlib
import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path
from analyze_training_trace import start, end, duration, union, category


def summarize(path):
    raw=path.read_bytes();events=json.loads(raw,parse_float=Decimal)
    hw={e['pid'] for e in events if e.get('ph')=='M' and e.get('name')=='process_name'
        and e.get('args',{}).get('name')=='Ascend Hardware'}
    labels=sorted([e for e in events if e.get('ph')=='X' and
                   e.get('name','').startswith('training_perf_probe/')],key=start)
    assert len(labels)==2
    result={'trace':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'iterations':[]}
    for i,label in enumerate(labels):
        lo,hi=start(label),end(label)
        tasks=sorted([e for e in events if e.get('ph')=='X' and e.get('pid') in hw
            and lo<=start(e) and end(e)<=hi and e.get('args',{}).get('Task Type','').startswith('KERNEL_')],key=start)
        gemms=[e for e in tasks if 'BasicMatmulTla' in e['name']]
        record={'index':i,'label':label['name'],'profile_label_us':float(duration(label)),
            'kernel_names_and_counts':dict(Counter(category(e) for e in tasks))}
        if '/shmem_catlass/' in label['name']:
            assert len(gemms)==24
            assert sum(e['name']=='dispatch_kernel' for e in tasks)==2
            assert sum(e['name']=='combine_kernel' for e in tasks)==2
            record['gemm_stages']={name:{'kernel_union_us':union(gemms[j*4:(j+1)*4]),
                'begin_us_from_label':float(start(gemms[j*4])-lo),
                'end_us_from_label':float(end(gemms[j*4+3])-lo)}
                for j,name in enumerate(('up','down','dW2','dA','dW1','dXe'))}
            boundaries=[('activation',end(gemms[3]),start(gemms[4])),
                ('gate_combine_dispatch_dp_dz',end(gemms[7]),start(gemms[8])),
                ('activation_grad',end(gemms[15]),start(gemms[16])),
                ('return_and_tail',end(gemms[23]),hi)]
            record['gaps']={name:{'span_us':float(b-a),
                'kernels':[{'name':category(e),'duration_us':float(duration(e)),
                    'begin_us_from_gap':float(start(e)-a),'stream':e['tid'],'task_id':e['args']['Task Id']}
                    for e in tasks if a<=start(e) and end(e)<=b]}
                for name,a,b in boundaries}
            record['GEMM_kernel_union_us']=union(gemms)
        else:
            assert len(gemms)==8
            fused=[e for e in tasks if e['name'] in ('AlltoAllvGroupedMatMul','GroupedMatMulAlltoAllv')]
            aicpu=[e for e in tasks if e['args']['Task Type']=='KERNEL_AICPU']
            assert len(fused)==4 and len(aicpu)==4
            record['fused_compute_kernel_durations_us']=[{'name':e['name'],'duration_us':float(duration(e)),
                'begin_us_from_label':float(start(e)-lo)} for e in fused]
            record['aicpu_kernel_lifetimes_us']=[{'name':e['name'],'duration_us':float(duration(e)),
                'begin_us_from_label':float(start(e)-lo)} for e in aicpu]
            record['CATLASS_FP32_wgrad_kernel_union_us']=union(gemms)
        result['iterations'].append(record)
    return result


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    reports=[summarize(p) for p in sorted(args.root.rglob('*_trace.json'))]
    result={'scope':'Per-device hardware tasks inside explicit aligned forward_backward CPU labels',
        'limitations':['Profiler timings are not official benchmark timings. First and second calls remain separately visible.',
            'No cross-device timestamp comparison. Outside/cross-label hardware events are excluded.',
            'P0 six GEMM stages mapped from unchanged qualified training.py order, each four local experts; counts checked.',
            'AICPU kernel lifetime includes waits. Do not add it to compute durations or infer communication overlap from kernel co-residency.',
            'No wire bytes, achieved network bandwidth, comm-only control or hardware utilization counters in this report.'],
        'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256(Path(__file__).with_name('analyze_training_trace.py').read_bytes()).hexdigest(),
        'traces':reports}
    p0=[r['iterations'][1] for r in reports if '/shmem_catlass/' in r['iterations'][1]['label']]
    if p0:
        result['P0_second_step_summary']={'kind':'P0 second profiled step per rank; observations not benchmark',
            'rank_count':len(p0),
            'GEMM_kernel_union_us_median':statistics.median(x['GEMM_kernel_union_us'] for x in p0),
            'gaps':{k:{'median_us':statistics.median(x['gaps'][k]['span_us'] for x in p0),
                       'min_us':min(x['gaps'][k]['span_us'] for x in p0),
                       'max_us':max(x['gaps'][k]['span_us'] for x in p0)} for k in p0[0]['gaps']}}
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'traces':len(reports),'iterations':sum(len(r['iterations']) for r in reports),'output':str(args.output)}))
