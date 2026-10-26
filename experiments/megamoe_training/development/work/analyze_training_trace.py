"""Read real hardware tracks; never sum CPU ancestors or derived overlap tracks."""
import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path


def duration(e):
    return Decimal(str(e.get('dur', 0)))


def start(e):
    return Decimal(str(e['ts']))


def end(e):
    return start(e) + duration(e)


def category(e):
    name = e['name']
    if 'BasicMatmulTla' in name:
        return 'CATLASS_GEMM'
    if name in ('notify_full_kernel', 'dispatch_kernel', 'combine_kernel'):
        return name
    if e.get('args', {}).get('Task Type') == 'KERNEL_AICPU':
        return 'AICPU_lifetime_including_wait'
    if name in ('UBDMA', 'SDMA_SQE', 'NOTIFY_WAIT_SQE'):
        return name
    return name


def union(events, lo=None, hi=None):
    spans = sorted((max(start(e), lo) if lo is not None else start(e),
                    min(end(e), hi) if hi is not None else end(e)) for e in events)
    total, stop = Decimal(0), None
    for a, b in spans:
        if b <= a:
            continue
        total += b - max(a, stop) if stop is not None and b > stop else (b-a if stop is None else 0)
        stop = b if stop is None else max(stop, b)
    return float(total)


def item(e, origin):
    return dict(name=category(e), start_us=float(start(e)-origin), duration_us=float(duration(e)),
                stream=e.get('tid'), task_type=e.get('args', {}).get('Task Type'),
                task_id=e.get('args', {}).get('Task Id'))


def analyze(path):
    raw = path.read_bytes()
    all_events = json.loads(raw, parse_float=Decimal)
    if isinstance(all_events, dict):
        all_events = all_events['traceEvents']
    hw = {e['pid'] for e in all_events if e.get('ph') == 'M' and e.get('name') == 'process_name'
          and e.get('args', {}).get('name') == 'Ascend Hardware'}
    events = sorted((e for e in all_events if e.get('pid') in hw and e.get('ph') == 'X'), key=start)
    find = lambda name: [e for e in events if name in e['name']]
    out = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), hardware_pids=sorted(hw),
               hardware_event_count=len(events), iterations=[])
    gemms, dispatches, combines, notifies = find('BasicMatmulTla'), find('dispatch_kernel'), find('combine_kernel'), find('notify_full_kernel')
    if gemms and len(gemms) == 8*len(dispatches) == 8*len(combines) == 8*len(notifies):
        for i, (notify, dispatch, combine) in enumerate(zip(notifies, dispatches, combines)):
            up, down = gemms[i*8:i*8+4], gemms[i*8+4:i*8+8]
            assert end(notify) <= start(dispatch) <= end(dispatch) <= start(up[0])
            assert end(up[-1]) <= start(down[0]) <= end(down[-1]) <= start(combine)
            lo, hi = start(notify), end(combine)
            inside = [e for e in events if lo <= start(e) and end(e) <= hi]
            # Device task spans are observations, not an additive attribution of wall time.
            out['iterations'].append(dict(index=i, first_profile_iteration=(i==0),
                window='notify start through combine end; excludes output handoff and host API scope',
                span_us=float(hi-lo), notify_us=float(duration(notify)), dispatch_us=float(duration(dispatch)),
                gmm1_union_us=union(up), gmm2_union_us=union(down), combine_us=float(duration(combine)),
                after_notify_before_dispatch_us=float(start(dispatch)-end(notify)),
                after_gmm1_before_gmm2_us=float(start(down[0])-end(up[-1])),
                after_gmm2_before_combine_us=float(start(combine)-end(down[-1])),
                contained_hardware_union_us=union(inside),
                tasks=[item(e, lo) for e in inside if e.get('args',{}).get('Task Type','').startswith('KERNEL_')]))
        out['path_kind'] = 'P0_forward'
    else:
        ups = [e for e in events if e['name']=='AlltoAllvGroupedMatMul']
        downs = [e for e in events if e['name']=='GroupedMatMulAlltoAllv']
        aicpu_up = find('AlltoAllvGroupedMatMulMc2AicpuKernel')
        aicpu_down = find('GroupedMatMulAlltoAllvMc2AicpuKernel')
        if len(ups) == len(downs) == len(aicpu_up) == len(aicpu_down) and ups:
            for i, (up, down, cu, cd) in enumerate(zip(ups, downs, aicpu_up, aicpu_down)):
                lo, hi = min(start(up), start(cu)), max(end(down), end(cd))
                out['iterations'].append(dict(index=i, first_profile_iteration=(i==0),
                    window='up compute/AICPU start through down compute/AICPU end; excludes pack and output combine',
                    span_us=float(hi-lo), compute_up_us=float(duration(up)), compute_down_us=float(duration(down)),
                    aicpu_up_lifetime_us=float(duration(cu)), aicpu_down_lifetime_us=float(duration(cd)),
                    between_compute_kernels_us=float(start(down)-end(up)),
                    dma_observation={n:dict(clipped_union_us=union(find(n),lo,hi),event_count=sum(start(e)<hi and end(e)>lo for e in find(n))) for n in ('UBDMA','SDMA_SQE')},
                    tasks=[item(e,lo) for e in (up,down,cu,cd)]))
            out['path_kind'] = 'CANN_AICPU_forward'
        else:
            out['path_kind']='unrecognized_no_inferred_iterations'
            out['task_types']=sorted({e.get('args',{}).get('Task Type','') for e in events})
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    reports=[analyze(f) for f in sorted(a.root.rglob('*_trace.json'))]
    report=dict(scope='Local per-device profiler observations only; timestamps use exact decimal subtraction',
                limitations=['First profile iteration includes profiler initialization rank skew; retain it but do not treat as steady state.',
                             'No cross-rank clock alignment, comm-only/compute-only controls, wire byte counters or overlap efficiency measurement.',
                             'AICPU lifetimes include waits. DMA unions do not establish useful payload, bandwidth or hidden communication.',
                             'Contained hardware union excludes events crossing a window boundary; it is not total busy time.',
                             'Different recorded P0/CANN device windows cannot be used for E2E comparison; use unprofiled same-scope timings.'],
                analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),traces=reports)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(trace_count=len(reports),kinds={k:sum(r['path_kind']==k for r in reports) for k in {r['path_kind'] for r in reports}},output=str(a.output))))


if __name__=='__main__':main()
