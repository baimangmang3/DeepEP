"""P0/P1 full-step hardware observations; never substitutes for formal timing.

Only the unchanged six-GEMM-stage serial schedules are recognized. P2's
reordering needs a separate mapping; do not silently classify it as P1.
"""
import argparse
import hashlib
import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path
from analyze_training_trace import start, end, duration, union, category, item


def summarize(path):
    raw = path.read_bytes()
    events = json.loads(raw, parse_float=Decimal)
    if isinstance(events, dict):
        events = events['traceEvents']
    hw = {e['pid'] for e in events if e.get('ph') == 'M' and e.get('name') == 'process_name'
          and e.get('args', {}).get('name') == 'Ascend Hardware'}
    labels = sorted((e for e in events if e.get('ph') == 'X' and
                     e.get('name', '').startswith(('training_perf_probe/', 'p1_comparison/'))), key=start)
    if len(labels) != 2 or not hw:
        raise ValueError('Expected two explicit full-step labels and hardware tracks')
    result = {'trace': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'iterations': []}
    for index, label in enumerate(labels):
        parts = label['name'].split('/')
        if len(parts) != 3 or parts[2] != 'forward_backward' or parts[1] not in (
                'shmem_catlass', 'shmem_catlass_pointwise_p1'):
            raise ValueError('Unknown schedule or profiling scope; no inferred stage mapping')
        backend = parts[1]
        lo, hi = start(label), end(label)
        tasks = sorted((e for e in events if e.get('ph') == 'X' and e.get('pid') in hw
                        and lo <= start(e) and end(e) <= hi
                        and e.get('args', {}).get('Task Type', '').startswith('KERNEL_')), key=start)
        gemms = [e for e in tasks if 'BasicMatmulTla' in e['name']]
        if (len(gemms) != 24 or sum(e['name'] == 'dispatch_kernel' for e in tasks) != 2
                or sum(e['name'] == 'combine_kernel' for e in tasks) != 2):
            raise ValueError('Expected routed-only EP16 four-expert serial full-step task counts')
        groups = {name: gemms[4*j:4*j+4] for j, name in enumerate(('up', 'down', 'dW2', 'dA', 'dW1', 'dXe'))}
        if any(end(a) > start(b) for a, b in zip(gemms, gemms[1:])):
            raise ValueError('Overlapping GEMMs do not match this serial stage mapping')
        boundaries = [('activation', end(gemms[3]), start(gemms[4])),
                      ('gate_combine_dispatch_dp_dz', end(gemms[7]), start(gemms[8])),
                      ('activation_grad', end(gemms[15]), start(gemms[16])),
                      ('return_and_tail', end(gemms[23]), hi)]
        record = {'index': index, 'backend': backend, 'label': label['name'],
                  'profile_label_us': float(duration(label)),
                  'kernel_names_and_counts': dict(Counter(category(e) for e in tasks)),
                  'GEMM_kernel_union_us': union(gemms),
                  'gemm_stages': {name: {'kernel_union_us': union(group),
                                        'begin_us_from_label': float(start(group[0])-lo),
                                        'end_us_from_label': float(end(group[-1])-lo)}
                                  for name, group in groups.items()},
                  'gaps': {name: {'span_us': float(b-a),
                                  'kernels': [item(e, a) for e in tasks if a <= start(e) and end(e) <= b]}
                           for name, a, b in boundaries}}
        pointwise = [e for e in tasks if 'Pointwise<' in e['name']]
        # Keep all task names if compiler/profiler naming changes; do not infer
        # a vector kernel's role from timing alone or hide unmatched kernels.
        record['recognized_pointwise_kernels'] = [item(e, lo) for e in pointwise]
        record['pointwise_count_matches_four'] = len(pointwise) == 4 if backend.endswith('_p1') else None
        record['all_hardware_kernels'] = [item(e, lo) for e in tasks]
        result['iterations'].append(record)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    reports = [summarize(path) for path in sorted(args.root.rglob('*_trace.json'))]
    if not reports:
        raise ValueError('No traces found')
    summaries = {}
    for backend in sorted({r['iterations'][1]['backend'] for r in reports}):
        later = [r['iterations'][1] for r in reports if r['iterations'][1]['backend'] == backend]
        summaries[backend] = {'trace_count': len(later),
            'GEMM_kernel_union_us_median': statistics.median(r['GEMM_kernel_union_us'] for r in later),
            'gaps': {name: {'median_us': statistics.median(r['gaps'][name]['span_us'] for r in later),
                            'min_us': min(r['gaps'][name]['span_us'] for r in later),
                            'max_us': max(r['gaps'][name]['span_us'] for r in later)} for name in later[0]['gaps']}}
    result = {'scope': 'Per-device routed-only serial full-step hardware observations',
        'limitations': ['Formal performance must come from separate unprofiled per-rank benchmark samples.',
                       'First and second steps retained separately; summaries use only the second step.',
                       'Gap spans include multiple kernels, submission and waiting; they are not pure communication time.',
                       'No cross-rank clock alignment, payload/link counters or comm-only/compute-only controls.',
                       'No stage-duration addition, bandwidth/overlap-efficiency estimate or MFU from these traces.',
                       'P0/P1 same serial GEMM order only; P2, shared and changed expert counts are rejected.'],
        'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256': hashlib.sha256(Path(__file__).with_name('analyze_training_trace.py').read_bytes()).hexdigest(),
        'second_step_summary': summaries, 'traces': reports}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'trace_count': len(reports), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
