"""Stdlib component audit. Never grants full-training/performance qualification."""
import math
def audit(report,manifest,compare_identity):
    expected=manifest['component_configuration'];cases=expected['cases'];metrics=empty=native=0
    if report.get('all_passed') is not True or report.get('configuration')!=expected or report.get('error'):raise ValueError('Component report failed/config differs')
    if report.get('source_sha256')!=manifest['package_source_sha256']:raise ValueError('Component source inventory differs')
    def identity(a,b):
        compare_identity(a,b)
        for value in (a,b):
            c=value['content'];selected=value['selected_library']
            if c['sources']!=manifest['package_source_sha256'] or c['configuration']!=expected or c['selected_SO_SHA256']!=manifest['SO']:raise ValueError('Identity source/config/SO differs')
            if c.get('precision')!={'hf32':False,'internal_raw':'disable'} or c.get('hardware',{}).get('device')!=0:raise ValueError('Identity precision/device differs')
            table={v['path']:v for v in value['actual_maps']['libraries']}
            if selected.get('path')!=manifest['library'] or selected.get('sha256')!=manifest['SO'] or selected.get('bytes')!=606568 or table.get(selected['path'])!=selected:raise ValueError('Selected actual FD record mismatch')
            core=value.get('torch_npu_C',{})
            if not core or table.get(core.get('path'))!=core:raise ValueError('PTA extension FD missing')
    identity(report.get('identity_initial'),report.get('identity_final'))
    if report.get('final_maps_error') or report.get('final_best_effort_maps',{}).get('errors'):raise ValueError('Final map capture failed')
    actual=report.get('cases',[])
    if len(actual)!=len(cases):raise ValueError('Incomplete case sequence')
    for index,(spec,item) in enumerate(zip(cases,actual)):
        m=sum(spec['counts']);h,i=spec['H'],spec['I']
        if any(item.get(k)!=v for k,v in spec.items()) or item.get('seed')!=expected['seed']+101*index or item.get('M')!=m:raise ValueError('Case/counts/seed differs')
        if any(item.get(k) is not True for k in ('passed','inputs_unchanged','independent_outputs_and_flags','streams_distinct')):raise ValueError('Case failed ownership/flags')
        if item.get('error') or item.get('identity_final_error') or item.get('synchronize_error'):raise ValueError('Case error present')
        identity(item.get('identity_initial'),item.get('identity_final'))
        invocations=item.get('invocations',[])
        if len(invocations)!=2 or len(item.get('streams',[]))!=2:raise ValueError('Invocation/stream count differs')
        if spec.get('concurrent') and len(set(item['streams']))!=2:raise ValueError('Dual streams not distinct')
        for invocation in invocations:
            if invocation.get('passed') is not True or invocation.get('flags_exact') is not True or set(invocation.get('metrics',{}))!={'GU','A','Z'}:raise ValueError('Invocation flags/metric set failed')
            for name,width in (('GU',2*i),('A',i),('Z',h)):
                metric=invocation['metrics'][name];n=m*width
                if (metric.get('shape')!=[m,width] or metric.get('elements')!=n or metric.get('dtype')!='torch.bfloat16' or metric.get('limits')!=expected['limits'] or metric.get('finite') is not True or metric.get('badcount')!=0 or metric.get('passed') is not True):raise ValueError('Full element metric invalid')
                for k in ('RMS_error','reference_RMS','RMS_relative_error','max_abs_error','max_relative_error_floor_1e_8'):
                    if type(metric.get(k)) not in (int,float) or not math.isfinite(metric[k]) or metric[k]<0:raise ValueError('Invalid metric number')
                rms,ref=metric['RMS_error'],metric['reference_RMS']
                if not math.isclose(metric['RMS_relative_error'],rms/max(ref,1e-12),rel_tol=1e-9,abs_tol=1e-12) or (rms/max(ref,1e-12)>.02 if ref else rms>.003):raise ValueError('Fixed RMS failed')
                metrics+=1
                if m: native+=1
                else: empty+=1
    if (metrics,empty,native)!=(48,6,42):raise ValueError('Metric accounting differs')
    return dict(component_only=True,metric_summaries=metrics,host_empty_summaries=empty,native_summaries=native,full_training_qualified=False,performance_qualified=False)
