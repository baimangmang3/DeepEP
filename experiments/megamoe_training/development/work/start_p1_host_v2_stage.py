import sys,pathlib,json,subprocess
index=int(sys.argv[1]);tag=sys.argv[2];assert index in range(1,7)
configs=[('p1-host-v2-ep2-bf16',2,8,4,False,'bf16',32,128,128),('p1-host-v2-ep2-shared-fp32',2,8,4,True,'fp32',32,128,128),('p1-host-v2-ep16-bf16-small',16,64,8,False,'bf16',32,128,128),('p1-host-v2-ep16-shared-fp32-small',16,64,8,True,'fp32',32,128,128),('p1-host-v2-ep16-bf16-main',16,64,8,False,'bf16',8192,6144,2048),('p1-host-v2-ep16-shared-bf16-main',16,64,8,True,'bf16',8192,6144,2048)]
if index>1:
 prevtag,prevstage=sys.argv[3:5];a=json.loads(pathlib.Path('outputs','training_results_'+prevtag,'p1-stage-accuracy-audit.json').read_text());assert a['stage']==prevstage and a['complete'] and a['all_rank_accuracy_passed']
 ctl=json.loads(pathlib.Path('outputs','training_round_'+prevtag,prevstage+'-controller','summary.json').read_text())
 for h in ctl['launches']:
  machine='rack05os7' if '.27.' in h else 'rack05os8';r=json.loads(pathlib.Path('outputs','training_results_'+prevtag,prevstage,machine,'own-process-check.json').read_text());assert all(v.get('gone') for v in r['processes'])
stage,world,e,k,shared,dtype,b,h,i=configs[index-1];T='/home/zzy/deploy/megamoe-20261024/training'
args=[sys.executable,'work/run_p1_host_v2_round.py','--execute','--run-tag',tag,'--stage',stage,'--nnodes','1' if world==2 else '2','--nproc','2' if world==2 else '8','--limit','600','--script',T+'/candidate-p1-host-v2/p1_pointwise/validate_training_p1.py','--launcher',T+'/scripts/run_training_p1_host_v2_checked.sh','--','--run-npu','--training-source-root',T+'/qualified-p0-20261025/experiments/megamoe_training','--candidate-source-root',T+'/candidate-p1-host-v2/p1_pointwise','--pointwise-library',T+'/candidate-p1/build/libmegamoe_pointwise.so','--seed','4317','--oracle-chunk','1024','--max-reference-gib','12','--world-size',str(world),'--B',str(b),'--H',str(h),'--I',str(i),'--E',str(e),'--K',str(k),'--cases','balanced' if b==8192 else 'balanced,zero_gate,empty,skew','--weight-dtype',dtype]+(['--shared'] if shared else [])
raise SystemExit(subprocess.run(args).returncode)
