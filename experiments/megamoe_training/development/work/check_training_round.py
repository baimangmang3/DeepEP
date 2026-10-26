"""One read-only device snapshot per host; credentials only from environment."""
import os,paramiko,pathlib,json,sys,hashlib
host,round_tag=sys.argv[1:3];do_quick='--quick' in sys.argv;machine='rack05os7' if host=='177.22.27.1' else 'rack05os8'
t='/home/zzy/deploy/megamoe-20261024/training'
c=paramiko.SSHClient();c.load_system_host_keys();c.connect(host,username='root',password=os.environ['DEPLOY_PASSWORD'],timeout=15)
quick=f'RUN_TAG={round_tag} timeout 90 /root/miniconda3/envs/zzy/bin/python {t}/scripts/quick_training_identity.py > {t}/logs/quick-{machine}.stdout\n' if do_quick else ''
command=f'''set -e
{quick}
timeout 45 /root/miniconda3/envs/zzy/bin/python {t}/scripts/training_occupancy_gate.py > {t}/logs/gate-{machine}.stdout
timeout 30 /root/miniconda3/envs/zzy/bin/python {t}/scripts/training_npu_process_owners.py > {t}/logs/gate-{machine}-owners.stdout
'''
_,out,err=c.exec_command(command,timeout=100);out.read();error=err.read().decode();rc=out.channel.recv_exit_status();assert rc==0,(rc,error)
s=c.open_sftp();remote=t+'/runs/20261025_0300/preflight'
name=sorted(x for x in s.listdir(remote) if x.startswith(machine+'-') and x.endswith('.txt'))[-1]
prefix=name[:-4];dest=pathlib.Path('outputs')/('training_occupancy_'+round_tag);dest.mkdir(parents=True,exist_ok=True)
for suffix in ('.txt','.json','-owners.json'):s.get(remote+'/'+prefix+suffix,str(dest/(prefix+suffix)))
if do_quick:s.get(t+'/runs/'+round_tag+'/preflight/'+machine+'-quick-identity.json',str(dest/(machine+'-quick-identity.json')))
s.close();c.close()
obs=json.loads((dest/(prefix+'.json')).read_text());own=json.loads((dest/(prefix+'-owners.json')).read_text())
rows=own['processes'];first=rows[0] if rows else {}
report=dict(host=machine,time_utc=obs['time_utc'],free_cards=obs['free_cards'],all_eight_free=obs['all_eight_free'],npu_smi_rc=obs['npu_smi_rc'],npu_pids=[r['pid'] for r in rows],own_device_processes=own['own_device_processes'],foreign_count=len(own['foreign_device_processes']),existing_test_processes=obs['processes'],existing_rank_results=obs['existing_rank_results'],parent_pid=first.get('ppid'),cwd=first.get('cwd'),parent_args=first.get('parent_args'),evidence_prefix=str(dest/prefix))
(dest/(machine+'-summary.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,ensure_ascii=False))
