import pathlib,json,paramiko,os,time
pid=1763046;base=pathlib.Path('outputs/training_results_20261025_0933/pure-hccl-control-once/rack05os7');r0=json.loads((base/'own-process-start-snapshot.json').read_text());old=next(x for x in r0['processes'] if x['pid']==pid);start=old['stat'].rsplit(')',1)[1].split()[19]
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('177.22.27.1',username='root',password=os.environ['DEPLOY_PASSWORD_27'],timeout=15)
code=f'''import pathlib,json,time
pid={pid};p=pathlib.Path('/proc')/str(pid);r={{'pid':pid,'time_ns':time.time_ns()}}
try:
 stat=(p/'stat').read_text();cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode();assert stat.rsplit(')',1)[1].split()[19]=={start!r} and '/runs/20261025_0933/pure-hccl-control-once' in cmd
 r['verified_argv']=cmd;r['stat']=stat;r['wchan']=(p/'wchan').read_text()
 try:r['kernel_stack']=(p/'stack').read_text()
 except OSError as e:r['kernel_stack_error']=repr(e)
 children=(p/'task'/str(pid)/'children').read_text().split();r['direct_children']=[]
 for n in children:
  q=pathlib.Path('/proc')/n;v={{'pid':int(n)}}
  for f in ('stat','wchan','cmdline'):
   try:v[f]=(q/f).read_bytes().decode(errors='replace').replace('\\0',' ')
   except OSError as e:v[f+'_error']=repr(e)
  r['direct_children'].append(v)
except OSError:r['gone']=True
print(json.dumps(r))'''
_,o,e=c.exec_command("timeout 20 /root/miniconda3/envs/zzy/bin/python - <<'PY'\n"+code+"\nPY",timeout=30);r=json.loads(o.read());assert o.channel.recv_exit_status()==0,e.read().decode();c.close();(base/'torchrun-exit-observation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
