import os,json,pathlib,paramiko
T='/home/zzy/deploy/megamoe-20261024/training';tag='20261025_0904';stage='p1-host-v2-ep16-bf16-small';run=T+'/runs/'+tag+'/'+stage
payload=f'''import os,signal,time,pathlib,json
+gid=1722184;run={run!r};r={{'target_pgid':gid,'signals':[],'before':[]}}
+def members():
+ rows=[]
+ for p in pathlib.Path('/proc').glob('[0-9]*/stat'):
+  try:
+   st=p.read_text();parts=st.rsplit(')',1)[1].split()
+   if int(parts[2])!=gid:continue
+   cmd=(p.parent/'cmdline').read_bytes().decode(errors='replace').replace('\\0',' ')
+   rows.append({{'pid':int(p.parent.name),'ppid':int(parts[1]),'pgid':int(parts[2]),'state':parts[0],'wchan':(p.parent/'wchan').read_text(),'cmdline':cmd}})
+  except (OSError,ValueError,IndexError):pass
+ return rows
+r['before']=members()
+if r['before']:
+ assert all(run in p['cmdline'] for p in r['before']),'Refuse unknown/foreign group member'
+ try:os.killpg(gid,signal.SIGTERM);r['signals'].append('TERM')
+ except ProcessLookupError:pass
+ deadline=time.monotonic()+30
+ while members() and time.monotonic()<deadline:time.sleep(1)
+ remain=members()
+ if remain:
+  assert all(run in p['cmdline'] for p in remain),'Refuse changed group ownership'
+  try:os.killpg(gid,signal.SIGKILL);r['signals'].append('KILL')
+  except ProcessLookupError:pass
+r['after']=members();print(json.dumps(r))
+'''.replace('\n+','\n')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('177.22.27.1',username='root',password=os.environ['DEPLOY_PASSWORD_27'],timeout=15)
_,o,e=c.exec_command("/root/miniconda3/envs/zzy/bin/python - <<'OWN_CLEANUP'\n"+payload+"\nOWN_CLEANUP\n",timeout=50);data=o.read().decode();error=e.read().decode();rc=o.channel.recv_exit_status();c.close();p=pathlib.Path('outputs/training_results_'+tag);(p/'own-controlled-cleanup.json').write_text(data if rc==0 else json.dumps({'rc':rc,'stdout':data,'stderr':error}));print(data,error)
