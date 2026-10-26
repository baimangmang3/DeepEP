import os,paramiko,pathlib
c=paramiko.SSHClient();c.load_system_host_keys();c.connect(os.environ['DEPLOY_HOST'],username='root',password=os.environ['DEPLOY_PASSWORD'],timeout=20)
d='/home/zzy/deploy/megamoe-20261024/training/DeepEP/experiments/megamoe_training'
_,o,e=c.exec_command(f'mkdir -p {d}/native {d}/megamoe_training');assert o.channel.recv_exit_status()==0
s=c.open_sftp()
for name in ('CMakeLists.txt','native/gemm.cpp','megamoe_training/gemm.py'):
 p=pathlib.Path('work/DeepEP/experiments/megamoe_training')/name
 s.put(str(p),d+'/'+name+'.tmp');s.posix_rename(d+'/'+name+'.tmp',d+'/'+name)
s.close();c.close();print('Native source snapshot uploaded')
