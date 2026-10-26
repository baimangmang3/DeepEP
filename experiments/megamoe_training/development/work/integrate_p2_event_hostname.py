"""Finite reviewed wrapper integration; local source preparation only."""
import pathlib,shutil,json,hashlib,difflib,ast,subprocess
OLD=pathlib.Path('work/training_diagnostics/p2_ep2_serial_child_first_control')
NEW=OLD.with_name('p2_ep2_serial_hostname_control');NEW.mkdir(exist_ok=False)
OUT=pathlib.Path('outputs/p2_ep2_serial_hostname_preparation');OUT.mkdir(exist_ok=False)
for p in OLD.iterdir():
 if p.is_file():shutil.copyfile(p,NEW/p.name)
raw=pathlib.Path('work/torchrun_event_hostname_wrapper/launcher.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='7e15cc8ebaf480b09ab7d53185b9ad8077990a3a163adac7b7281cb4bf299e4d'
(NEW/'launcher.py').write_bytes(raw)
R='/home/zzy/deploy/megamoe-20261024/training/validation-p2-ep2-serial-hostname-20261026'
for name in ('controller.py','run_dispatch_host_trace_checked.sh'):
 s=(NEW/name).read_text().replace('validation-p2-ep2-serial-child-first-20261026','validation-p2-ep2-serial-hostname-20261026').replace('p2_ep2_serial_child_first_preparation','p2_ep2_serial_hostname_preparation').replace('p2_ep2_serial_child_first_control','p2_ep2_serial_hostname_control')
 if name.endswith('.sh'):s=s.replace('CMD=(python -m torch.distributed.run','CMD=(python "$DISPATCH_DIAGNOSTIC_ROOT/scripts/launcher.py"')
 (NEW/name).write_text(s,encoding='utf-8',newline='\n')
p=NEW/'cleanup_own.py';s=p.read_text().replace("VALIDATOR=str(T/",f"WRAPPER={R+'/scripts/launcher.py'!r}\nVALIDATOR=str(T/")
s=s.replace("('torch.distributed.run',VALIDATOR)","('torch.distributed.run',VALIDATOR,WRAPPER)");p.write_text(s,encoding='utf-8',newline='\n')
p=NEW/'cleanup_policy.py';s=p.read_text().replace('def external_launcher_evidence(rows,own_tag):','def external_launcher_evidence(rows,own_tag,known_wrappers=()):').replace("recognized=any(a in (", "recognized=any(a in known_wrappers or a in (")
p.write_text(s,encoding='utf-8',newline='\n')
p=NEW/'external_launcher_gate.py';s=p.read_text().replace("VISIBLE=(",f"WRAPPER={R+'/scripts/launcher.py'!r}\nVISIBLE=(").replace("],'__gate__')", ",known_wrappers=(WRAPPER,))" ) if False else p.read_text()
s=s.replace("VISIBLE=(",f"WRAPPER={R+'/scripts/launcher.py'!r}\nVISIBLE=(")
s=s.replace("],'__gate__')['busy_or_uncertain']", "],'__gate__',known_wrappers=(WRAPPER,))['busy_or_uncertain']").replace('external_launcher_evidence(rows,tag)', 'external_launcher_evidence(rows,tag,known_wrappers=(WRAPPER,))')
p.write_text(s,encoding='utf-8',newline='\n')
for p in NEW.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
r=subprocess.run(['C:/Program Files/Git/bin/bash.exe','-n',str(NEW/'run_dispatch_host_trace_checked.sh')],capture_output=True,text=True);assert r.returncode==0,r.stderr
patch=''
for name in ('controller.py','run_dispatch_host_trace_checked.sh','cleanup_own.py','cleanup_policy.py','external_launcher_gate.py'):
 patch+=''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True),(NEW/name).read_text().splitlines(True),fromfile='child-first/'+name,tofile='hostname/'+name))
(OUT/'minimal-wrapper-integration.patch').write_text(patch,encoding='utf-8')
print(json.dumps({'source':str(NEW),'deployed':False,'bash_n':0}))
