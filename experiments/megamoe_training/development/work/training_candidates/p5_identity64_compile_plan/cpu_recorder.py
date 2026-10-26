"""Bounded Linux CPU child-first recorder; inert until run_recorded() is called."""
import json,os,pathlib,signal,subprocess,time,traceback

class LinuxOps:
    clock=staticmethod(time.monotonic)
    sleep=staticmethod(time.sleep)
    def spawn(self,argv,cwd,stdout,stderr):
        return subprocess.Popen(argv,cwd=cwd,stdout=stdout,stderr=stderr,start_new_session=True)
    def identity(self,pid):
        p=pathlib.Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start=s[19],uid=p.stat().st_uid,ppid=int(s[1]),session=int(s[3]),state=s[0],argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v])
    def scan(self,root,owned):
        rows=[];errors=[]
        for p in pathlib.Path('/proc').glob('[0-9]*'):
            try:
                row=self.identity(int(p.name))
                rows.append(row)
            except FileNotFoundError:continue
            except OSError as exc:errors.append(dict(pid=int(p.name),error=repr(exc)))
        # Revalidate recorded births before using a PPID as ancestry authority.
        births={(v['pid'],v['start'],v['uid']) for v in list(owned.values())+[root]}
        proven={v['pid'] for v in rows if (v['pid'],v['start'],v['uid']) in births}
        session_active=any(v['pid'] in proven and v['session']==root['session'] for v in rows)
        if not session_active:
            for row in rows:
                if row['session']==root['session'] and row['uid']==root['uid']:
                    errors.append(dict(pid=row['pid'],error='session ID has no still-verified original birth; ownership unknown'))
        found={}
        changed=True
        while changed:
            changed=False
            for row in rows:
                if row['uid']==root['uid'] and (session_active and row['session']==root['session'] or row['ppid'] in proven or (row['pid'],row['start'],row['uid']) in births):
                    found[row['pid']]=row
                    if row['pid'] not in proven:proven.add(row['pid']);changed=True
        return list(found.values()),errors
    def current(self,saved):
        try:r=self.identity(saved['pid'])
        except FileNotFoundError:return None
        if any(r[k]!=saved[k] for k in ('pid','start','uid')):return None
        return r
    def signal(self,saved,sig):
        r=self.current(saved)
        if r is not None and r['state'] not in ('Z','X'):os.kill(saved['pid'],sig);return True
        return False

def run_recorded(argv,cwd,stdout_path,stderr_path,status_path,*,limit=300,term_seconds=5,kill_seconds=5,reap_seconds=5,ops=None):
    ops=ops or LinuxOps();r=dict(argv=argv,cwd=str(cwd),returncode=None,timeout=False,owned=[],cleanup_errors=[],signals=[],released=False,passed=False)
    p=None;root=None;owned={};uncertain=False
    def first(message):r.setdefault('first_error',message)
    def save():pathlib.Path(status_path).write_bytes((json.dumps(r,indent=2)+'\n').encode())
    def safe_save():
        try:save()
        except BaseException as exc:
            first(repr(exc));r['passed']=False;r.setdefault('report_io_errors',[]).append(repr(exc))
    def observe():
        nonlocal uncertain
        live=[]
        try:
            rows,errors=ops.scan(root,owned)
            if errors:uncertain=True;r['cleanup_errors']+=errors;first('ownership_observation_error: '+repr(errors[0]))
            for row in rows:owned[(row['pid'],row['start'],row['uid'])]=row
        except BaseException as exc:
            uncertain=True;first(repr(exc));r['cleanup_errors'].append(dict(stage='scan',error=repr(exc)))
        r['owned']=list(owned.values())
        for v in owned.values():
            try:
                current=ops.current(v)
                if current is not None and current['state'] not in ('Z','X'):live.append(current)
            except BaseException as exc:
                uncertain=True;first(repr(exc));r['cleanup_errors'].append(dict(pid=v['pid'],stage='current',error=repr(exc)))
        return live
    def emit(saved,sig):
        record=dict(pid=saved['pid'],start=saved['start'],uid=saved['uid'],signal=int(sig),time_ns=time.time_ns(),monotonic=ops.clock())
        r['signals'].append(record)
        try:record['sent']=ops.signal(saved,sig)
        except BaseException as exc:record['error']=repr(exc);r['cleanup_errors'].append(record.copy())
        safe_save()
    def cleanup():
        nonlocal uncertain
        if root is None:
            uncertain=True;r['cleanup_errors'].append('driver birth/descendant ownership unknown; preserve driver');return
        # All body errors, timeouts and early driver exits use this SAME path.
        for sig,seconds in ((signal.SIGTERM,term_seconds),(getattr(signal,'SIGKILL',9),kill_seconds)):
            deadline=ops.clock()+seconds
            while True:
                children=[v for v in observe() if v['pid']!=root['pid']]
                parents={v['pid']:v['ppid'] for v in owned.values()}
                def depth(row):
                    pid=row['pid'];seen=set()
                    while pid in parents and pid not in seen:seen.add(pid);pid=parents[pid]
                    return len(seen)
                children.sort(key=depth,reverse=True)
                for child in children:emit(child,sig) # ops revalidates exact birth
                if not children or ops.clock()>=deadline:break
                ops.sleep(.05)
        remaining=observe()
        if uncertain or any(v['pid']!=root['pid'] for v in remaining):
            r['cleanup_errors'].append('driver preserved: descendant termination/ownership unconfirmed');return
        for sig,seconds in ((signal.SIGTERM,term_seconds),(getattr(signal,'SIGKILL',9),kill_seconds)):
            if p.poll() is not None:break
            emit(root,sig)
            try:p.wait(timeout=seconds)
            except subprocess.TimeoutExpired:pass
    try:
        with pathlib.Path(stdout_path).open('xb') as out,pathlib.Path(stderr_path).open('xb') as err:
            p=ops.spawn(argv,cwd,out,err);r['owned_child_pid']=p.pid
            captured=ops.identity(p.pid);r['driver_identity']=captured
            if captured['session']!=p.pid:raise RuntimeError('Popen did not establish unique CPU session')
            root=captured;r['driver_identity_verified']=True
            owned[(root['pid'],root['start'],root['uid'])]=root;save()
            end=ops.clock()+limit
            while p.poll() is None:
                observe();save()
                if uncertain:first('ownership_observation_error');break
                if ops.clock()>=end:r['timeout']=True;first('tool_timeout');break
                ops.sleep(.05)
            live=observe()
            if live and p.poll() is not None:first('driver_exited_with_live_descendants')
    except BaseException as exc:
        first(repr(exc));r['error_traceback']=traceback.format_exc()
    finally:
        if p is not None:
            try:
                if root is None or p.poll() is None or observe():cleanup()
            except BaseException as exc:
                uncertain=True;r['cleanup_errors'].append(dict(stage='cleanup',error=repr(exc)))
            try:r['returncode']=p.wait(timeout=reap_seconds)
            except BaseException as exc:
                first('driver_not_reaped');r['cleanup_errors'].append(dict(stage='waitpid',error=repr(exc)))
            if root is not None:r['remaining']=observe()
            else:r['remaining']='unknown'
            r['ownership_unknown']=uncertain or root is None
            r['released']=r['returncode'] is not None and r['remaining']==[] and not r['ownership_unknown']
        r['passed']=r['returncode']==0 and r['released'] and not r['timeout'] and not r.get('first_error') and not r['cleanup_errors']
        safe_save()
    return r
