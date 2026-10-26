"""Independent stdlib P4 ABI and schedule model; no padding or device access."""
U32=(1<<32)-1
I64=(1<<63)-1
DEFAULT_CAP=12<<30
def product(*values):
    result=1
    for v in values:
        if type(v) is not int or v<0 or (v and result>I64//v):raise ValueError('checked64 product overflow/type')
        result*=v
    return result

def plan(rows,h,i,counts,cap_bytes=DEFAULT_CAP):
    if any(type(v) is not int for v in (rows,h,i,cap_bytes)):raise ValueError('Integer dimensions/cap required')
    if not 0<=rows<=U32 or not 0<h<=1<<20 or not 0<i<=1<<20 or h%128 or i%128:raise ValueError('Native128 H/I and uint32 rows required')
    if not isinstance(counts,(list,tuple)) or len(counts)!=4 or any(type(v) is not int or not 0<=v<=U32 for v in counts) or sum(counts)!=rows:raise ValueError('Exactly4 host counts covering X required')
    row_prefix=[0];job_prefix=[0];nt=(i+255)//256
    for c in counts:
        row_prefix.append(row_prefix[-1]+c);job_prefix.append(job_prefix[-1]+((c+127)//128)*nt)
    if job_prefix[-1]>U32:raise ValueError('Pair prefix uint32 overflow')
    sizes=dict(x=product(rows,h,2),w1=product(4,2,i,h,2),w2=product(4,h,i,2),
               gu=product(rows,2,i,2),a=product(rows,i,2),z=product(rows,h,2),flags=product(2,job_prefix[-1],32,4))
    total=sum(sizes.values())
    if not 0<cap_bytes<=I64 or total>I64 or total>cap_bytes:raise ValueError('Component live-byte cap exceeded')
    return dict(candidate='P4_compute_wave512_GU_A_Z_v1_UNQUALIFIED',rows=rows,H=h,I=i,counts=list(counts),
        row_prefix=row_prefix,job_prefix=job_prefix,jobs=job_prefix[-1],wave_rows=512,
        bytes=sizes,total_bytes=total,cap_bytes=cap_bytes,required_AIC=32,required_AIV=64,
        activation_subcore=0,idle_subcore=1,flag_channels=['GU_ready','activation_ready'],host_empty=rows==0)

def waves(p):
    ni=(p['I']+255)//256;nh=(p['H']+255)//256
    for e,c in enumerate(p['counts']):
        for start in range(0,c,512):
            wr=min(512,c-start);mt=(wr+127)//128;base=p['job_prefix'][e]+start//128*ni
            up=[];down=[]
            for j in range(mt*ni):
                mr,n=divmod(j,ni);up.append(dict(job=base+j,expert=e,mtile=(start//128)+mr,
                    row=p['row_prefix'][e]+start+mr*128,col=n*256,rows=min(128,wr-mr*128),cols=min(256,p['I']-n*256),core=j%32))
            for t in range(mt*nh):
                mr,n=divmod(t,nh);down.append(dict(expert=e,mtile=(start//128)+mr,
                    row=p['row_prefix'][e]+start+mr*128,col=n*256,rows=min(128,wr-mr*128),cols=min(256,p['H']-n*256),core=t%32,
                    wait_jobs=[base+mr*ni+j for j in range(ni)]))
            yield dict(expert=e,start=start,rows=wr,up=up,down=down)
