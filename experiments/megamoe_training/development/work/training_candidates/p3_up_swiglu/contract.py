"""Pure-stdlib native launch contract; no Torch, device access or padding."""
U32=(1<<32)-1
I64=(1<<63)-1
DEFAULT_CAP=12<<30

def product(*values):
    result=1
    for value in values:
        if type(value) is not int or value<0 or (value and result>I64//value):
            raise ValueError('Invalid/overflowing signed64 byte arithmetic')
        result*=value
    return result

def plan(rows,h,i,counts,cap_bytes=DEFAULT_CAP):
    if any(type(v) is not int for v in (rows,h,i,cap_bytes)):
        raise ValueError('Dimensions/cap must be Python integers')
    if not 0<=rows<=U32 or not 0<h<=1<<20 or not 0<i<=1<<20 or h%128 or i%128:
        raise ValueError('Require uint32 rows and positive native128 H/I, <=2**20')
    if not isinstance(counts,(tuple,list)) or len(counts)!=4:
        raise ValueError('Exactly four host expert counts required')
    if any(type(c) is not int or not 0<=c<=U32 for c in counts) or sum(counts)!=rows:
        raise ValueError('Counts must exactly cover expert-contiguous X')
    nt=(i+255)//256
    prefix=[0];row_prefix=[0]
    for c in counts:
        prefix.append(prefix[-1]+((c+127)//128)*nt)
        row_prefix.append(row_prefix[-1]+c)
    if prefix[-1]>U32:raise ValueError('Native job prefix uint32 overflow')
    sizes=dict(x=product(rows,h,2),w=product(4,2,i,h,2),gu=product(rows,2,i,2),
               a=product(rows,i,2),flags=product(prefix[-1],32,4))
    total=sum(sizes.values())
    if not 0<cap_bytes<=I64 or total>I64 or total>cap_bytes:
        raise ValueError('Input+output+privateflags byte budget exceeds cap')
    return dict(rows=rows,H=h,I=i,counts=list(counts),row_prefix=row_prefix,
                job_prefix=prefix,jobs=prefix[-1],bytes=sizes,total_bytes=total,
                cap_bytes=cap_bytes,required_AIC=32,required_AIV=64)

def jobs(p):
    """Independent schedule model with native job coordinates/ownership."""
    for e,c in enumerate(p['counts']):
        for r in range(0,c,128):
            for col in range(0,p['I'],256):
                j=p['job_prefix'][e]+(r//128)*((p['I']+255)//256)+col//256
                yield dict(job=j,expert=e,global_row=p['row_prefix'][e]+r,
                           rows=min(128,c-r),col=col,cols=min(256,p['I']-col),aic=j%32)
