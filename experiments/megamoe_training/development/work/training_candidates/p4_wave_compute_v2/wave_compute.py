"""Raw P4 GU/A/Z component: no autograd, fallback, gate or communication."""
import ctypes
from dataclasses import dataclass
from pathlib import Path
if __package__:from .contract import plan,DEFAULT_CAP
else:from contract import plan,DEFAULT_CAP

@dataclass
class Invocation:
    gu:object
    a:object
    z:object
    flags:object
    contract:dict

class WaveCompute:
    def __init__(self,library,cap_bytes=DEFAULT_CAP):
        if type(cap_bytes) is not int or not 0<cap_bytes<(1<<63):raise ValueError('Invalid byte cap')
        self.path=str(Path(library).resolve(strict=True));self.cap_bytes=cap_bytes
        self.lib=ctypes.CDLL(self.path);self.fn=self.lib.megamoe_train_wave_compute
        self.fn.argtypes=[ctypes.c_void_p]*7+[ctypes.c_uint32]*3+[ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint64,ctypes.c_void_p]
        self.fn.restype=ctypes.c_int;self.lib.megamoe_train_wave_compute_backend.restype=ctypes.c_char_p
        if self.lib.megamoe_train_wave_compute_backend().decode()!='experimental-CATLASS950-BF16-512wave-GU-A-Z-v1':raise ValueError('Wrong P4 backend')

    def __call__(self,x,w1,w2,counts):
        import torch
        if any(not isinstance(t,torch.Tensor) for t in (x,w1,w2)):raise ValueError('Tensor inputs required')
        if x.ndim!=2 or w1.ndim!=3 or w2.ndim!=3 or w1.shape[0]!=4 or w2.shape[0]!=4 or w1.shape[1]!=2*w2.shape[2] or w1.shape[2]!=x.shape[1] or w2.shape[1]!=x.shape[1]:raise ValueError('Canonical X[M,H],W1[4,2I,H],W2[4,H,I] required')
        p=plan(x.shape[0],x.shape[1],w2.shape[2],counts,self.cap_bytes)
        if any(t.device.type!='npu' or t.device!=x.device or t.dtype!=torch.bfloat16 or not t.is_contiguous() or t.requires_grad for t in (x,w1,w2)):raise ValueError('Detached contiguous BF16 tensors on same NPU required')
        if len({t.untyped_storage().data_ptr() for t in (x,w1,w2)})!=3:raise ValueError('Aliased inputs rejected')
        gu=torch.empty((p['rows'],2*p['I']),device=x.device,dtype=torch.bfloat16)
        a=torch.empty((p['rows'],p['I']),device=x.device,dtype=torch.bfloat16)
        z=torch.empty((p['rows'],p['H']),device=x.device,dtype=torch.bfloat16)
        # Fresh private flags every call. No slot/epoch reuse or cross-call cache.
        flags=torch.zeros((2*p['jobs'],32),device=x.device,dtype=torch.int32)
        result=Invocation(gu,a,z,flags,p)
        if not p['rows']:return result # host-empty path, not hardware qualification
        tensors=(x,w1,w2,gu,a,z,flags)
        if len({t.untyped_storage().data_ptr() for t in tensors})!=7:raise RuntimeError('Invocation storage alias')
        counts_c=(ctypes.c_uint32*4)(*counts)
        with torch.npu.device(x.device):
            stream=torch.npu.current_stream(x.device)
            try:
                rc=self.fn(*(t.data_ptr() for t in tensors),p['rows'],p['H'],p['I'],counts_c,p['bytes']['flags'],stream.npu_stream)
                if rc:raise RuntimeError(f'P4 native rejection {rc};3 requires physical32AIC/64AIV')
            finally:
                for t in tensors:t.record_stream(stream)
        return result
