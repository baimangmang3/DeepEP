"""Raw compute-only MIX launcher. No implicit autograd, fallback, cache or padding."""
import ctypes
from dataclasses import dataclass
from pathlib import Path
if __package__:
    from .contract import plan,DEFAULT_CAP
else:
    from contract import plan,DEFAULT_CAP

@dataclass
class Invocation:
    gu: object
    a: object
    flags: object
    contract: dict

class GroupedUpSwiGLU:
    def __init__(self,library,cap_bytes=DEFAULT_CAP):
        if type(cap_bytes) is not int or not 0<cap_bytes<(1<<63):raise ValueError('Invalid byte cap')
        self.path=str(Path(library).resolve(strict=True));self.cap_bytes=cap_bytes
        self.lib=ctypes.CDLL(self.path)
        self.fn=self.lib.megamoe_train_up_swiglu
        self.fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_uint32]*3+[
            ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint64,ctypes.c_void_p]
        self.fn.restype=ctypes.c_int
        self.lib.megamoe_train_up_swiglu_backend.restype=ctypes.c_char_p
        self.backend=self.lib.megamoe_train_up_swiglu_backend().decode()
        if self.backend!='experimental-CATLASS950-BF16-grouped-up-SwiGLU-v1':
            raise ValueError('Unexpected native backend')

    def __call__(self,x,w,counts):
        import torch
        # All basic checks precede empty/zeros, device context, stream getter or native call.
        if not isinstance(x,torch.Tensor) or not isinstance(w,torch.Tensor):raise ValueError('Tensor inputs required')
        if x.ndim!=2 or w.ndim!=3 or w.shape[0]!=4 or w.shape[1]%2 or x.shape[1]!=w.shape[2]:
            raise ValueError('Require X[M,H], canonical W[4,2I,H]')
        p=plan(x.shape[0],x.shape[1],w.shape[1]//2,counts,self.cap_bytes)
        if any(t.device.type!='npu' or t.dtype!=torch.bfloat16 or not t.is_contiguous() for t in (x,w)) or x.device!=w.device:
            raise ValueError('Contiguous BF16 X/W on the same NPU required')
        if x.requires_grad or w.requires_grad:raise ValueError('Raw component requires detached inputs; no autograd registration')
        # Reject shared storage even when visible views do not overlap.
        if x.untyped_storage().data_ptr()==w.untyped_storage().data_ptr():raise ValueError('Input alias rejected')
        gu=torch.empty((p['rows'],2*p['I']),device=x.device,dtype=torch.bfloat16)
        a=torch.empty((p['rows'],p['I']),device=x.device,dtype=torch.bfloat16)
        flags=torch.zeros((p['jobs'],32),device=x.device,dtype=torch.int32)
        result=Invocation(gu,a,flags,p)
        if not p['rows']:return result
        tensors=(x,w,gu,a,flags)
        pointers=[t.untyped_storage().data_ptr() for t in tensors]
        if len(set(pointers))!=len(pointers):raise RuntimeError('Invocation buffers alias')
        c=(ctypes.c_uint32*4)(*counts)
        with torch.npu.device(x.device):
            stream=torch.npu.current_stream(x.device) # Drains PTA submissions incl. zero flags.
            try:
                code=self.fn(*(t.data_ptr() for t in tensors),p['rows'],p['H'],p['I'],c,p['bytes']['flags'],stream.npu_stream)
                if code:raise RuntimeError(f'UpSwiGLU native rejection {code} (3 means physical32AIC/64AIV required)')
            finally:
                for t in tensors:t.record_stream(stream)
        return result
