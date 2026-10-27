"""Standalone training pointwise candidate; no implicit fallback or autograd registration."""
import ctypes
from pathlib import Path
import torch
import torch_npu  # noqa: F401


class TrainingPointwise:
    def __init__(self, library, vector_cores=64):
        if type(vector_cores) is not int or not 1 <= vector_cores <= 64:
            raise ValueError('Require 1..64 configured AIV cores on the qualified Ascend950')
        self.path = str(Path(library).resolve(strict=True))
        self.lib = ctypes.CDLL(self.path)
        self.fn = self.lib.megamoe_train_pointwise
        self.fn.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*4 + [ctypes.c_uint32]*2 + [ctypes.c_void_p, ctypes.c_uint32]
        self.fn.restype = ctypes.c_int
        self.vector_cores = vector_cores

    def _run(self, op, x, extra=None):
        if (x.device.type != 'npu' or x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous()):
            raise ValueError('Require contiguous BF16 matrix on NPU')
        rows, width = x.shape
        if op in (0,1):
            if width % 2:
                raise ValueError('GU has two equal halves')
            width //= 2
        if not 0 <= rows < 2**32 or not 0 < width <= 2**20 or width % 128:
            raise ValueError('Native width must be a positive multiple of 128; no padding is performed')
        if extra is not None:
            shape, dtype = ((rows,width),torch.bfloat16) if op==1 else ((rows,),torch.float32)
            if extra.shape != shape or extra.dtype != dtype or extra.device != x.device or not extra.is_contiguous():
                raise ValueError('Incorrect dA/gate shape, dtype, device or layout')
        elif op in (1,2):
            raise ValueError('Missing dA/gate')
        out = torch.empty((rows,width*2 if op==1 else width),device=x.device,dtype=torch.bfloat16)
        if not rows:
            return out
        with torch.npu.device(x.device):
            stream = torch.npu.current_stream(x.device)
            # PTA current-stream getter drains its host submission queue before this raw launch.
            code = self.fn(op,x.data_ptr(),extra.data_ptr() if op==1 else None,
                           extra.data_ptr() if op==2 else None,out.data_ptr(),rows,width,
                           stream.npu_stream,self.vector_cores)
            if code:
                raise RuntimeError(f'Native pointwise launch rejected: {code}')
            for tensor in (x,extra,out):
                if tensor is not None:
                    tensor.record_stream(stream)
        return out

    def activation(self, gu):
        return self._run(0,gu)

    def activation_grad(self, da, gu):
        return self._run(1,gu,da)

    def gate(self, z, p):
        return self._run(2,z,p)
