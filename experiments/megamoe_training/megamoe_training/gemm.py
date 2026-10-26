"""Raw CATLASS launcher: no torch.mm or HCCL fallback in the implementation."""
import ctypes
import os
from pathlib import Path
import torch
import torch_npu  # noqa: F401


class CatlassGemm:
    def __init__(self, library=None, cores=32):
        path = Path(library or os.environ.get("MEGAMOE_TRAINING_LIBRARY", ""))
        if not library and not os.environ.get("MEGAMOE_TRAINING_LIBRARY"):
            path = Path(__file__).resolve().parents[1] / "build/libmegamoe_training.so"
        self.path = str(path.resolve())
        self.lib = ctypes.CDLL(self.path)
        self.fn = self.lib.megamoe_train_gemm
        self.fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32] * 3 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        self.fn.restype = ctypes.c_int
        self.lib.megamoe_train_backend.restype = ctypes.c_char_p
        self.backend = self.lib.megamoe_train_backend().decode()
        self.cores = cores

    def __call__(self, a, b, *, trans_a=False, trans_b=False, fp32=False, out=None):
        if (a.device.type != "npu" or a.device != b.device or
                a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or
                a.ndim != 2 or b.ndim != 2 or not a.is_contiguous() or not b.is_contiguous()):
            raise ValueError("CATLASS expects contiguous BF16 matrices on the same NPU")
        m, k = a.shape[::-1] if trans_a else a.shape
        kb, n = b.shape[::-1] if trans_b else b.shape
        if k != kb:
            raise ValueError(f"GEMM K mismatch: {k} != {kb}")
        dtype = torch.float32 if fp32 else torch.bfloat16
        if out is None:
            out = torch.empty((m, n), dtype=dtype, device=a.device)
        if out.shape != (m, n) or out.dtype != dtype or not out.is_contiguous() or out.device != a.device:
            raise ValueError("Invalid CATLASS output")
        if m == 0 or n == 0:
            return out
        if k == 0:
            return out.zero_()
        with torch.npu.device(a.device):
            stream = torch.npu.current_stream(a.device)
            code = self.fn(a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k,
                           int(trans_a), int(trans_b), int(fp32), stream.npu_stream, self.cores)
            if code:
                raise RuntimeError(f"CATLASS launch failed: {code}")
            # Raw C launches bypass PTA's allocator stream bookkeeping.
            for t in (a, b, out):
                t.record_stream(stream)
        return out

    def grouped(self, x, weights, counts, *, transpose_weights=True):
        n = weights.shape[1] if transpose_weights else weights.shape[2]
        out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
        start = 0
        for e, count in enumerate(counts):
            end = start + count
            self(x[start:end], weights[e], trans_b=transpose_weights, out=out[start:end])
            start = end
        if start != x.shape[0]:
            raise ValueError("Grouped counts do not cover the input")
        return out

    def weight_grad(self, dy, x, counts):
        out = torch.empty((len(counts), dy.shape[1], x.shape[1]), device=x.device, dtype=torch.float32)
        start = 0
        for e, count in enumerate(counts):
            end = start + count
            self(dy[start:end], x[start:end], trans_a=True, fp32=True, out=out[e])
            start = end
        if start != x.shape[0]:
            raise ValueError("Grouped counts do not cover the gradient")
        return out
