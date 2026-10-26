"""CPU-only Python math/context regression, never NPU qualification.

Executes the actual training.py with only its NPU device comparison changed to
CPU in an AST copy. Communication and CATLASS are independent CPU stubs. This
does not validate native GEMM, transport, distributed ordering or performance.
"""
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


class CpuGemm:
    def __init__(self, *args):
        pass

    def __call__(self, a, b, *, trans_a=False, trans_b=False, fp32=False, out=None):
        af, bf = a.float(), b.float()
        value = (af.T if trans_a else af) @ (bf.T if trans_b else bf)
        value = value.to(torch.float32 if fp32 else torch.bfloat16)
        if out is not None:
            out.copy_(value)
            return out
        return value

    def grouped(self, x, weights, counts, *, transpose_weights=True):
        rows, offset = [], 0
        for weight, count in zip(weights, counts):
            rows.append(self(x[offset:offset+count], weight, trans_b=transpose_weights))
            offset += count
        assert offset == len(x)
        return torch.cat(rows)

    def weight_grad(self, dy, x, counts):
        rows, offset = [], 0
        for count in counts:
            rows.append(self(dy[offset:offset+count], x[offset:offset+count], trans_a=True, fp32=True))
            offset += count
        return torch.stack(rows)


class CpuBuffer:
    def __init__(self, **kwargs):
        pass

    def dispatch(self, x, *, handle=None, topk_idx=None, topk_weights=None, num_experts=None, **kwargs):
        if handle is None:
            b, k = topk_idx.shape
            order = torch.argsort(topk_idx.flatten(), stable=True)
            handle = types.SimpleNamespace(is_low_latency=False, b=b, k=k,
                order=order, tokens=order//k,
                num_recv_tokens_per_expert_list=torch.bincount(topk_idx.flatten(), minlength=num_experts).tolist())
            pe = topk_weights.flatten()[order]
        else:
            pe = None
        return x[handle.tokens].contiguous(), None, pe, handle, None

    def combine(self, xe, handle, *, topk_weights=None):
        value = torch.zeros((handle.b, xe.shape[1]), dtype=torch.float32)
        value.index_add_(0, handle.tokens, xe.float())
        dp = None
        if topk_weights is not None:
            flat = torch.empty(handle.b*handle.k, dtype=torch.float32)
            flat[handle.order] = topk_weights
            dp = flat.reshape(handle.b, handle.k)
        return value.to(torch.bfloat16), dp, None

    def destroy(self):
        pass


def load_actual_training():
    path = Path(__file__).resolve().parents[1]/'megamoe_training/training.py'
    tree = ast.parse(path.read_text())
    class CpuDeviceCheck(ast.NodeTransformer):
        count = 0
        def visit_Compare(self, node):
            if (ast.unparse(node.left) == 'x.device.type' and len(node.comparators) == 1
                    and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 'npu'):
                node.comparators[0].value = 'cpu'
                self.count += 1
            return self.generic_visit(node)
    rewrite = CpuDeviceCheck()
    tree = rewrite.visit(tree)
    assert rewrite.count == 1, 'Only the expected validator device check may change'
    ast.fix_missing_locations(tree)
    package = types.ModuleType('_cpu_actual_training')
    package.__path__ = []
    gemm_module = types.ModuleType('_cpu_actual_training.gemm')
    gemm_module.CatlassGemm = CpuGemm
    deep_ep = types.ModuleType('deep_ep')
    deep_ep.ElasticBuffer = CpuBuffer
    module = types.ModuleType('_cpu_actual_training.training')
    with patch.dict(sys.modules, {'_cpu_actual_training':package,
            '_cpu_actual_training.gemm':gemm_module, 'deep_ep':deep_ep}):
        exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


ACTUAL = load_actual_training()


def oracle(x, w1, w2, idx, p, dy, shared=None):
    """Independent expert/token selection plus explicit BF16 boundary formulas."""
    ysum = torch.zeros_like(x, dtype=torch.float32)
    dxsum = torch.zeros_like(x, dtype=torch.float32)
    dw1, dw2 = torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32)
    dp = torch.zeros_like(p)
    def expert(xe, first, second, de, pe=None):
        first, second = first.bfloat16().float(), second.bfloat16().float()
        gu = (xe.float() @ first.T).bfloat16()
        g, u = gu.float().chunk(2, -1)
        a = (torch.nn.functional.silu(g)*u).bfloat16()
        z = (a.float() @ second.T).bfloat16()
        dz = de if pe is None else (de.float()*pe[:, None]).bfloat16()
        dsecond = dz.float().T @ a.float()
        da = (dz.float() @ second).bfloat16().float()
        s = torch.sigmoid(g)
        dgu = torch.cat((da*u*s*(1+g*(1-s)), da*g*s), -1).bfloat16()
        dfirst = dgu.float().T @ xe.float()
        dx = (dgu.float() @ first).bfloat16()
        return z, dx, dfirst, dsecond
    for e in range(len(w1)):
        token, slot = torch.where(idx == e)
        z, dx, dw1[e], dw2[e] = expert(x[token], w1[e], w2[e], dy[token], p[token, slot])
        ysum.index_add_(0, token, (z.float()*p[token, slot, None]).bfloat16().float())
        dxsum.index_add_(0, token, dx.float())
        dp[token, slot] = (dy[token].float()*z.float()).sum(-1)
    y, dx = ysum.bfloat16(), dxsum.bfloat16()
    sdw = None
    if shared:
        sy, sx, sw1, sw2 = expert(x, *shared, dy)
        y, dx, sdw = y+sy, dx+sx, (sw1, sw2)
    return y, dx, dw1, dw2, dp, sdw


class TrainingMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(185)

    def fixture(self, dtype, idx=None):
        idx = torch.tensor([[0,2], [2,0], [0,2], [2,0]]) if idx is None else idx
        x = (torch.randn(4,4)*.5).bfloat16().requires_grad_()
        w1 = (torch.randn(4,6,4)*.3).to(dtype).requires_grad_()
        w2 = (torch.randn(4,4,3)*.3).to(dtype).requires_grad_()
        p = torch.rand(4,2); p[0,0] = 0; p.requires_grad_()
        dy = torch.randn(4,4).bfloat16()
        return x, w1, w2, idx, p, dy

    def engine(self):
        with patch.object(torch.distributed, 'get_world_size', return_value=1):
            return ACTUAL.TrainingMoE(group=None, max_tokens=4, hidden=4, topk=2, num_experts=4)

    def verify(self, engine, data, y, shared=None):
        x, w1, w2, idx, p, dy = data
        expected = oracle(x.detach(), w1.detach(), w2.detach(), idx, p.detach(), dy,
                          None if shared is None else tuple(t.detach() for t in shared))
        y.backward(dy)
        for actual, reference in zip((y, x.grad, *engine.last_grads_fp32, p.grad), expected[:5]):
            torch.testing.assert_close(actual.float(), reference.float(), atol=1e-6, rtol=1e-6)
        for leaf, reference in zip((w1,w2), expected[2:4]):
            torch.testing.assert_close(leaf.grad, reference.to(leaf.dtype), atol=0, rtol=0)
        self.assertNotEqual(p.grad[0,0].item(), 0, 'p=0 must retain nonzero dp')
        if shared:
            for leaf, reference, saved in zip(shared, expected[5], engine.last_shared_grads_fp32):
                torch.testing.assert_close(saved, reference, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(leaf.grad, reference.to(leaf.dtype), atol=0, rtol=0)

    def test_zero_gate_empty_experts_bf16(self):
        engine, data = self.engine(), self.fixture(torch.bfloat16)
        self.verify(engine, data, engine(*data[:5]))
        self.assertEqual(engine.last_counts, [4,0,4,0])
        self.assertEqual(torch.count_nonzero(engine.last_grads_fp32[0][[1,3]]).item(), 0)

    def test_fp32_master_and_shared(self):
        engine, data = self.engine(), self.fixture(torch.float32)
        shared = tuple((torch.randn(*shape)*.3).requires_grad_() for shape in ((6,4),(4,3)))
        self.verify(engine, data, engine(*data[:5], shared_w1=shared[0], shared_w2=shared[1]), shared)
        self.assertEqual(data[1].grad.dtype, torch.float32)

    def test_two_routes_reverse_backward_context(self):
        engine = self.engine()
        first = self.fixture(torch.float32)
        second = self.fixture(torch.float32, torch.tensor([[3,1],[1,3],[3,1],[1,3]]))
        y1, y2 = engine(*first[:5]), engine(*second[:5])
        self.verify(engine, second, y2)
        self.verify(engine, first, y1)

    def test_closed_engine_backward_rejected(self):
        engine, data = self.engine(), self.fixture(torch.bfloat16)
        y = engine(*data[:5]); engine.close()
        with self.assertRaisesRegex(RuntimeError, 'before backward'):
            y.backward(data[-1])


if __name__ == '__main__':
    unittest.main()
