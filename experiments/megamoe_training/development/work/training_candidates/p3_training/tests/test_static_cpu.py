"""No Torch import: AST equivalence and real adapter constructor guards."""
import ast,hashlib,json,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run():
    path=ROOT/'training_adapter.py';tree=ast.parse(path.read_text())
    p1=ast.parse((ROOT.parent/'p1_pointwise/training_adapter.py').read_text())
    classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    routed=classes['_RoutedUpSwiGLU'];assert [ast.unparse(b) for b in routed.bases]==['_RoutedP1']
    assert all(not isinstance(n,ast.FunctionDef) or n.name!='backward' for n in routed.body)
    original=next(n for n in p1.body if isinstance(n,ast.ClassDef) and n.name=='_RoutedP1')
    def forward(c):return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    first,second=forward(original),forward(routed)
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,n):
            return ast.copy_location(ast.Attribute(value=ast.Name(id='engine',ctx=ast.Load()),attr='gemm',ctx=n.ctx),n) if n.id=='gemm' else n
    def filtered(fn,is_p3):
        body=[]
        for stmt in fn.body:
            if isinstance(stmt,ast.Assign):
                target=ast.unparse(stmt.targets[0])
                if (not is_p3 and target in ('gemm','gu','a')) or (is_p3 and target in ('invocation','(gu, a)')):continue
            body.append(Normalize().visit(stmt))
        return [ast.dump(n,include_attributes=False) for n in body]
    assert filtered(first,False)==filtered(second,True),'Forward differs outside approved fused GU/A substitution'
    checks=['inherits_exact_P1_backward','forward_AST_equal_outside_fused_GU_A']
    calls=[]
    class Parent:
        def __init__(self,**kw):calls.append(('parent',kw))
    class Routed:
        @staticmethod
        def backward(*args):return 'unchanged'
        @classmethod
        def apply(cls,*args):calls.append(('routed_apply',cls.__name__));return 5
    class Shared:
        @staticmethod
        def apply(*args):calls.append(('shared_apply',));return 7
    class Up:
        def __init__(self,*args):calls.append(('up_init',))
    world=[1]
    scope=dict(torch=types.SimpleNamespace(distributed=types.SimpleNamespace(get_world_size=lambda g:world[0])),
        PointwiseTrainingMoE=Parent,_RoutedP1=Routed,_SharedP1=Shared,GroupedUpSwiGLU=Up)
    tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(tree,str(path),'exec'),scope);cls=scope['UpSwiGLUTrainingMoE']
    assert scope['_RoutedUpSwiGLU'].backward is Routed.backward
    for w,e,c in ((1,3,32),(16,256,32),(1,4.,32),(1,4,16),(0,0,32)):
        world[0]=w;before=len(calls)
        try:cls(up_swiglu_library='not-loaded',num_experts=e,cores=c)
        except ValueError:pass
        else:raise AssertionError('Guard accepted invalid configuration')
        assert len(calls)==before;checks.append(f'guard_before_native_and_parent:{w}/{e}/{c}')
    for w in (1,2,16):
        world[0]=w;before=len(calls);owner=cls(up_swiglu_library='stub-only',num_experts=4*w)
        assert [v[0] for v in calls[before:]]==['up_init','parent'];checks.append(f'four_local_experts:{w}')
        assert owner(None,None,None,None,None)==5
        assert owner(None,None,None,None,None,1,2)==12
        before=len(calls)
        try:owner(None,None,None,None,None,1,None)
        except ValueError:pass
        else:raise AssertionError('Missing shared pair accepted')
        assert len(calls)==before;checks.append(f'shared_P1_apply_and_pair_guard:{w}')
    return dict(passed=True,checks=len(checks),details=checks,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='No Torch import; actual AST/constructor with metadata-only stubs, not native or numerical qualification')

if __name__=='__main__':
    result=run();Path(__file__).with_name('static_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
