"""CPU stdlib tests: no torch, native libraries, NPU or environment access."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from gate import FIXED_LIMIT,NAMES,expected_counts,qualify


def identity(value):
    if not isinstance(value,dict) or not value.get('content_digest'):raise ValueError('Missing identity')
    return value['content_digest']


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name);self.shape=dict(B=4,H=8,I=4,E=4,K=2)
        self.limits={name:dict(FIXED_LIMIT) for name in NAMES};self.reports=[]
        for rank in range(2):
            counts=expected_counts(self.shape,2)[rank*2:(rank+1)*2];recv=sum(counts)
            sizes=dict(Y=32,dx=32,dp=8,dW1=128,dW2=64,leaf_dW1=128,leaf_dW2=64)
            metrics={name:dict(passed=True,finite=True,elements=size,limits=dict(FIXED_LIMIT),
                actual_dtype='torch.float32' if name in ('dp','dW1','dW2') else 'torch.bfloat16',
                out_of_tolerance_elements=0,RMS_error=0.,reference_RMS=1.,RMS_relative_error=0.,
                max_abs_error=0.,max_reference_abs=1.) for name,size in sizes.items()}
            shapes=dict(x=[4,8],xe=[recv,8],w1=[2,8,8],gu=[recv,8],a=[recv,4],w2=[2,8,4],y=[4,8],z=[recv,8])
            case=dict(case='balanced',all_ranks_numerical_pass=True,counts_match=True,oracle_counts=counts,
                      candidate_counts=counts,candidate_shapes=shapes,metrics=metrics)
            self.reports.append(dict(rank=rank,world=2,shape=self.shape,seed=4317,shared=False,
                backend='shmem_catlass',weight_dtype='torch.bfloat16',all_passed=True,limits=self.limits,
                identity=dict(content_digest='current'),cases=[case]))
        self.save()

    def save(self):
        for rank,record in enumerate(self.reports):(self.directory/f'rank{rank}.json').write_text(json.dumps(record))

    def run_gate(self,backend='shmem_catlass',shared=False):
        return qualify(self.directory,backend,2,self.shape,4317,shared,dict(content_digest='current'),self.limits,identity)

    def test_valid_full_reports(self):self.assertEqual(len(self.run_gate()),2)

    def test_contract_changes_rejected(self):
        for key,value in [('seed',9),('shared',True),('backend','cann_ai_cpu'),('weight_dtype','torch.float32'),
                          ('world',16),('all_passed',False),('identity',None),('identity',dict(content_digest='stale'))]:
            with self.subTest(key=key,value=value):
                old=self.reports[1][key];self.reports[1][key]=value;self.save()
                with self.assertRaises(ValueError):self.run_gate()
                self.reports[1][key]=old;self.save()

    def test_missing_rank_and_forward_only_reports_rejected(self):
        (self.directory/'rank1.json').unlink()
        with self.assertRaises(FileNotFoundError):self.run_gate()
        self.save();self.reports[1]['cases'][0]['metrics'].pop('dW1');self.save()
        with self.assertRaises(ValueError):self.run_gate()

    def test_counts_shapes_and_incomplete_metrics_rejected(self):
        original=copy.deepcopy(self.reports[1]['cases'][0])
        for kind in ('counts','shapes','elements','finite','limit','pass','RMS','NaN'):
            case=copy.deepcopy(original)
            if kind=='counts':case['candidate_counts'][0]+=1
            elif kind=='shapes':case['candidate_shapes']['xe'][0]+=1
            elif kind=='elements':case['metrics']['dx']['elements']=1
            elif kind=='finite':case['metrics']['dp']['finite']=False
            elif kind=='limit':case['metrics']['Y']['limits']['atol']=1
            elif kind=='RMS':case['metrics']['Y']['RMS_relative_error']=.1
            elif kind=='NaN':case['metrics']['Y']['max_abs_error']=float('nan')
            else:case['all_ranks_numerical_pass']=False
            self.reports[1]['cases'][0]=case;self.save()
            with self.assertRaises(ValueError):self.run_gate()

    def test_cann_routed_report_accepted_but_cann_shared_rejected(self):
        for report in self.reports:
            report['backend']='cann_ai_cpu';shapes=report['cases'][0]['candidate_shapes']
            shapes.pop('z');shapes['packed_z']=[8,8]
        self.save();self.assertEqual(len(self.run_gate('cann_ai_cpu')),2)
        with self.assertRaises(ValueError):self.run_gate('cann_ai_cpu',True)

    def test_shared_requires_every_shared_gradient(self):
        for report in self.reports:report['shared']=True
        self.save()
        with self.assertRaises(ValueError):self.run_gate(shared=True)

    def test_matching_shared_full_gradients_accepted(self):
        for report in self.reports:
            report['shared']=True;metrics=report['cases'][0]['metrics']
            for name,size in [('dSW1',64),('dSW2',32),('leaf_dSW1',64),('leaf_dSW2',32)]:
                item=copy.deepcopy(metrics['dW1']);item['elements']=size
                item['actual_dtype']='torch.bfloat16' if name.startswith('leaf_') else 'torch.float32'
                metrics[name]=item
        self.save();self.assertEqual(len(self.run_gate(shared=True)),2)

    def test_full_train_step_rejects_no_grad_before_work(self):
        tree=ast.parse(Path(__file__).with_name('probe.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step')
        events=[]
        y=SimpleNamespace(backward=lambda dy:events.append(('backward',dy)))
        ns=dict(torch=SimpleNamespace(is_grad_enabled=lambda:False),
                validation=SimpleNamespace(clear_grads=lambda ts:events.append('clear_grads')),
                x=1,w1=2,w2=3,ids=4,p=5,dy=6,shared=(),engines={'shmem_catlass':lambda *a,**kw:y})
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<full-train-step>','exec'),ns)
        with self.assertRaises(RuntimeError):ns['step']('shmem_catlass')
        self.assertEqual(events,[])
        ns['torch'].is_grad_enabled=lambda:True
        self.assertIs(ns['step']('shmem_catlass'),y)
        self.assertEqual(events,['clear_grads',('backward',6)])


if __name__=='__main__':unittest.main()
