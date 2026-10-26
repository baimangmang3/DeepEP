"""Stdlib CPU regression: python -m unittest discover -s tests -p test_provenance.py.

No torch, NPU, SSH, SDK, native library loading or dependency installation.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from provenance import (REQUIRED_ROLES,build_identity,canonical_digest,
                        require_training_qualification,unique_loaded,validate_identity,collective_identity,
                        ensure_required_cann_mapped)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.files={}
        for index,role in enumerate(sorted(REQUIRED_ROLES)):
            path=self.root/'hostA'/f'content{index}.bin'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(('fixture '+role).encode())
            self.files[role]=path
        self.versions={'torch':'test-torch','torch_npu':'test-pta'}
        self.precision={'matmul_allow_hf32':False,'allow_internal_format':False}
        self.identity=self.current()
        self.shape=dict(B=32,H=128,I=128,E=8,K=4)
        self.reports=self.root/'reports';self.reports.mkdir()
        self.records=[dict(rank=rank,world=2,backend='shmem_catlass',shape=self.shape,
            all_passed=True,weight_dtype='torch.bfloat16',identity=copy.deepcopy(self.identity),
            cases=[dict(case='balanced',all_ranks_numerical_pass=True)]) for rank in range(2)]
        self.save()

    def current(self):
        return build_identity(self.files,self.versions,self.precision)

    def save(self):
        for rank,record in enumerate(self.records):
            (self.reports/f'rank{rank}.json').write_text(json.dumps(record))

    def qualify(self,identity=None):
        return require_training_qualification(self.reports,2,self.shape,identity or self.identity)

    def test_matching_content_passes(self):
        self.assertEqual(len(self.qualify()),2)

    def test_different_host_paths_same_contents_pass(self):
        other={}
        for role,path in self.files.items():
            target=self.root/'different-host-prefix'/path.name
            target.parent.mkdir(exist_ok=True);target.write_bytes(path.read_bytes())
            other[role]=target
        identity=build_identity(other,self.versions,self.precision)
        self.assertNotEqual(identity['diagnostic_paths'],self.identity['diagnostic_paths'])
        self.assertEqual(identity['content_digest'],self.identity['content_digest'])
        self.assertEqual(len(self.qualify(identity)),2)

    def test_changed_source_or_each_native_library_rejected(self):
        for role in sorted(REQUIRED_ROLES):
            with self.subTest(role=role):
                path=self.files[role];original=path.read_bytes()
                path.write_bytes(original+b'CHANGED')
                with self.assertRaises(ValueError):self.qualify(self.current())
                path.write_bytes(original)

    def test_missing_or_tampered_identity_rejected(self):
        variants=[None,{},copy.deepcopy(self.identity)]
        variants[-1]['content']['files'][0]['sha256']='0'*64
        for variant in variants:
            self.records[1]['identity']=variant;self.save()
            with self.assertRaises(ValueError):self.qualify()

    def test_missing_role_rejected_even_if_digest_recomputed(self):
        identity=copy.deepcopy(self.identity)
        identity['content']['files'].pop()
        identity['content_digest']=canonical_digest(identity['content'])
        with self.assertRaises(ValueError):validate_identity(identity)

    def test_missing_file_and_framework_version_rejected(self):
        self.files['native.catlass'].unlink()
        with self.assertRaises(FileNotFoundError):self.current()
        identity=copy.deepcopy(self.identity)
        identity['content']['versions'].pop('torch_npu')
        identity['content_digest']=canonical_digest(identity['content'])
        with self.assertRaises(ValueError):validate_identity(identity)

    def test_changed_framework_version_rejected(self):
        self.versions['torch_npu']='new-pta'
        with self.assertRaises(ValueError):self.qualify(self.current())

    def test_missing_or_changed_precision_rejected(self):
        for precision in ({},{'matmul_allow_hf32':True,'allow_internal_format':False}):
            with self.assertRaises(ValueError):build_identity(self.files,self.versions,precision)

    def test_backend_shape_pass_and_rank_checks_remain_enforced(self):
        for field,value in [('backend','cann_ai_cpu'),('backend',None),('shape',dict(self.shape,B=64)),
                            ('all_passed',False),('rank',0),('weight_dtype','torch.float32')]:
            with self.subTest(field=field,value=value):
                original=self.records[1][field];self.records[1][field]=value;self.save()
                with self.assertRaises(ValueError):self.qualify()
                self.records[1][field]=original;self.save()
        self.records[1]['cases'][0]['all_ranks_numerical_pass']=False;self.save()
        with self.assertRaises(ValueError):self.qualify()

    def test_missing_report_and_directory_rejected(self):
        (self.reports/'rank1.json').unlink()
        with self.assertRaises(ValueError):self.qualify()
        with self.assertRaises(ValueError):require_training_qualification(None,2,self.shape,self.identity)

    def test_ambiguous_actual_library_contents_rejected(self):
        first=self.root/'a/libshmem.so';second=self.root/'b/libshmem.so'
        first.parent.mkdir();second.parent.mkdir();first.write_bytes(b'one');second.write_bytes(b'two')
        with self.assertRaises(ValueError):unique_loaded({first,second},lambda n:n=='libshmem.so','SHMEM')

    def test_collection_errors_are_exchanged_before_local_or_peer_failure(self):
        for local_failure in (True,False):
            with self.subTest(local_failure=local_failure):
                events=[]
                class Status:
                    def item(self):events.append('read_global_status');return 0
                def tensor(*args,**kwargs):events.append('local_status');return Status()
                def all_reduce(*args,**kwargs):events.append('all_reduce')
                fake_torch=SimpleNamespace(tensor=tensor,int32='int32',npu=SimpleNamespace(current_device=lambda:0))
                fake_dist=SimpleNamespace(all_reduce=all_reduce,ReduceOp=SimpleNamespace(MIN='min'))
                kwargs={'side_effect':ValueError('identity missing')} if local_failure else {'return_value':self.identity}
                with patch('provenance.collect_runtime_identity',**kwargs):
                    with self.assertRaises(ValueError):collective_identity(fake_torch,fake_dist)
                self.assertEqual(events,['local_status','all_reduce','read_global_status'])

    def formal_cann_fixture(self):
        root=self.root/'formal-cann';(root/'lib64').mkdir(parents=True)
        paths={}
        for name in ('libascendcl.so','libruntime.so','libopapi.so'):
            path=root/'lib64'/name;path.write_bytes(name.encode())
            paths[name]=path.resolve()
        return root,paths

    def test_lazy_cann_load_requires_formal_file_and_mapping(self):
        root,paths=self.formal_cann_fixture()
        before={paths['libascendcl.so'],paths['libruntime.so']}
        after=set(paths.values())
        with patch.dict('os.environ',{'ASCEND_HOME_PATH':str(root)}),patch('ctypes.CDLL') as loader:
            with patch('provenance.mapped_libraries',return_value=after):
                mapped,loads=ensure_required_cann_mapped(before)
            self.assertEqual(mapped,after);self.assertEqual(len(loads),1)
            self.assertEqual(loader.call_args.args,(str(paths['libopapi.so']),))
            loader.reset_mock()
            mapped,loads=ensure_required_cann_mapped(after)
            self.assertEqual(loads,[]);loader.assert_not_called()

    def test_lazy_cann_missing_environment_or_formal_file_rejected(self):
        root,paths=self.formal_cann_fixture()
        before={paths['libascendcl.so'],paths['libruntime.so']}
        with patch.dict('os.environ',{},clear=True),patch('ctypes.CDLL') as loader:
            with self.assertRaises(ValueError):ensure_required_cann_mapped(before)
            loader.assert_not_called()
        paths['libopapi.so'].unlink()
        with patch.dict('os.environ',{'ASCEND_HOME_PATH':str(root)}),patch('ctypes.CDLL') as loader:
            with self.assertRaises(FileNotFoundError):ensure_required_cann_mapped(before)
            loader.assert_not_called()

    def test_lazy_cann_load_without_actual_mapping_rejected(self):
        root,paths=self.formal_cann_fixture()
        before={paths['libascendcl.so'],paths['libruntime.so']}
        with patch.dict('os.environ',{'ASCEND_HOME_PATH':str(root)}),patch('ctypes.CDLL'):
            with patch('provenance.mapped_libraries',return_value=before):
                with self.assertRaises(ValueError):ensure_required_cann_mapped(before)


if __name__=='__main__':
    unittest.main()
