"""Pure stdlib simulation; imports no torch/paramiko and contacts no host."""
import tempfile
import unittest
from threading import Lock
from training_native_round import run_round, HOSTS, EXPECTED_IDENTITY


class FakeBackend:
    def __init__(self, occupied=None, own=False, prior=False, fail_hash=False):
        self.events=[];self.lock=Lock();self.occupied=occupied;self.own=own;self.prior=prior;self.fail_hash=fail_hash
        self.client_identity={host:object() for host in HOSTS}
    def record(self, name, host=None):
        with self.lock:self.events.append((name,host,self.client_identity.get(host)))
    def connect(self):self.record('connect')
    def close(self):self.record('close')
    def verify(self, host, tag):
        self.record('hash',host)
        if self.fail_hash and host==HOSTS[1]:raise ValueError('changed source')
        return {'passed':True,'cached_content_digest':EXPECTED_IDENTITY}
    def gate(self, host, tag):
        self.record('gate',host);busy=host==self.occupied or self.own
        row={'pid':42,'own':self.own}
        return {'observation':{'all_eight_free':not busy,'npu_smi_rc':0,'time_utc':'test','free_cards':0 if busy else 8,'processes':[]},
                'owners':{'processes':[row] if busy else [],'own_device_processes':[row] if self.own else [],'foreign_device_processes':[row] if busy and not self.own else []},
                'prior_native':{'marker':self.prior,'rc':None,'passed':False}}
    def launch(self, host, tag):
        self.record('launch',host)
        return {'status':'native_supervisor_started','pid':123}


class ControllerTests(unittest.TestCase):
    def run_fake(self, **kwargs):
        fake=FakeBackend(**kwargs)
        with tempfile.TemporaryDirectory() as directory:result=run_round(fake,'cpu_mock_round',directory)
        return fake,result
    def test_all_hashes_before_gates_and_launch_uses_same_connection(self):
        fake,result=self.run_fake();names=[x[0] for x in fake.events]
        self.assertTrue(result['native_started']);self.assertLess(max(i for i,x in enumerate(names) if x=='hash'),min(i for i,x in enumerate(names) if x=='gate'))
        self.assertLess(max(i for i,x in enumerate(names) if x=='gate'),names.index('launch'))
        objects=[x[2] for x in fake.events if x[1]==HOSTS[0]]
        self.assertTrue(all(x is objects[0] for x in objects));self.assertEqual(names.count('connect'),1)
    def test_foreign_on_either_host_prevents_launch(self):
        for host in HOSTS:
            fake,result=self.run_fake(occupied=host)
            self.assertFalse(result['native_started']);self.assertFalse(any(x[0]=='launch' for x in fake.events))
    def test_own_residual_prevents_duplicate_launch(self):
        _,result=self.run_fake(own=True);self.assertEqual(result['status'],'occupied_or_uncertain_deferred')
    def test_existing_stage_prevents_duplicate_launch(self):
        _,result=self.run_fake(prior=True);self.assertEqual(result['status'],'prior_native_found_no_duplicate_launch')
    def test_hash_failure_never_reaches_gate_or_device(self):
        fake,result=self.run_fake(fail_hash=True)
        self.assertEqual(result['status'],'control_error_no_further_launch');self.assertFalse(any(x[0] in ('gate','launch') for x in fake.events))
    def test_invalid_tag_rejected_before_connect(self):
        fake=FakeBackend()
        with self.assertRaises(ValueError):run_round(fake,'bad; tag','unused')
        self.assertEqual(fake.events,[])


if __name__=='__main__':unittest.main()
