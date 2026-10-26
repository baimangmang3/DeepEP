import os,pathlib,json,ctypes,socket
import torch,torch_npu
torch.npu.set_device(0)
path=pathlib.Path(os.environ['ASCEND_HOME_PATH'])/'lib64/libascendcl.so'
lib=ctypes.CDLL(str(path))
fn=lib.aclrtGetDeviceInfo;fn.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.POINTER(ctypes.c_int64)];fn.restype=ctypes.c_int
result={'host':socket.gethostname(),'lib':str(path.resolve()),'device':0,'attributes':{}}
for name,attr in [('AICORE_CORE_NUM',101),('CUBE_CORE_NUM',102)]:
 value=ctypes.c_int64();rc=fn(0,attr,ctypes.byref(value));result['attributes'][name]={'enum':attr,'rc':rc,'value':value.value}
cube=result['attributes']['CUBE_CORE_NUM'];assert cube['rc']==0 and cube['value']>=32,result
out=pathlib.Path('/home/zzy/deploy/megamoe-20261024/training/runs')/os.environ['RUN_TAG']/'preflight';out.mkdir(parents=True,exist_ok=True)
(out/(socket.gethostname()+'-core-info.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
