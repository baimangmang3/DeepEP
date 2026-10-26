import os,pathlib,json,torch,torch_npu
d=pathlib.Path('/home/zzy/deploy/megamoe-20261024/training/logs')
keys=('alltoall','all_to_all','grouped_matmul','grouped_mat_mul')
schemas=[]
for s in torch._C._jit_get_all_schemas():
 text=str(s)
 if any(k in text.lower() for k in keys):schemas.append(text)
functions={}
for n in dir(torch_npu):
 if any(k in n.lower() for k in keys):
  f=getattr(torch_npu,n)
  functions[n]=getattr(f,'__doc__',None)
dispatch={}
for op in ('npu::npu_alltoallv_gmm','npu::npu_gmm_alltoallv'):
 try:dispatch[op]=torch._C._dispatch_dump(op)
 except Exception as e:dispatch[op]=str(e)
result=dict(torch=torch.__version__,torch_npu=torch_npu.__version__,schemas=sorted(schemas),functions=functions,dispatch=dispatch)
(d/'torch-native-apis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
