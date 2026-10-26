import argparse,sys,pathlib,json,hashlib

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]/'build/native-gemm-results.json')
    args=parser.parse_args()
    import torch,torch_npu
    root=pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    from megamoe_training.gemm import CatlassGemm
    torch.npu.set_device(0)
    torch.manual_seed(20261024)
    gemm=CatlassGemm(cores=32)
    results=[]
    for m,n,k in ((128,256,256),(17,384,128)):
     for ta,tb in ((False,False),(False,True),(True,False)):
      a=(torch.randn((k,m) if ta else (m,k))*0.1).bfloat16()
      b=(torch.randn((n,k) if tb else (k,n))*0.1).bfloat16()
      ref=(a.float().T if ta else a.float())@(b.float().T if tb else b.float())
      an,bn=a.npu(),b.npu()
      for fp32 in (False,True):
       out=gemm(an,bn,trans_a=ta,trans_b=tb,fp32=fp32)
       torch.npu.synchronize()
       got=out.float().cpu();target=ref if fp32 else ref.bfloat16().float()
       error=got-target
       rmse=float(error.square().mean().sqrt()/(target.square().mean().sqrt()+1e-12))
       max_abs=float(error.abs().max())
       passed=bool(torch.isfinite(got).all()) and rmse<(1e-4 if fp32 else 3e-3)
       row=dict(m=m,n=n,k=k,ta=ta,tb=tb,fp32=fp32,relative_rmse=rmse,max_abs_error=max_abs,passed=passed)
       results.append(row);print(json.dumps(row),flush=True)
       assert passed,row
    for ta,a_shape,b_shape in ((False,(17,0),(0,256)),(True,(0,128),(0,256))):
     for fp32 in (False,True):
      out=gemm(torch.empty(a_shape,dtype=torch.bfloat16,device='npu'),torch.empty(b_shape,dtype=torch.bfloat16,device='npu'),trans_a=ta,fp32=fp32)
      torch.npu.synchronize();assert bool((out==0).all()),'zeroK not zero'
      results.append(dict(zero_k=True,ta=ta,fp32=fp32,shape=list(out.shape),passed=True))
    report=dict(backend=gemm.backend,library=gemm.path,sha256=hashlib.sha256(pathlib.Path(gemm.path).read_bytes()).hexdigest(),cases=results,passed=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print('NATIVE_GEMM_PASS',len(results),flush=True)


if __name__=='__main__':
    main()
