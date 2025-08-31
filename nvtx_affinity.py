from tqdm import tqdm

from Universal_Attention.utils import *
from Universal_Attention.triton.affinity import _gen_affinity_scores_torch, _gen_affinity_scores_torch_fused
from Universal_Attention.triton.affinity import _gen_affinity_scores as _gen_affinity_scores_triton
# from Universal_Attention.triton.affinity_split import _gen_affinity_scores as _gen_affinity_scores_triton

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
        
    b, h, l, d = 8, 8, 4096, 128
    dtype = torch.bfloat16

    funcs = {
        'triton': _gen_affinity_scores_triton,
        'torch': _gen_affinity_scores_torch,
        'torch_fused': _gen_affinity_scores_torch_fused
    }
    
    print("Running NVTX profiler...")
    n = 5
    for _ in tqdm(range(n)):
        k, src, dest, aff_grad = gen_tensors(b, h, l, d, dtype, requires_grad=True)
        for name in funcs:
            fwd_time, bwd_time = nvtx_benchmark(funcs[name], k, src, dest, aff_grad=aff_grad)
        del k, src, dest
