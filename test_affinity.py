from tqdm import tqdm

from Universal_Attention.utils import *
from Universal_Attention.triton.affinity import _gen_affinity_scores_torch, _gen_affinity_scores_torch_fused
from Universal_Attention.triton.affinity_new import _gen_affinity_scores as _gen_affinity_scores_triton
# from Universal_Attention.triton.affinity_split import _gen_affinity_scores as _gen_affinity_scores_triton
from Universal_Attention.triton.affinity_new import get_optimal_configs

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    profiler = get_profiler()
        
    b, h, l, d = 8, 8, 4096, 128

    dtype = torch.bfloat16

    funcs = {
        'triton': _gen_affinity_scores_triton,
        'torch': _gen_affinity_scores_torch,
        'torch_fused': _gen_affinity_scores_torch_fused
    }

    results = {}
    for name in funcs:
        results[name] = {
            'fwd_time': 0, 
            'bwd_time': 0, 
            'aff': None,
            'k.grad': None, 
            'src.grad': None,
            'dest.grad': None
        }

    k, src, dest, aff_grad = gen_tensors(b, h, l, d, dtype, requires_grad=True)

    print("Checking running time...")
    n = 5
    for _ in tqdm(range(n)):
        print('\n')
        k, src, dest, aff_grad = gen_tensors(b, h, l, d, dtype, requires_grad=True)
        for name in funcs:
            fwd_time, bwd_time = timed_benchmark(funcs[name], k, src, dest, aff_grad=aff_grad)
            results[name]['fwd_time'] += fwd_time
            results[name]['bwd_time'] += bwd_time
            print(f"{name} kernel time: (fwd) {fwd_time} ms, (bwd) {bwd_time} ms")
        del k, src, dest
    print("--- Summary ---")
    for name in funcs:
        print(f"{name} kernel time: (fwd) {results[name]['fwd_time']/n} ms, (bwd) {results[name]['bwd_time']/n} ms")

    get_optimal_configs()

    print("Checking closeness to ground truth...")

    k, src, dest, aff_grad = gen_tensors(b, h, l, d, dtype, requires_grad=True)

    for name in funcs:
        k_clone, src_clone, dest_clone, aff_grad_clone = clone_tensors(k, src, dest, aff_grad)
        aff = funcs[name](k_clone, src_clone, dest_clone)
        aff.backward(aff_grad_clone)
        
        results[name]['aff'] = aff
        results[name]['k.grad'] = k_clone.grad
        results[name]['src.grad'] = src_clone.grad
        results[name]['dest.grad'] = dest_clone.grad
        
        del k_clone, src_clone, dest_clone, aff, aff_grad_clone

    for attr in ['aff', 'k.grad', 'src.grad', 'dest.grad']:
        check_closeness(results['triton'][attr], results['torch_fused'][attr], attr, atol=1e-2, rtol=1e-2)
