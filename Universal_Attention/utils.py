import torch
import torch.cuda.nvtx as nvtx

def check_closeness(triton_tensor, pytorch_tensor, name, atol=1e-2, rtol=1e-2):
    valid_mask = torch.abs(pytorch_tensor) < 1e8
    triton_valid = triton_tensor[valid_mask]
    pytorch_valid = pytorch_tensor[valid_mask]
    
    abs_diff = torch.abs(triton_valid - pytorch_valid)
    tolerance_bound = atol + rtol * torch.abs(pytorch_valid)
    within_tolerance = abs_diff <= tolerance_bound
    
    percent_within_tolerance = (torch.sum(within_tolerance).float() / triton_valid.numel()) * 100
    max_abs_diff = torch.max(abs_diff).item()
    max_rel_diff = torch.max(abs_diff / (torch.abs(pytorch_valid) + 1e-12)).item() # Add epsilon for stability

    print(f"--- Closeness Check for: {name} ---")
    print(f"  - Percentage of elements within tolerance ({atol=}, {rtol=}): {percent_within_tolerance:.4f}%")
    print(f"  - Maximum Absolute Difference: {max_abs_diff:.6f}")
    print(f"  - Maximum Relative Difference: {max_rel_diff:.6f}")
    if percent_within_tolerance < 100.0:
        print(f"  - ⚠️ Not all elements are within the tolerance.")
    else:
        print(f"  - ✅ All elements are within the tolerance.")
    print("-" * (29 + len(name)))

def get_profiler():
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("/gpfs/hshen/profile_traces"),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )
    return profiler

def gen_tensors(b, h, l, d, dtype=torch.bfloat16, requires_grad=True):
    k = torch.rand((b, h, l, d), device='cuda', dtype=dtype)
    k /= k.pow(2).sum(-1, True).sqrt().add(1e-6)
    src = torch.rand((b, h, l), device='cuda', dtype=dtype).sigmoid()
    dest = torch.rand((b, h, l), device='cuda', dtype=dtype).sigmoid()
    
    if requires_grad:
        k.requires_grad_(True)
        src.requires_grad_(True)
        dest.requires_grad_(True)
        aff_grad = torch.rand((b, h, l, l), device='cuda', dtype=dtype)
        aff_grad /= aff_grad.pow(2).sum(-1, True).sqrt().add(1e-6)
    else:
        aff_grad = None
        
    return k, src, dest, aff_grad

def clone_tensors(k, src, dest, aff_grad=None):
    k_clone = k.clone().detach().requires_grad_(True)
    src_clone = src.clone().detach().requires_grad_(True)
    dest_clone = dest.clone().detach().requires_grad_(True)
    
    if aff_grad is not None:
        aff_grad_clone = aff_grad.clone().detach()
    else:
        aff_grad_clone = None
    return k_clone, src_clone, dest_clone, aff_grad_clone

def timed_benchmark(func, k, src, dest, aff_grad=None):
    k_clone = k.clone().detach().requires_grad_(True)
    src_clone = src.clone().detach().requires_grad_(True)
    dest_clone = dest.clone().detach().requires_grad_(True)

    for _ in range(5):
        aff = func(k_clone, src_clone, dest_clone)

    torch.cuda.synchronize()
    fwd_start = torch.cuda.Event(True)
    fwd_end = torch.cuda.Event(True)
    fwd_start.record()
    for _ in range(10):
        aff = func(k_clone, src_clone, dest_clone)
    fwd_end.record()
    torch.cuda.synchronize()

    fwd_time = fwd_start.elapsed_time(fwd_end)/10

    if aff_grad is not None:
        aff_grad_clone = aff_grad.clone().detach()
        k_clone.retain_grad()
        src_clone.retain_grad()
        dest_clone.retain_grad()

        for _ in range(5):
            aff.backward(aff_grad_clone, retain_graph=True)

        torch.cuda.synchronize()
        bwd_start = torch.cuda.Event(True)
        bwd_end = torch.cuda.Event(True)
        bwd_start.record()
        for _ in range(10):
            aff.backward(aff_grad_clone, retain_graph=True)
        bwd_end.record()
        torch.cuda.synchronize()

        bwd_time = bwd_start.elapsed_time(bwd_end)/10
        del aff_grad_clone
    else:
        bwd_time = 0
    
    del aff, k_clone, src_clone, dest_clone
    return fwd_time, bwd_time
    
def nvtx_benchmark(func, k, src, dest, aff_grad=None):
    k_clone = k.clone().detach().requires_grad_(True)
    src_clone = src.clone().detach().requires_grad_(True)
    dest_clone = dest.clone().detach().requires_grad_(True)

    for _ in range(5):
        aff = func(k_clone, src_clone, dest_clone)

    torch.cuda.synchronize()
    with torch.cuda.nvtx.range(f"{func.__name__} (fwd)"):
        for _ in range(10):
            aff = func(k_clone, src_clone, dest_clone)
    torch.cuda.synchronize()

    if aff_grad is not None:
        aff_grad_clone = aff_grad.clone().detach()
        k_clone.retain_grad()
        src_clone.retain_grad()
        dest_clone.retain_grad()

        for _ in range(5):
            aff.backward(aff_grad_clone, retain_graph=True)

        torch.cuda.synchronize()
        with torch.cuda.nvtx.range(f"{func.__name__} (bwd)"):
            for _ in range(10):
                aff.backward(aff_grad_clone, retain_graph=True)
        torch.cuda.synchronize()
        
        del aff_grad_clone
    del aff, k_clone, src_clone, dest_clone
    