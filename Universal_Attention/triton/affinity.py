import torch
import triton
import triton.language as tl

import numpy as np
import inspect
import time

configs = [
    triton.Config({'BLOCK_I': BLOCK_I, 'BLOCK_J': BLOCK_J, 'BLOCK_D': BLOCK_D}, num_stages=stages, num_warps=warps) \
    for BLOCK_I in [16, 32, 64, 128]\
    for BLOCK_J in [16, 32, 64, 128]\
    for BLOCK_D in [16, 32, 64, 128]\
    for stages in [1, 2, 3]\
    for warps in [2, 4, 8]\
]

'''
######################################
#     Forward Kernel & Interface     #
######################################
'''
@triton.autotune(
    configs=configs,
    key=['l', 'd'],
)
@triton.jit
def _affinity_fwd_kernel(
    # Pointers to matrices
    k, src, dest, aff, 
    # Matrix dimensions                                                 
    B, H, L, D,
    # Strides
    str_k_b, str_k_h, str_k_l, str_k_d,             # b h l d
    str_src_b, str_src_h, str_src_li,               # b h l
    str_dest_b, str_dest_h, str_dest_lj,            # b h l
    str_aff_b, str_aff_h, str_aff_li, str_aff_lj,   # b h l l
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_D: tl.constexpr, 
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)

    k_ptr = k + pid_b * str_k_b + pid_h * str_k_h
    aff_ptr = aff + pid_b * str_aff_b + pid_h * str_aff_h
    dest_ptr = dest + pid_b * str_dest_b + pid_h * str_dest_h

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    src_vec = tl.load(
        src + pid_b * str_src_b + pid_h * str_src_h + offs_i * str_src_li,
        mask=(offs_i < L),
        other=0.0
    ).cast(src_vec, tl.float32)
    src_mat = tl.exp2(tl.log2(src_vec) / 2.0)[:, None]

    prev_sum = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for j_offset in range(0, L, BLOCK_J):
        offs_j = j_offset + tl.arange(0, BLOCK_J)

        dest_vec = tl.load(
            dest_ptr + offs_j * str_dest_lj,
            mask=(offs_j < L),
            other=0.0
        ).cast(dest_vec, tl.float32)
        dest_mat = tl.exp2(tl.log2(dest_vec) / 2.0)[:, None]

        affinity = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

        for d_offset in range(0, D, BLOCK_D):
            offs_d = d_offset + tl.arange(0, BLOCK_D)
            k_mat = tl.load(
                k_ptr + offs_i[:, None] * str_k_l + offs_d[None, :] * str_k_d, 
                mask=(offs_i[:, None] < L) & (offs_d[None, :] < D), 
                other=0.0
            ).cast(k_mat, tl.float32)

            kt_mat = tl.load(
                k_ptr + offs_j[:, None] * str_k_l + offs_d[None, :] * str_k_d, 
                mask=(offs_j[:, None] < L) & (offs_d[None, :] < D), 
                other=0.0
            ).cast(kt_mat, tl.float32)

            # Use ieee to use fp32, otherwise the default would be tf32
            affinity += tl.dot(k_mat*src_mat, tl.trans(kt_mat*dest_mat), input_precision="ieee")

        # .relu().pow(2/3), already in fp32
        affinity = tl.exp2(tl.log2(tl.maximum(affinity, 0.0)) * 2.0 / 3.0)

        # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
        affinity = tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - 1e-6)) 

        # .triu(1)
        affinity = tl.where((offs_i[:, None] < offs_j[None, :]), affinity, 0.0)

        # .cumsum(3)
        curr_sum = tl.sum(affinity, axis=1, keep_dims=False) 
        affinity = tl.cumsum(affinity, axis=1) + prev_sum[:, None]  
        prev_sum += curr_sum
        
        # .masked_fill(mask.tril(-1), -1e12)
        affinity = tl.where((offs_i[:, None] > offs_j[None, :]), -1e12, affinity).cast(affinity, DTYPE)

        tl.store(
            aff_ptr + offs_i[:, None] * str_aff_li + offs_j[None, :] * str_aff_lj, 
            affinity, 
            mask=(offs_i[:, None] < L) & (offs_j[None, :] < L)
        )

def _affinity_fwd(k, src, dest):
    '''
    Inputs:
    k:    b h l d
    src:  b h l
    dest: b h l

    Outputs:
    aff:  b h l l
    '''    
    b,h,l,d = k.shape
    assert src.shape == (b,h,l) and dest.shape == (b,h,l)

    dtype = k.dtype
    device = k.device
    DTYPE_FLAG = tl.bfloat16 if dtype == torch.bfloat16 else tl.float32

    aff = torch.empty(b,h,l,l, dtype=dtype, device=device)

    grid = lambda META: (b, h, triton.cdiv(l, META['BLOCK_I']))

    _affinity_fwd_kernel[grid](
        k, src, dest, aff,                                                  
        b, h, l, d, 
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        src.stride(0), src.stride(1), src.stride(2), 
        dest.stride(0), dest.stride(1), dest.stride(2), 
        aff.stride(0), aff.stride(1), aff.stride(2), aff.stride(3), 
        DTYPE=DTYPE_FLAG, 
    )

    return aff.transpose(-1, -2)

'''
#######################################
#     Backward Kernel & Interface     #
#######################################
'''
@triton.autotune(
    configs=configs,
    key=['l', 'd'],
)
@triton.jit
def _affinity_bwd_kernel(
    # Pointers to matrices
    k, src, dest, daff, dk, dsrc, ddest,                                                                                              
    # Matrix dimensions                                                 
    B, H, L, D, 
    # Strides
    str_k_b, str_k_h, str_k_l, str_k_d,                 # b h l d
    str_src_b, str_src_h, str_src_li,                   # b h l
    str_dest_b, str_dest_h, str_dest_lj,                # b h l
    str_daff_b, str_daff_h, str_daff_li, str_daff_lj,   # b h l l
    str_dk_b, str_dk_h, str_dk_l, str_dk_d,             # b h l d
    str_dsrc_b, str_dsrc_h, str_dsrc_li,                # b h l
    str_ddest_b, str_ddest_h, str_ddest_lj,             # b h l
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_D: tl.constexpr, 
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)

    k_ptr = k + pid_b * str_k_b + pid_h * str_k_h
    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h
    dest_ptr = dest + pid_b * str_dest_b + pid_h * str_dest_h
    daff_ptr = daff + pid_b * str_daff_b + pid_h * str_daff_h
    dk_ptr = dk + pid_b * str_dk_b + pid_h * str_dk_h
    dsrc_ptr = dsrc + pid_b * str_dsrc_b + pid_h * str_dsrc_h
    ddest_ptr = ddest + pid_b * str_ddest_b + pid_h * str_ddest_h

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    for j_offset in range(0, L, BLOCK_J):
        offs_j = j_offset + tl.arange(0, BLOCK_J)


def _affinity_bwd(k, src, dest, daff):
    '''
    Inputs:
    k:     b h l d
    src:   b h l
    dest:  b h l
    daff:  b h l l

    Outputs:
    dk:    b h l d
    dsrc:  b h l
    ddest: b h l
    '''    
    b,h,l,d = k.shape
    assert src.shape == (b,h,l) and dest.shape == (b,h,l) and daff.shape == (b,h,l,l)

    daff = daff.transpose(-1, -2).contiguous() # Transpose it back

    dtype = k.dtype
    DTYPE_FLAG = tl.bfloat16 if dtype == torch.bfloat16 else tl.float32

    dk = torch.empty_like(k)
    dsrc = torch.empty_like(src)
    ddest = torch.empty_like(dest)

    grid = lambda META: (b, h, triton.cdiv(l, META['BLOCK_I']))

    _affinity_bwd_kernel[grid](
        k, src, dest, daff, dk, dsrc, ddest,                                                 
        b, h, l, d, 
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        src.stride(0), src.stride(1), src.stride(2), 
        dest.stride(0), dest.stride(1), dest.stride(2), 
        daff.stride(0), daff.stride(1), daff.stride(2), daff.stride(3), 
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dsrc.stride(0), dsrc.stride(1), dsrc.stride(2), 
        ddest.stride(0), ddest.stride(1), ddest.stride(2), 
        DTYPE=DTYPE_FLAG, 
    )

    return dk, dsrc, ddest

'''
#################################
#     Ground Truth Autograd     #
#################################
'''
def _gen_affinity_scores_pytorch(k, src, dest):
    kkt = torch.einsum('bnqh, bnkh -> bnqk', k, k).relu().pow(2/3).float()
    affinity = kkt * src.pow(1/3).unsqueeze(-1) * dest.pow(1/3).unsqueeze(-2)
    affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
    affinity = affinity.triu(1).cumsum(3)
    return torch.transpose(affinity.masked_fill(torch.ones_like(affinity, dtype=torch.bool).tril(-1), -1e12), -1, -2).contiguous()

def _gen_affinity_scores_pytorch_fused(k, src, dest):
    affinity = torch.einsum('bnqh, bnkh -> bnqk', k*src.sqrt().unsqueeze(-1), k*dest.sqrt().unsqueeze(-1)).relu().float().pow(2/3)
    affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
    affinity = affinity.triu(1).cumsum(3).to(dtype=k.dtype)
    return torch.transpose(affinity.masked_fill(torch.ones_like(affinity, dtype=torch.bool).tril(-1), -1e12), -1, -2).contiguous()

if __name__ == "__main__":
    b, h, l, d = 8, 8, 4096, 128

    dtype = torch.bfloat16

    test = "forward"
    # test = "backward"

    if test == "forward":
        print("Testing forward pass")
        k = torch.rand((b, h, l, d), device='cuda', dtype=dtype)
        src = torch.rand((b, h, l), device='cuda', dtype=dtype)
        dest = torch.rand((b, h, l), device='cuda', dtype=dtype)

        warm_up = 10
        for _ in range(warm_up):
            _ = _affinity_fwd(k, src, dest)
            _ = _gen_affinity_scores_pytorch(k, src, dest)
            _ = _gen_affinity_scores_pytorch_fused(k, src, dest)

        print("Checking running time...")
        n = 500
        triton_time, torch_time, torch_fused_time = 0, 0, 0
        for _ in range(n):
            k = torch.rand((b, h, l, d), device='cuda', dtype=dtype)
            src = torch.rand((b, h, l), device='cuda', dtype=dtype)
            dest = torch.rand((b, h, l), device='cuda', dtype=dtype)

            start_time = time.time()
            _ = _affinity_fwd(k, src, dest)
            triton_time += time.time() - start_time

            start_time = time.time()
            _ = _gen_affinity_scores_pytorch(k, src, dest)
            torch_time += time.time() - start_time

            start_time = time.time()
            _ = _gen_affinity_scores_pytorch_fused(k, src, dest)
            torch_fused_time += time.time() - start_time

            del k, src, dest

        print(f"Triton kernel time: {triton_time}s")
        print(f"Pytorch kernel time: {torch_time}s")
        print(f"Pytorch fused kernel time: {torch_time}s")

        print("Checking closeness to ground truth...")

        k = torch.rand((b, h, l, d), device='cuda', dtype=dtype)
        src = torch.rand((b, h, l), device='cuda', dtype=dtype)
        dest = torch.rand((b, h, l), device='cuda', dtype=dtype)

        aff_triton = _affinity_fwd(k, src, dest)
        aff_pytorch = _gen_affinity_scores_pytorch(k, src, dest)
        aff_pytorch_fused = _gen_affinity_scores_pytorch_fused(k, src, dest)

        torch.testing.assert_close(aff_triton, aff_pytorch, atol=1e-3, rtol=1e-5)
        torch.testing.assert_close(aff_triton, aff_pytorch_fused, atol=1e-3, rtol=1e-5)
            