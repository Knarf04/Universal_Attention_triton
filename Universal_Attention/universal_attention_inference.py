import math
import torch
import triton
import triton.language as tl

@triton.jit
def ua_logits_kernel(
    k_ptr,          # float32 [H, L, D]
    k_new_ptr,      # float32 [H, D]
    q_ptr,          # float32 [H, R, D]
    r_ptr,          # float32 [H, L]
    static_dest_ptr,# float32 [H]
    a_ptr,          # float32 [H, L] (updated in-place)
    qk_ptr,         # float32 [H, L, R]
    qk_new_ptr,     # float32 [H, R]

    H: tl.constexpr,
    L: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
):
    h = tl.program_id(0)  # one program per head

    # Base offsets for this head
    k_head_off       = h * L * D
    q_head_off       = h * R * D
    r_head_off       = h * L
    a_head_off       = h * L
    qk_head_off      = h * L * R
    qk_new_head_off  = h * R

    static_dest = tl.load(static_dest_ptr + h)

    # -----------------------------------------------------
    # Loop over sequence positions: compute kk, qk, decay, a
    # -----------------------------------------------------
    for l in range(0, L):
        # kk = dot(k[h, l, :], k_new[h, :])
        kk = 0.0

        # rates / affs at position l
        r_val = tl.load(r_ptr + r_head_off + l)
        a_old = tl.load(a_ptr + a_head_off + l)

        # clear qk[h, l, r] for all r (we will accumulate over D)
        for r_idx in range(0, R):
            qk_index = qk_head_off + l * R + r_idx
            tl.store(qk_ptr + qk_index, 0.0)

        # accumulate kk and qk across D
        for d in range(0, D):
            k_val    = tl.load(k_ptr + k_head_off + l * D + d)
            k_new_v  = tl.load(k_new_ptr + h * D + d)
            kk      += k_val * k_new_v

            # for all destinations r
            for r_idx in range(0, R):
                q_val   = tl.load(q_ptr + q_head_off + r_idx * D + d)
                qk_idx  = qk_head_off + l * R + r_idx
                qk_val  = tl.load(qk_ptr + qk_idx)
                qk_val += k_val * q_val
                tl.store(qk_ptr + qk_idx, qk_val)

        # decay = (relu(kk)**2 * r * static_dest)^(1/3)
        decay = kk
        decay = tl.maximum(decay, 0.0)
        decay = decay * decay
        decay = decay * r_val * static_dest
        # cubic root
        decay = decay ** (1.0 / 3.0)
        # clamp to [0, 1 - 1e-6]
        decay = tl.minimum(tl.maximum(decay, 0.0), 1.0 - 1e-6)
        # log1p(-decay)
        decay = tl.log1p(-decay)

        a_new = a_old + decay
        tl.store(a_ptr + a_head_off + l, a_new)

    # -----------------------------------------------------
    # Compute qk_new[h, r] = dot(k_new[h, :], q[h, r, :])
    # -----------------------------------------------------
    for r_idx in range(0, R):
        acc = 0.0
        for d in range(0, D):
            k_new_v = tl.load(k_new_ptr + h * D + d)
            q_val   = tl.load(q_ptr + q_head_off + r_idx * D + d)
            acc    += k_new_v * q_val
        tl.store(qk_new_ptr + qk_new_head_off + r_idx, acc)


# ---------------------------------------------------------
# PyTorch wrapper that mirrors your original snippet
# ---------------------------------------------------------
def ua_decode_step_triton(
    k, v, r, a, q,
    k_, v_,
    static_src, static_dest,
    thresh=None,
):
    """
    Triton-accelerated UA decoding step with cache pruning.

    Shapes (must match your snippet):
      k, v          : [1, H, L, D]
      r, a          : [1, H, L]
      q             : [1, H, R, D]
      k_, v_        : [1, H, D]
      static_src    : [1, H]
      static_dest   : [1, H]

    Returns:
      attn_out: [1, H, R, D]
      (k, v, r, a): updated caches
    """
    assert k.ndim == 4 and v.ndim == 4
    assert r.ndim == 3 and a.ndim == 3
    assert q.ndim == 4
    assert k_.shape == v_.shape
    assert k_.shape[0] == 1 and k.shape[0] == 1, "batch_size must be 1"
    assert q.shape[0] == 1, "batch_size must be 1"
    # Your original q_len == 1 constraint is about time-step, not R
    # (you only decode one new token, but q can have multiple destinations)
    # so we don't assert on q.shape[2] here.

    if thresh is None:
        thresh = math.log(1e-4)

    device = k.device
    dtype  = k.dtype

    B, H, L, D = k.shape
    _, _, R, _ = q.shape

    # Squeeze batch dim for Triton, ensure contiguous
    k_hld = k.squeeze(0).contiguous()       # [H, L, D]
    v_hld = v.squeeze(0).contiguous()       # [H, L, D]
    r_hl  = r.squeeze(0).contiguous()       # [H, L]
    a_hl  = a.squeeze(0).contiguous()       # [H, L]
    q_hrd = q.squeeze(0).contiguous()       # [H, R, D]

    k_new_hd      = k_.squeeze(0).contiguous()         # [H, D]
    v_new_hd      = v_.squeeze(0).contiguous()         # [H, D]
    static_src_h  = static_src.squeeze(0).contiguous() # [H]
    static_dest_h = static_dest.squeeze(0).contiguous()# [H]

    # Allocate outputs for logits
    qk_hlr    = torch.empty((H, L, R), device=device, dtype=dtype)
    qk_new_hr = torch.empty((H, R),     device=device, dtype=dtype)

    # Launch Triton kernel (one program per head)
    grid = (H,)
    ua_logits_kernel[grid](
        k_hld, k_new_hd, q_hrd, r_hl, static_dest_h, a_hl,
        qk_hlr, qk_new_hr,
        H=H, L=L, R=R, D=D,
        num_warps=4, num_stages=1,
    )

    # Put batch dim back
    a = a_hl.unsqueeze(0)            # [1, H, L]
    r = r_hl.unsqueeze(0)            # [1, H, L]
    qk = qk_hlr.unsqueeze(0)         # [1, H, L, R]
    qk_new = qk_new_hr.unsqueeze(0)  # [1, H, R]

    # ---------------------------------------------
    # Now reproduce your Python logic exactly
    # ---------------------------------------------
    # a_max over heads
    a_max = a.max(dim=1).values      # [1, L]
    a_min_idx = torch.argmin(a_max, dim=1).item()

    # Compute qk_ = einsum('bhd,bhrd->bhr') was done in kernel as qk_new

    if a_max[0, a_min_idx] < thresh:
        # Replace cache slot
        k[:, :, a_min_idx] = k_              # [1, H, D]
        v[:, :, a_min_idx] = v_              # [1, H, D]
        r[:, :, a_min_idx] = static_src      # [1, H]
        a[:, :, a_min_idx] = 0.0

        # update qk for that position to qk_new
        # qk: [1, H, L, R], qk_new: [1, H, R]
        qk[:, :, a_min_idx, :] = qk_new
    else:
        # Append new cache entry
        k = torch.cat((k, k_.unsqueeze(2)), dim=2)  # L -> L+1
        v = torch.cat((v, v_.unsqueeze(2)), dim=2)
        r = torch.cat((r, static_src.unsqueeze(2)), dim=2)
        a = torch.cat(
            (a, torch.zeros(1, H, 1, device=device, dtype=a.dtype)),
            dim=2
        )
        qk = torch.cat((qk, qk_new.unsqueeze(2)), dim=2)  # [1, H, L+1, R]

    # ---------------------------------------------
    # Final scaled attention = softmax(qk + a) @ v
    # ---------------------------------------------
    # qk: [1, H, L', R], a: [1, H, L']
    # attn weights over L' dimension
    attn = (
        qk.float()
          .add(a.unsqueeze(-1))      # broadcast over R
          .softmax(dim=2)
          .to(dtype=v.dtype)
          .transpose(-1, -2)         # [1, H, R, L']
          .matmul(v)                 # v: [1, H, L', D] -> [1, H, R, D]
    )

    keys, values, rates, affs = k, v, r, a
    return attn, (keys, values, rates, affs)
