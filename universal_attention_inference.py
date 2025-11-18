import math
import torch

def ua_inference(q, k_, v_, static_src, static_dest, k, v, r, a, thresh=math.log(1e-4)):
    batch_size = q.shape[0]
    kvheads = q.shape[1]
    
    assert batch_size == 1, "UA decoding with cache pruning only support batch_size=1"

    # Compute the decay first
    kk = torch.einsum('bhld,bhd->bhl', k, k_)
    qk = torch.einsum('bhld,bhrd->bhlr', k, q)
    decay = kk.relu().float().pow(2)
    decay = (decay * r * static_dest.unsqueeze(-1)).pow(1/3)
    decay = torch.log1p(decay.clamp(min=0, max=1-1e-6).neg())
    a = a + decay

    # Check to see if the minimal decay falls below threshold
    a_max = a.max(dim=1).values
    a_min_idx = torch.argmin(a_max, dim=1).item()

    qk_ = torch.einsum('bhd,bhrd->bhr', k_, q)

    # Update cache
    if a_max[0, a_min_idx] < thresh:
        k[:, :, a_min_idx] = k_
        v[:, :, a_min_idx] = v_
        r[:, :, a_min_idx] = static_src
        a[:, :, a_min_idx] = 0
        qk[:, :, a_min_idx] = qk_
    else:
        k = torch.cat((k, k_.unsqueeze(2)), dim=2)
        v = torch.cat((v, v_.unsqueeze(2)), dim=2)
        r = torch.cat((r, static_src.unsqueeze(2)), dim=2)
        a = torch.cat((a, torch.zeros(batch_size, kvheads, 1, device=a.device, dtype=a.dtype)), dim=2)
        qk = torch.cat((qk, qk_.unsqueeze(2)), dim=2)

    # Perform scaled attention
    attn = qk.float().add(a.unsqueeze(-1)).softmax(dim=2).to(dtype=v.dtype).transpose(-1,-2).matmul(v)  # b h r d

    (keys, values, rates, affs) = k,v,r,a

    return attn, (keys, values, rates, affs)
