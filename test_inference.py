import torch
import math
from Universal_Attention.utils import *
from universal_attention_inference import ua_inference as ua_inference_torch

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    device = "cuda"

    batch_size = 1
    seq_len = 4096
    q_len = 1
    nheads = 32
    kvheads = 8
    emb_kq_per_head = 128
    emb_v_per_head = 128

    # -----------------------
    # Create inputs on CUDA
    # -----------------------
    queries = torch.rand(batch_size, q_len, nheads, emb_kq_per_head, device=device)
    keys = torch.rand(batch_size, q_len, kvheads, emb_kq_per_head, device=device)
    keys = keys / keys.pow(2).sum(-1, True).sqrt().add(1e-6)
    values = torch.rand(batch_size, q_len, kvheads, emb_v_per_head, device=device)
    static_src = torch.rand(batch_size, kvheads, q_len, device=device).sigmoid()
    static_dest = torch.rand(batch_size, kvheads, q_len, device=device).sigmoid()

    k = torch.rand(batch_size, kvheads, seq_len, emb_kq_per_head, device=device)
    v = torch.rand(batch_size, kvheads, seq_len, emb_v_per_head, device=device)
    r = torch.rand(batch_size, kvheads, seq_len, device=device)
    a = torch.rand(batch_size, kvheads, seq_len, device=device)
    past_key_value_state = (k, v, r, a)

    (k, v, r, a) = past_key_value_state  # bhld, bhld, bhl, bhl
    k_ = keys.squeeze(1)  # b h d
    v_ = values.squeeze(1)  # b h d
    q = queries.view(batch_size, kvheads, -1, emb_kq_per_head)  # b h r d
    static_src = static_src.squeeze(2)  # b h
    static_dest = static_dest.squeeze(2)  # b h

    # -----------------------
    # Warmup (not profiled)
    # -----------------------
    for _ in range(10):
        _out, _state = ua_inference_torch(
            q, k_, v_, static_src, static_dest, k, v, r, a, thresh=math.log(1e-4)
        )
    torch.cuda.synchronize()

    # -----------------------
    # Profiling
    # -----------------------
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(30):
            _, _ = ua_inference_torch(
                q, k_, v_, static_src, static_dest, k, v, r, a, thresh=math.log(1e-4)
            )
            torch.cuda.synchronize()
            prof.step()  # mark iteration

    prof.export_chrome_trace("/gpfs/hshen/traces/ua_inference_torch.json")
