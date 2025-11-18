import torch 
import math
from Universal_Attention.utils import *
from universal_attention_inference import ua_inference as ua_inference_torch

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    # profiler = get_profiler()

    batch_size = 1
    seq_len = 4096
    q_len = 1
    nheads = 32
    kvheads = 8
    emb_kq_per_head = 128
    emb_v_per_head = 128

    queries = torch.rand(batch_size, q_len, nheads, emb_kq_per_head)
    keys = torch.rand(batch_size, q_len, kvheads, emb_kq_per_head)
    keys = keys / keys.pow(2).sum(-1, True).sqrt().add(1e-6)
    values = torch.rand(batch_size, q_len, kvheads, emb_v_per_head)
    static_src = torch.rand(batch_size, kvheads, q_len).sigmoid()
    static_dest = torch.rand(batch_size, kvheads, q_len).sigmoid()

    k = torch.rand(batch_size, kvheads, seq_len, emb_kq_per_head)
    v = torch.rand(batch_size, kvheads, seq_len, emb_v_per_head)
    r = torch.rand(batch_size, kvheads, seq_len)
    a = torch.rand(batch_size, kvheads, seq_len)
    past_key_value_state = (k, v, r, a)

    (k, v, r, a) = past_key_value_state  # bhld, bhld, bhl, bhl
    k_ = keys.squeeze(1)  # b h d
    v_ = values.squeeze(1)  # b h d
    q = queries.view(batch_size, kvheads, -1, emb_kq_per_head)  # b h r d
    static_src = static_src.squeeze(2)  # b h
    static_dest = static_dest.squeeze(2)  # b h

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(30):
            _, _ = ua_inference_torch(q, k_, v_, static_src, static_dest, k, v, r, a, thresh=math.log(1e-4))

    torch.cuda.synchronize()
    prof.export_chrome_trace(f'/gpfs/hshen/traces/ua_inference_torch.json')

