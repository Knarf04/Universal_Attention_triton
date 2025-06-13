import torch
from torch.autograd import Function

# The pytorch autograd version is borrowed from here: 
# https://github.com/daviswer/torchtitan/blob/sandbox-selfprune-clean-wd/torchtitan/models/llama/utils.py

class UniversalAttention(Function):
    @staticmethod
    def forward(kc, vc, xq, static_src, static_dest):
        b,h,r,l,d = xq.shape
        _,_,n,c,_ = kc.shape
        mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
        out = torch.empty(b,h,r,l,d,n, dtype=xq.dtype, device=xq.device)
        denom = torch.empty(b,h,r,l,n, dtype=xq.dtype, device=xq.device)
        static_src = static_src.pow(1/3)
        static_dest = static_dest.pow(1/3)
        kt = kc.view(b,h,l,d).transpose(-2,-1)  # b h d l
        for i in range(n):
            k_ = kc[:,:,i]  # b h c d
            v_ = vc[:,:,i]  # b h c d
            static_src_ = static_src[:,:,i]  # b h c

            # Calculate decay matrix
            affinity = k_.matmul(kt).relu().pow(2/3).float()  # deltanet style decay
            affinity = affinity * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)  # incorporate mamba-style and per-token decay
            affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c l
            affinity = affinity.triu(i*c+1).cumsum(3)  # Accumulate decay with causal masking
            affinity = affinity.masked_fill(mask.tril(i*c-1), -1e12)  # Re-mask, with 1s on diagonal

            # Perform actual attention operation
            score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(affinity.unsqueeze(2))  # b h r c l
            denom_ = score.logsumexp(dim=-2)  # b h r l
            out_ = score.transpose(-1,-2).softmax(dim=-1).to(dtype=xq.dtype).matmul(v_.unsqueeze(2))  # b h r l d

            out[...,i] = out_
            denom[...,i] = denom_
        return out, denom

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        kc,vc,xq,ss,sd = inputs
        # out,denom = outputs
        ctx.save_for_backward(kc,vc,xq,ss,sd)

    @staticmethod
    def backward(ctx, g_out, g_denom):
        # Note: when using mixed precision, g_out is downcast but g_denom is always fp32
        print(g_out.shape, g_denom.shape)
        kc,vc,xq,static_src,static_dest = ctx.saved_tensors
        dkc,dvc,dxq,dstat_src,dstat_dest = [torch.zeros_like(x) for x in [kc,vc,xq,static_src,static_dest]]
        b,h,r,l,d = xq.shape
        _,_,n,c,_ = kc.shape
        mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
        static_src = static_src.pow(1/3)
        static_dest = static_dest.pow(1/3)
        kt = kc.view(b,h,l,d).transpose(-2,-1)  # b h d l

        for i in range(n):
            k_ = kc[:,:,i]  # b h c d
            v_ = vc[:,:,i]  # b h c d
            static_src_ = static_src[:,:,i]  # b h c
            dout_ = g_out[...,i]
            ddenom_ = g_denom[...,i]

            # Rerun forward pass
            aff1 = k_.matmul(kt)
            aff2 = aff1.relu().pow(2/3).float()
            aff3 = aff2 * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)
            score = torch.log1p(aff3.clamp(min=0,max=1-1e-6).neg()).triu(i*c+1).cumsum(3).masked_fill(mask.tril(i*c-1), -1e12)
            score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(score.unsqueeze(2))  # b h r c l
            sscore = score.softmax(dim=-2)

            # Backward pass
            dvc[:,:,i] += sscore.to(dtype=dvc.dtype).matmul(dout_).sum(2)  # bhrcl,bhrld -> bhcd
            
            dscore = v_.unsqueeze(2).matmul(dout_.transpose(-1,-2))  # bhcd,bhrld -> bhrcl   <-- from out
            dscore = dscore.sub(dscore.mul(sscore).sum(-2,True)).mul(sscore)  # <-- from softmax
            dscore += score.softmax(-2) * ddenom_.unsqueeze(-2)  # b h r c l   <-- from denom

            dxq += dscore.to(dtype=dxq.dtype).transpose(-1,-2).matmul(k_.unsqueeze(2))  # bhrcl, bhcd -> bhrld
            dkc[:,:,i] += dscore.to(dtype=dkc.dtype).transpose(2,3).flatten(3,4).matmul(xq.flatten(2,3))  # bhrcl, bhrld -> bhcd

            daff = dscore.sum(2)  # b h c l
            daff = daff.flip([3]).cumsum(3).flip([3]).triu(i*c+1)  # <-- from cumsum
            daff /= aff3.clamp(min=1e-6, max=1-1e-6)-1  # <-- from ln(1-x)
            daff *= aff3.ge(0)
            daff *= aff3.le(1-1e-6)
            dstat = daff.mul(aff2).to(dtype=static_src.dtype)  # b h c l

            dstat_src[:,:,i] += dstat.mul(static_dest.unsqueeze(-2)).sum(-1).div(static_src_.pow(2).mul(3))  # bhcl, bhl -> bhc
            dstat_dest += dstat.mul(static_src_.unsqueeze(-1)).sum(-2).div(static_dest.pow(2).mul(3))  # bhcl, bhc -> bhl

            daff = daff.mul(static_src_.unsqueeze(-1)*static_dest.unsqueeze(-2))  # <-- from prod with statics
            daff = daff.to(dtype=xq.dtype) * aff1.abs().add(1e-9).pow(-1/3).mul(2/3).mul(aff1.gt(0))  # <-- from relu + pow

            dkc += daff.transpose(-1,-2).matmul(k_).view(b,h,n,c,d)  # bhcl, bhcd -> bhld, <-- grad via kt
            dkc[:,:,i] += daff.matmul(kt.transpose(-1,-2))  # bhcl, bhdl -> bhcd, <-- grad via k_

        return dkc,dvc,dxq,dstat_src,dstat_dest

