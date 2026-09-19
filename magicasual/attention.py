"""Manual causal Grouped Query Attention; no fused attention kernels."""
import math
import torch
import torch.nn as nn
from .rope import apply_rope, precompute_rope

class CausalGQA(nn.Module):
    """Explicit Q/K/V projections, score matrix, causal mask, softmax and value mix."""
    def __init__(self, cfg):
        super().__init__()
        self.query_heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.groups = self.query_heads // self.kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.query_heads * self.head_dim, cfg.hidden_size, bias=False)
        cos, sin = precompute_rope(self.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, return_intermediates=False):
        """INPUT [B,T,D] -> OUTPUT [B,T,D].

        Q asks what a token seeks; K describes what a token contains. Q @ K^T makes
        [B,H,T,Dh] @ [B,H,Dh,T] = [B,H,T,T], all token-to-token comparisons.
        K is transposed so Dh is the contracted axis. Dividing by sqrt(Dh) keeps the
        variance from growing with Dh and prevents softmax saturation.

        MHA stores K/V for every query head. MQA shares one pair and saves cache but
        can lose quality. GQA is the compromise: groups share K/V. Here 6 query heads
        share 2 K/V heads, reducing KV storage 3x while retaining multiple groups.
        """
        batch, length, hidden = x.shape
        q = self.q_proj(x).view(batch, length, self.query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        # q [B,Hq,T,Dh], k/v [B,Hkv,T,Dh]
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        k = k.repeat_interleave(self.groups, dim=1)  # [B,Hq,T,Dh]
        v = v.repeat_interleave(self.groups, dim=1)  # [B,Hq,T,Dh]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,Hq,T,T]
        mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # [B,Hq,T,T]
        context = torch.matmul(probabilities, v)  # [B,Hq,T,Dh]
        context = context.transpose(1, 2).contiguous().view(batch, length, hidden)  # [B,T,D]
        output = self.o_proj(context)  # [B,T,D]
        if return_intermediates:
            return output, {"q": q, "k": k, "v": v, "scores": scores, "probabilities": probabilities}
        return output
