import torch

def precompute_rope(head_dim, max_seq_len, theta=10000.0, device=None):
    assert head_dim % 2 == 0
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device).float()
    freq = torch.outer(pos, inv)
    return freq.cos(), freq.sin()

def apply_rope(x, cos, sin):
    t = x.size(-2)
    cos = cos[:t][None, None].to(x.dtype)
    sin = sin[:t][None, None].to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out
