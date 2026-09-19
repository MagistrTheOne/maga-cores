"""Rotary Position Embeddings as explicit pairwise rotations."""
import torch

def precompute_rope(head_dim, max_seq_len, theta=10000.0):
    """INPUT scalar D,T -> OUTPUT cosine/sine tables [T,D/2].

    Learned absolute positions bind content to fixed indices. RoPE (2021) rotates Q
    and K pairs by position-dependent angles, so their dot product contains relative
    distance. It needs no learned table and became the practical decoder default.
    """
    if head_dim % 2:
        raise ValueError("RoPE requires even head_dim")
    pair_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)  # [D/2]
    inverse_frequency = 1.0 / (theta ** (pair_indices / head_dim))  # [D/2]
    positions = torch.arange(max_seq_len, dtype=torch.float32)  # [T]
    angles = torch.outer(positions, inverse_frequency)  # [T,D/2]
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    """INPUT [B,H,T,D] + [T,D/2] -> OUTPUT rotated [B,H,T,D]."""
    length = x.size(-2)
    cos = cos[:length][None, None].to(device=x.device, dtype=x.dtype)
    sin = sin[:length][None, None].to(device=x.device, dtype=x.dtype)
    even = x[..., 0::2]  # [B,H,T,D/2]
    odd = x[..., 1::2]  # [B,H,T,D/2]
    output = torch.empty_like(x)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output
