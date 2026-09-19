"""Readable normalization and feed-forward layers."""
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Normalize magnitude without subtracting the mean.

    LayerNorm came from earlier Transformers. RMSNorm (2019) showed that mean
    subtraction is unnecessary for stable training, saving arithmetic while keeping
    the useful re-scaling. That simplicity made it common in modern decoder LLMs.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        """INPUT [B,T,D] -> OUTPUT [B,T,D]; normalize each token across D."""
        dtype = x.dtype
        x_float = x.float()
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)  # [B,T,1]
        normalized = x_float * torch.rsqrt(mean_square + self.eps)  # [B,T,D]
        return normalized.to(dtype) * self.weight  # [B,T,D] * [D]

def silu(x):
    """INPUT any shape -> OUTPUT same shape; SiLU(x)=x*sigmoid(x)."""
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """down(SiLU(gate(x))*value(x)).

    FFNs moved from ReLU to GELU for smooth gradients. GLUs then added a learned
    gate. SwiGLU combines that gate with SiLU and repeatedly improved quality per
    parameter, so LLaMA-like models adopted it. Three matrices cost more than two,
    hence its intermediate width is smaller than the historical 4D FFN.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x, return_intermediates=False):
        """INPUT [B,T,D] -> two [B,T,I] paths -> OUTPUT [B,T,D]."""
        gate = self.gate_proj(x)  # [B,T,D] -> [B,T,I]
        value = self.up_proj(x)  # [B,T,D] -> [B,T,I]
        gated = silu(gate) * value  # [B,T,I]
        output = self.down_proj(gated)  # [B,T,I] -> [B,T,D]
        if return_intermediates:
            return output, {"gate": gate, "value": value, "gated": gated}
        return output
