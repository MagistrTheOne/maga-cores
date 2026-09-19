import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, sequence, hidden]
        # Keep normalization math in fp32 for stability, then return to x dtype.
        input_dtype = x.dtype
        x_float = x.float()

        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)

        return (x_norm.to(input_dtype) * self.weight)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network used inside each Transformer block."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, T, I]
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # SiLU(gate) * up
        hidden = F.silu(gate) * up

        # [B, T, I] -> [B, T, D]
        return self.down_proj(hidden)
