import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5):

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )


    def forward(self, x):

        # x:
        # [batch, sequence, hidden]

        variance = x.pow(2).mean(
            dim=-1,
            keepdim=True
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * x