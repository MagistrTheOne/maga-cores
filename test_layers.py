import torch

from magicasual.layers import RMSNorm
from magicasual.layers import SwiGLU


B = 2
T = 16
D = 384


x = torch.randn(
    B,
    T,
    D
)


print("INPUT")
print(x.shape)


norm = RMSNorm(
    hidden_size=D
)

x_norm = norm(x)

print("RMSNorm")
print(x_norm.shape)


mlp = SwiGLU(
    hidden_size=384,
    intermediate_size=1024
)

y = mlp(x_norm)

print("SwiGLU")
print(y.shape)