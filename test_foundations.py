"""Stage 0: NumPy thinking -> Tensor -> Embedding -> Linear."""
import numpy as np
import torch
import torch.nn as nn

def main():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    tensor = torch.from_numpy(array)
    print("NumPy -> Tensor", array.shape, "->", tuple(tensor.shape))
    ids = torch.tensor([[1, 3, 2], [0, 4, 1]])  # [B=2,T=3]
    hidden = nn.Embedding(5, 4)(ids)  # [B,T] -> [B,T,D=4]
    print("Embedding      ", tuple(ids.shape), "->", tuple(hidden.shape))
    projected = nn.Linear(4, 7, bias=False)(hidden)  # [B,T,4] -> [B,T,7]
    print("Linear         ", tuple(hidden.shape), "->", tuple(projected.shape))
    assert projected.shape == (2, 3, 7)
    print("FOUNDATIONS PASSED")

if __name__ == "__main__":
    main()
