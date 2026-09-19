import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import precompute_rope, apply_rope

class CausalGQA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.kv, self.d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.groups = self.h // self.kv
        self.q_proj = nn.Linear(cfg.hidden_size, self.h * self.d, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv * self.d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, cfg.hidden_size, bias=False)
        self.dropout = cfg.dropout
        cos, sin = precompute_rope(self.d, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B,T,self.h,self.d).transpose(1,2)
        k = self.k_proj(x).view(B,T,self.kv,self.d).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.kv,self.d).transpose(1,2)
        q, k = apply_rope(q,self.rope_cos,self.rope_sin), apply_rope(k,self.rope_cos,self.rope_sin)
        k, v = k.repeat_interleave(self.groups,1), v.repeat_interleave(self.groups,1)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.d)
        scores = scores.masked_fill(torch.ones(T,T,device=x.device,dtype=torch.bool).triu(1), float("-inf"))
        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        y = (probs @ v).transpose(1,2).contiguous().view(B,T,self.h*self.d)
        return self.o_proj(y)
