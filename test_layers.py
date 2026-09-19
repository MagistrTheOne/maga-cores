import torch
from magicasual.layers import RMSNorm,SwiGLU
from magicasual.config import load_config
from magicasual.model import MagiCasualLLM
cfg,_=load_config("configs/tiny.json"); x=torch.randn(2,16,cfg.hidden_size)
print("INPUT  ",x.shape); x=RMSNorm(cfg.hidden_size)(x); print("RMSNorm",x.shape)
x=SwiGLU(cfg.hidden_size,cfg.intermediate_size)(x); print("SwiGLU ",x.shape)
model=MagiCasualLLM(cfg); ids=torch.randint(0,cfg.vocab_size,(2,16)); logits,loss=model(ids,ids)
print("LOGITS ",logits.shape); print("LOSS   ",float(loss)); print("PARAMS ",f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")
