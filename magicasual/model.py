import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RMSNorm, SwiGLU
from .attention import CausalGQA

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn_norm=RMSNorm(cfg.hidden_size,cfg.rms_norm_eps)
        self.attn=CausalGQA(cfg)
        self.ffn_norm=RMSNorm(cfg.hidden_size,cfg.rms_norm_eps)
        self.mlp=SwiGLU(cfg.hidden_size,cfg.intermediate_size)
    def forward(self,x):
        x=x+self.attn(self.attn_norm(x))
        return x+self.mlp(self.ffn_norm(x))

class MagiCasualLLM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.embed_tokens=nn.Embedding(cfg.vocab_size,cfg.hidden_size)
        self.blocks=nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm=RMSNorm(cfg.hidden_size,cfg.rms_norm_eps)
        self.lm_head=nn.Linear(cfg.hidden_size,cfg.vocab_size,bias=False)
        if cfg.tie_word_embeddings: self.lm_head.weight=self.embed_tokens.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,0.0,0.02)
    def forward(self,input_ids,labels=None):
        x=self.embed_tokens(input_ids)
        for block in self.blocks: x=block(x)
        logits=self.lm_head(self.norm(x))
        loss=None
        if labels is not None:
            loss=F.cross_entropy(logits[:,:-1].contiguous().view(-1,self.cfg.vocab_size),
                                 labels[:,1:].contiguous().view(-1))
        return logits,loss
    @torch.no_grad()
    def generate(self,ids,max_new_tokens=64,temperature=0.8,top_k=40,eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits,_=self(ids[:,-self.cfg.max_seq_len:])
            z=logits[:,-1]/max(temperature,1e-5)
            if top_k:
                v,_=torch.topk(z,min(top_k,z.size(-1))); z[z<v[:,-1,None]]=float("-inf")
            nxt=torch.multinomial(torch.softmax(z,dim=-1),1)
            ids=torch.cat((ids,nxt),1)
            if eos_id is not None and torch.all(nxt.squeeze(-1)==eos_id): break
        return ids
