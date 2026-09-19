"""Decoder-only model assembled solely from this repository's layers."""
import torch
import torch.nn as nn
from .attention import CausalGQA
from .layers import RMSNorm, SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention = CausalGQA(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.feed_forward = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x):
        """INPUT [B,T,D] -> OUTPUT [B,T,D], with two identity residual paths."""
        attention_output = self.attention(self.attention_norm(x))  # [B,T,D]
        after_attention = x + attention_output  # [B,T,D]
        ffn_output = self.feed_forward(self.ffn_norm(after_attention))  # [B,T,D]
        return after_attention + ffn_output  # [B,T,D]

class MagiCasualLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._initialize)
        if cfg.tie_word_embeddings:
            # Input/output are the same vocabulary geometry. Weight tying reduces
            # parameters and has historically improved language-model perplexity.
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _initialize(module):
        """INPUT module -> initialized weights; shapes remain unchanged."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        """INPUT IDs [B,T] -> logits [B,T,V] and optional scalar loss []."""
        if input_ids.ndim != 2 or input_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError("input_ids must be [B,T] with T <= max_seq_len")
        hidden = self.embed_tokens(input_ids)  # [B,T] -> [B,T,D]
        for block in self.blocks:
            hidden = block(hidden)  # [B,T,D]
        logits = self.lm_head(self.final_norm(hidden))  # [B,T,D] -> [B,T,V]
        loss = None
        if labels is not None:
            # [B,T,V] -> [B*T,V]; every row predicts one next-token class.
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, self.cfg.vocab_size), labels.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=64, temperature=0.8, top_k=40, eos_id=None):
        """INPUT [B,T] -> OUTPUT [B,T+new]; intentionally simple uncached loop."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.cfg.max_seq_len:])
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)  # [B,V]
            if top_k:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < values[:, -1, None], float("-inf"))
            next_id = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)  # [B,1]
            ids = torch.cat((ids, next_id), dim=1)  # [B,T+1]
            if eos_id is not None and torch.all(next_id.squeeze(1) == eos_id):
                break
        return ids
