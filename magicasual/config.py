"""Configuration and architectural invariant checks."""
import json
from dataclasses import dataclass, fields
from pathlib import Path

@dataclass
class ModelConfig:
    vocab_size: int = 8192
    max_seq_len: int = 256
    hidden_size: int = 384
    num_layers: int = 6
    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 1024
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    def validate(self):
        """INPUT scalar fields -> OUTPUT validated config; no tensors yet."""
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("query heads must divide evenly into KV groups")
        if self.head_dim % 2:
            raise ValueError("RoPE rotates pairs, so head_dim must be even")
        return self

def load_config(path: str | Path):
    """INPUT JSON path -> OUTPUT ModelConfig and the full run dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    names = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in raw.items() if k in names}).validate(), raw
