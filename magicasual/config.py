import json
from dataclasses import dataclass


@dataclass
class ModelConfig:

    vocab_size: int
    max_seq_len: int

    hidden_size: int
    num_layers: int

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    intermediate_size: int

    rms_norm_eps: float
    rope_theta: float

    tie_word_embeddings: bool
    dropout: float


def load_config(path: str) -> dict:

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config