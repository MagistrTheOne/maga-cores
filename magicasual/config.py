import json
from dataclasses import dataclass, fields

@dataclass
class ModelConfig:
    vocab_size:int=8192; max_seq_len:int=256; hidden_size:int=384; num_layers:int=6
    num_attention_heads:int=6; num_key_value_heads:int=2; head_dim:int=64
    intermediate_size:int=1024; rms_norm_eps:float=1e-5; rope_theta:float=10000.0
    tie_word_embeddings:bool=True; dropout:float=0.0
    def validate(self):
        assert self.hidden_size == self.num_attention_heads*self.head_dim
        assert self.num_attention_heads % self.num_key_value_heads == 0
        return self

def load_config(path):
    with open(path,"r",encoding="utf-8") as f: raw=json.load(f)
    names={x.name for x in fields(ModelConfig)}
    return ModelConfig(**{k:v for k,v in raw.items() if k in names}).validate(), raw
