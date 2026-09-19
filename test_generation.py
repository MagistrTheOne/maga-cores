"""Stage 4: autoregressive generation without tokenizer artifacts."""
import torch
from magicasual import MagiCasualLLM, ModelConfig

def main():
    cfg = ModelConfig(vocab_size=64, max_seq_len=16, hidden_size=32, num_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      intermediate_size=64).validate()
    model = MagiCasualLLM(cfg)
    prompt = torch.tensor([[1, 5, 9]], dtype=torch.long)  # [B=1,T=3]
    result = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=8)
    print("prompt IDs   ", tuple(prompt.shape))
    print("generated IDs", tuple(result.shape))
    assert result.shape == (1, 8)
    print("GENERATION PASSED")

if __name__ == "__main__":
    main()
