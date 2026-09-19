"""Smoke test printing every important intermediate shape."""
import torch
from magicasual import CausalGQA, MagiCasualLLM, RMSNorm, SwiGLU, load_config
from training.trainer import TransparentSGD

def main():
    torch.manual_seed(7)
    cfg, _ = load_config("configs/tiny.json")
    x = torch.randn(2, 16, cfg.hidden_size)
    print("input             ", tuple(x.shape), "[B,T,D]")
    normalized = RMSNorm(cfg.hidden_size)(x)
    print("RMSNorm           ", tuple(normalized.shape), "[B,T,D]")
    ffn_output, ffn = SwiGLU(cfg.hidden_size, cfg.intermediate_size)(normalized, True)
    print("SwiGLU gate/value ", tuple(ffn["gate"].shape), "[B,T,I]")
    print("SwiGLU output     ", tuple(ffn_output.shape), "[B,T,D]")
    attention_output, attn = CausalGQA(cfg)(x, True)
    print("Q/K/V expanded    ", tuple(attn["q"].shape), tuple(attn["k"].shape), tuple(attn["v"].shape))
    print("attention scores  ", tuple(attn["scores"].shape), "[B,H,T,T]")
    print("attention output  ", tuple(attention_output.shape), "[B,T,D]")
    assert torch.all(attn["probabilities"].triu(1) == 0), "causal mask leaks future tokens"
    model = MagiCasualLLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(ids, targets)
    print("logits            ", tuple(logits.shape), "[B,T,V]")
    print("loss              ", tuple(loss.shape), float(loss))
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    before = model.embed_tokens.weight.detach().clone()
    loss.backward()
    optimizer = TransparentSGD(model.parameters(), learning_rate=0.001)
    optimizer.step()
    assert not torch.equal(before, model.embed_tokens.weight)
    print("backward + SGD    ", "parameter changed")
    print("parameters        ", f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
