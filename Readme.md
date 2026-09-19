# magicasualLLM

Educational decoder-only LLM built from first principles in PyTorch. The goal is not
benchmark quality: every shape, equation, gradient, and update must remain inspectable.

## Frozen v0.1 architecture

| Component | Value |
|---|---:|
| Vocabulary / context | 8,192 / 256 |
| Hidden / blocks | 384 / 6 |
| Query / KV heads | 6 / 2 |
| Head dimension / SwiGLU width | 64 / 1,024 |
| Dropout / weight tying | 0 / yes |
| Precision / optimizer | FP16 / handwritten SGD + momentum |

No pretrained model, Hugging Face model class, ready Transformer block,
FlashAttention, Triton, xFormers, fused kernel, DeepSpeed, Megatron, or tensor
parallelism is used. PyTorch supplies tensors, autograd, CUDA, Embedding, Linear and
cross entropy. Attention, RoPE, normalization, FFN, decoder, optimizer, batching and
generation are implemented here.

## Run stages in order

```powershell
python -m pip install -r requirements.txt
python test_foundations.py
python test_layers.py
python test_generation.py
```

The tests cover NumPy -> Tensor -> Embedding -> Linear; then RMSNorm, SwiGLU, Q/K/V,
causal scores, logits, loss, backward and a real SGD update; then generation. Full
equations and historical reasoning live in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Train

Put UTF-8 text in `data/raw/train.txt`, then execute each independent stage:

```powershell
python tokenizer/train.py
python scripts/prepare_data.py
python scripts/train.py
```

The tokenizer is SentencePiece Unigram with NFKC, byte fallback, digit splitting and
8,192 tokens. Batch 2 with 16-step accumulation gives an effective batch of 32
sequences on an RTX 2080 SUPER 8 GB.

```powershell
python scripts/generate.py checkpoints/step_000500.pt --prompt "Мага" --tokens 80
```

Complete path: `text -> IDs [B,T] -> embedding [B,T,384] -> 6 decoder blocks ->`
`RMSNorm -> logits [B,T,8192] -> cross entropy -> backward -> SGD`.

This version deliberately recomputes context during generation and materializes full
`[T,T]` attention. KV caching and optimized kernels belong after the fundamentals.
