# magicasualLLM

From-scratch educational causal LLM derived from the SHINRA design language and sized for an RTX 2080.

## Stack

- SentencePiece Unigram, vocab 8192
- NFKC, byte fallback, split digits
- decoder-only Transformer
- hidden 384, 6 layers, context 256
- GQA: 6 query heads / 2 KV heads / head dim 64
- RoPE
- RMSNorm
- SwiGLU, intermediate 1024
- tied embedding / LM head
- FP16 autocast + GradScaler
- **SGD + momentum** rather than AdamW

PyTorch is only the tensor/autograd/CUDA engine. Attention, RoPE, blocks, model, tokenizer pipeline, batching and generation live here.

## Run

```powershell
python -m pip install -r requirements.txt
python test_layers.py
```

Put any UTF-8 learning text into `data/raw/train.txt`. No corpus is bundled.

```powershell
python tokenizer/train.py
python scripts/prepare_data.py
python scripts/train.py
```

Generate after a checkpoint exists:

```powershell
python scripts/generate.py checkpoints/step_000500.pt --prompt "Мага" --tokens 80
```

## Tensor path

```
text
 -> Unigram
 -> IDs [B,T]
 -> embeddings [B,T,384]
 -> 6 x (RMSNorm -> GQA + RoPE -> residual -> RMSNorm -> SwiGLU -> residual)
 -> RMSNorm
 -> LM head
 -> logits [B,T,8192]
 -> cross entropy
 -> SGD
```

The default profile is intentionally small enough to inspect and train locally. Tiny/random corpora are for mechanics and overfitting experiments, not general language capability.
