# magicasualLLM: architecture and mathematics

## Token path

`[B,T] IDs -> Embedding -> [B,T,D]`. The embedding is a table `[V,D]`; indexing
selects one learned D-dimensional row for every token.

## RMSNorm

For one token: `rms(x)=sqrt(mean(x²)+eps)`, then `output=weight*x/rms(x)`.
LayerNorm came from older Transformers. RMSNorm removes mean subtraction because it
proved unnecessary for stable decoder training, retaining scale control with less work.

## RoPE

Feature pairs rotate by position: `even'=even*cos-odd*sin` and
`odd'=even*sin+odd*cos`. Only Q/K rotate. Their dot product therefore contains relative
distance without a learned absolute-position table.

## Grouped Query Attention

Q `[B,Hq,T,Dh]` asks; K `[B,Hkv,T,Dh]` advertises; V contains returned information.
After grouped K/V expansion, `Q @ K.transpose(-2,-1)` produces `[B,Hq,T,T]`.
Transpose places Dh on the contracted axis. Division by `sqrt(Dh)` controls score
variance. The upper triangle becomes `-inf`; softmax makes future probability zero.
GQA is the compromise between full MHA quality and MQA cache economy.

## SwiGLU and decoder

`[B,T,D] -> gate/value [B,T,I] -> SiLU(gate)*value -> [B,T,I] -> [B,T,D]`.
The learned gate improved quality per parameter over plain GELU FFNs. Every pre-norm
block computes `x=x+Attention(RMSNorm(x))`, then `x=x+SwiGLU(RMSNorm(x))`.

## Loss, backward, SGD

The head makes `[B,T,V]`; cross entropy asks which vocabulary item actually follows.
Autograd applies the chain rule through head, blocks, and embeddings. Our optimizer is
literally `W=W-lr*grad`, or with momentum `v=m*v+grad; W=W-lr*v`.
AdamW normally converges faster because it scales steps using gradient moments and
decouples weight decay. We accept slower, more sensitive SGD to expose the mechanism.

## RTX 2080 SUPER

The default is about 21M parameters. Batch 2, context 256, and accumulation 16 produce
an effective 32 sequences while retaining only two sequences' activations per backward.
FP16 parameters are small; activations, gradients, momentum, attention matrices, and
CUDA runtime make actual usage larger. This profile prioritizes clarity, not throughput.
