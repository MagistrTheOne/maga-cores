from pathlib import Path
import sentencepiece as spm
SPECIAL=["<|system|>","<|user|>","<|assistant|>","<|reasoning|>","<|code|>","<|language|>","<|fim_prefix|>","<|fim_suffix|>","<|fim_middle|>","<|repo|>","<|file|>","<|tool_call|>","<|tool_response|>","<|document|>","<|end_of_text|>"]
src=Path("data/raw/train.txt"); out=Path("tokenizer/artifacts"); out.mkdir(parents=True,exist_ok=True)
if not src.exists(): raise SystemExit("Put UTF-8 text in data/raw/train.txt")
spm.SentencePieceTrainer.train(input=str(src),model_prefix=str(out/"tokenizer"),model_type="unigram",
 vocab_size=8192,character_coverage=1.0,normalization_rule_name="nfkc",unk_id=0,bos_id=1,eos_id=2,pad_id=3,
 unk_piece="<|unk|>",bos_piece="<|bos|>",eos_piece="<|eot|>",pad_piece="<|pad|>",
 user_defined_symbols=SPECIAL,byte_fallback=True,split_digits=True,hard_vocab_limit=False)
print("saved tokenizer/artifacts/tokenizer.model")
