from pathlib import Path
import numpy as np
from tokenizer.tokenizer import MagiTokenizer
tok=MagiTokenizer(); text=Path("data/raw/train.txt").read_text(encoding="utf-8")
ids=tok.encode(text,bos=True,eos=True); dtype=np.uint16 if tok.vocab_size<65536 else np.uint32
a=np.asarray(ids,dtype=dtype); split=max(1,int(len(a)*.98)); out=Path("data/prepared"); out.mkdir(parents=True,exist_ok=True)
a[:split].tofile(out/"train.bin"); a[split:].tofile(out/"val.bin")
print(f"tokens={len(a):,} train={split:,} val={len(a)-split:,}")
