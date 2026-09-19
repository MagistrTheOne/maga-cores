import sentencepiece as spm
class MagiTokenizer:
    def __init__(self,path="tokenizer/artifacts/tokenizer.model"): self.sp=spm.SentencePieceProcessor(model_file=path)
    @property
    def vocab_size(self): return self.sp.vocab_size()
    def encode(self,text,bos=False,eos=False):
        x=self.sp.encode(text,out_type=int)
        return ([self.sp.bos_id()] if bos else [])+x+([self.sp.eos_id()] if eos else [])
    def decode(self,ids): return self.sp.decode(ids)
    def piece_to_id(self,p): return self.sp.piece_to_id(p)
