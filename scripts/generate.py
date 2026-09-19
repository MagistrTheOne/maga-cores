import argparse, torch
from magicasual.config import load_config
from magicasual.model import MagiCasualLLM
from tokenizer.tokenizer import MagiTokenizer
p=argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--prompt",default="Hello"); p.add_argument("--tokens",type=int,default=80); a=p.parse_args()
cfg,_=load_config("configs/tiny.json"); tok=MagiTokenizer(); dev="cuda" if torch.cuda.is_available() else "cpu"
model=MagiCasualLLM(cfg).to(dev); ck=torch.load(a.checkpoint,map_location=dev); model.load_state_dict(ck["model"])
ids=torch.tensor([tok.encode(a.prompt,bos=True)],dtype=torch.long,device=dev)
print(tok.decode(model.generate(ids,a.tokens,eos_id=tok.piece_to_id("<|eot|>"))[0].tolist()))
