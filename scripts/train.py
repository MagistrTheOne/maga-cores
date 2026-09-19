import math
from pathlib import Path
import torch
from magicasual.config import load_config
from magicasual.model import MagiCasualLLM
from training.dataset import TokenStream

cfg,run=load_config("configs/tiny.json")
if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
dev=torch.device("cuda"); torch.manual_seed(1337); torch.cuda.manual_seed_all(1337)
model=MagiCasualLLM(cfg).to(dev)
print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M GPU={torch.cuda.get_device_name(0)}")
data=TokenStream("data/prepared/train.bin",cfg.vocab_size)
opt=torch.optim.SGD(model.parameters(),lr=run["learning_rate"],momentum=run["momentum"],weight_decay=run["weight_decay"])
scaler=torch.amp.GradScaler("cuda"); accum=run["gradient_accumulation_steps"]
model.train(); opt.zero_grad(set_to_none=True)
for step in range(1,run["max_steps"]+1):
    total=0.0
    for _ in range(accum):
        x,y=data.batch(run["batch_size"],cfg.max_seq_len,dev)
        with torch.amp.autocast("cuda",dtype=torch.float16):
            _,loss=model(x,y)
        scaler.scale(loss/accum).backward(); total+=loss.item()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    if step<=run["warmup_steps"]: lr=run["learning_rate"]*step/max(1,run["warmup_steps"])
    else:
        q=(step-run["warmup_steps"])/max(1,run["max_steps"]-run["warmup_steps"])
        lr=run["learning_rate"]*.5*(1+math.cos(math.pi*q))
    for g in opt.param_groups: g["lr"]=lr
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    if step==1 or step%10==0: print(f"step={step:6d} loss={total/accum:.4f} lr={lr:.6g}")
    if step%500==0:
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save({"model":model.state_dict(),"config":cfg.__dict__,"step":step},f"checkpoints/step_{step:06d}.pt")
