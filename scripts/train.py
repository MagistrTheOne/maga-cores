import math
from pathlib import Path
import torch
from magicasual.config import load_config
from magicasual.model import MagiCasualLLM
from training.dataset import TokenStream
from training.trainer import TransparentSGD

cfg,run=load_config("configs/tiny.json")
if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
dev=torch.device("cuda"); torch.manual_seed(1337); torch.cuda.manual_seed_all(1337)
model=MagiCasualLLM(cfg).to(dev)
print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M GPU={torch.cuda.get_device_name(0)}")
data=TokenStream("data/prepared/train.bin",cfg.vocab_size)
# Intentionally avoid torch.optim: the update is visible in training/trainer.py.
opt=TransparentSGD(model.parameters(),learning_rate=run["learning_rate"],momentum=run["momentum"])
accum=run["gradient_accumulation_steps"]
# FP16 gradients can underflow. We multiply the loss before backward, then divide
# every gradient by the same constant. This is static loss scaling written openly.
loss_scale=1024.0
model.train(); opt.zero_grad(set_to_none=True)
for step in range(1,run["max_steps"]+1):
    total=0.0
    for _ in range(accum):
        x,y=data.batch(run["batch_size"],cfg.max_seq_len,dev)
        with torch.amp.autocast("cuda",dtype=torch.float16):
            _,loss=model(x,y)
        ((loss/accum)*loss_scale).backward(); total+=loss.item()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(loss_scale)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    if step<=run["warmup_steps"]: lr=run["learning_rate"]*step/max(1,run["warmup_steps"])
    else:
        q=(step-run["warmup_steps"])/max(1,run["max_steps"]-run["warmup_steps"])
        lr=run["learning_rate"]*.5*(1+math.cos(math.pi*q))
    opt.set_learning_rate(lr)
    opt.step(); opt.zero_grad()
    if step==1 or step%10==0: print(f"step={step:6d} loss={total/accum:.4f} lr={lr:.6g}")
    if step%500==0:
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save({"model":model.state_dict(),"config":cfg.__dict__,"step":step},f"checkpoints/step_{step:06d}.pt")
