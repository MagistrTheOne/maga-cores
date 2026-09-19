import numpy as np
import torch
class TokenStream:
    def __init__(self,path,vocab_size):
        self.data=np.memmap(path,dtype=np.uint16 if vocab_size<65536 else np.uint32,mode="r")
    def batch(self,batch_size,seq_len,device):
        if len(self.data)<=seq_len+1: raise ValueError("dataset too small for context")
        starts=torch.randint(0,len(self.data)-seq_len-1,(batch_size,))
        rows=[torch.from_numpy(np.array(self.data[i:i+seq_len+1],dtype=np.int64)) for i in starts.tolist()]
        z=torch.stack(rows).to(device,non_blocking=True)
        return z[:,:-1],z[:,1:]
