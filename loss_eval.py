import torch
import torch.distributed as dist
import math

#公式 loss/in(2)/total_bytes
@torch.no_grad
def evaluate_bpb(model,val_loader,eval_setp, token_bytes,device):
    total_nats = torch.tensor(0.0,dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    dataset_iter = iter(val_loader)
    token_bytes = token_bytes.to(device)
    print(f"[DEBUG] device:, {device} token_bytes device: {token_bytes.device}, dtype: {token_bytes.dtype}")
    for _ in range(eval_setp):
        x,y,_= next(dataset_iter)
        _,loss = model(x,y, reduction='none')
        loss = loss.view(-1)

        y = y.view(-1)

        if (y.int() < 0).any():
            #mask
            valid = y >= 0
            safe_y = torch.where(valid,y,torch.zeros_like(y)).to(device)
            num_token_bytes = torch.where(valid,token_bytes[safe_y],torch.zeros_like(y))
            total_nats += (loss * (num_token_bytes > 0)).sum()
            total_bytes += num_token_bytes.sum()
        else:
            num_token_bytes = token_bytes[y]
            total_nats += (loss * (num_token_bytes > 0)).sum()
            total_bytes += num_token_bytes.sum()

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()

    if total_bytes == 0 :
        return float('inf')

    bpe =  total_nats/(math.log(2) * total_bytes)
    return bpe

@torch.no_grad
def estimate_loss(model,eval_steps,train_loader,val_loader,ctx):
    out =[]
    model.eval()
    for spilt in ['train','val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            if spilt == "val":
                val_iter = iter(val_loader)
                x,y = next(val_iter)
            else:
                x,y = next(train_loader)
            with ctx:
                _,loss = model(x,y)
            losses[k] = loss.item()
        out[spilt] = losses.mean()
    model.train()
    return out