# coding=utf-8
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import math
import numpy as np
import time
import wandb
import gc
import json
from model import GPTConfig, GPTModule
from contextlib import nullcontext
from common import get_ddp_info
from loss_eval import evaluate_bpb
from transformers import AutoTokenizer
from yi_conversation_render import YiConversationRender

from pre_dataloader import pre_train_data_loader_best_fit

#-----------------------------------------------------
# 命令行参数 - 选择训练模式
import argparse
parser = argparse.ArgumentParser(description="预训练脚本")
parser.add_argument("--mode", type=str, default="code", choices=["code", "wiki", "fineweb"],
                    help="训练模式: code=代码预训练, wiki=中文语料, fineweb=英文语料")
parser.add_argument("-c ","--compile",type=bool, default=False,help="enable compile model")
parser.add_argument('-b',"--device-batch-size",type=int,default=1,help="device per batch")
parser.add_argument('-s',"--max-seq-len",type=int,default=1024,help="device per seq size")
parser.add_argument('-i',"--init-from",type=str,default="scratch",help="train mode 'scratch' or 'resume' or 'gpt2*'")
parser.add_argument("--save-every",type= int,default=1000,help="train save step")
parser.add_argument("--eval-step", type= int, default = 200, help="train eval step")
parser.add_argument("--resume-from-step",type=int, default=-1 ,help="resume training from this step (-1 = disable)")
args = parser.parse_args()
#-----------------------------------------------------

ddp,ddp_rank,ddp_local_rank,ddp_world_size = get_ddp_info()

master_process = False

if ddp:
    master_process = ddp_rank == 0
else:
    master_process = True

def print0(text):
    if master_process:
        print(text)

def save_checkpoint(checkpoint_dir,step,model_data,optimzer_data, meta_data,ddp_rank):
    if ddp_rank == 0:
        os.makedirs(checkpoint_dir,exist_ok = True)
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data,model_path)

        meta_path = os.path.join(checkpoint_dir,f"meta_{step:06d}.json")
        with open(meta_path,'w',encoding="utf-8") as f:
            json.dump(meta_data,f,indent=2)

    if optimzer_data is not None:
        os.makedirs(checkpoint_dir,exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir,f"optim_{step:06d}_rank{ddp_rank}.pt")
        torch.save(optimzer_data,optimizer_path)


def load_checkpoint(checkpoint_dir, step,device, load_optimzer = False, rank =0):
    model_path = os.path.join(checkpoint_dir,f"model_{step:06d}.pt")
    model_data  = torch.load(model_path,map_location=device)

    optimizer_data = None
    if load_optimzer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
        optimizer_data  = torch.load(optimizer_data,map_location=device)

    meta_path = os.path.join(checkpoint_dir,f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    
    return model_data,optimizer_data, meta_data

#I/O
out_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
init_from = args.init_from # 'scratch' or 'resume'
resume = init_from == 'resume'
#learing_rate
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 600
max_steps = 6000

#data
dataset = 'code_pretrain_data'
batch_size = args.device_batch_size
block_size = args.max_seq_len
#adawm optiomzer
beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1 #0.1
max_lr = max_lr * (batch_size / 32) ** 0.5
learning_rate = max_lr

#model 270M 可以改变数值 改变模型大小
# N = 20 * P  训练数据一般模型大小的20倍。
n_layer = 12
n_head = 16
n_embd = 1024 #1536
dropout = 0.1

#system
device = 'cuda' 

grad_clip = 1.0
max_iters = 6000
compile = args.compile
#所有ddp 一次看到处理的token
total_batch_size = 524288#0.5M: data one step data size

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

#log
wandb_log = False

# -----------------------------------------------------------------------------
# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging
config = {}
# ----------------------------------------------------------------------------

dtype = 'bfloat16'  if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' #混合精度

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32':torch.float32, 'bfloat16':torch.bfloat16,'float16':torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if ddp:
    init_process_group(backend=backend)
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)

#init tokenizer
ENCODING = "01-ai/Yi-6B"
tokenizer = AutoTokenizer.from_pretrained(
    ENCODING,
    trust_remote_code=True,
    use_fast=True
)

tokenizer = YiConversationRender(tokenizer,device= device_type)

#提高矩阵计算速度，使用TF32 来替代float32 指数部分相同，位数部门只保留10位
torch.set_float32_matmul_precision('high')

step = 0

#如果是其他模型，可以填其他模型vocab_size
meta_vocab_size = tokenizer.get_vocab_size()
print0(f"vocab size: {meta_vocab_size}")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  vocab_size=meta_vocab_size, dropout=dropout)

gpt_config = GPTConfig(**model_args)
model = GPTModule(gpt_config)
smooth_train_loss = 0

if resume:
    print0(f"Resuming training from {out_dir}")
    checkpoint_dir = os.path.join(out_dir, 'base.pt')
    model_data,optimizer_data,meta_data = load_checkpoint(out_dir,args.resume_from_step,device_type,True,ddp_rank)
    #state_dict = model_data['model']
    # unwanted_prefix = '_orig_mod.'
    # for k,v in list(state_dict.items()):
    #     if k.startswith(unwanted_prefix):
    #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(model_data)
    step = meta_data['step']
    batch_size = meta_data['device_batch_size']
    total_batch_size = meta_data['total_batch_size']
    smooth_train_loss = meta_data['smooth_train_loss']

    del model_data

#因为显存不够，使用梯度累加一次性处理0.5M的数据
token_per_fwdbwd = batch_size* block_size
world_token_per_fwdbwd = token_per_fwdbwd * ddp_world_size
accumulation_steps = max(1, total_batch_size // world_token_per_fwdbwd)
seed_offset = ddp_rank

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

model.to(device)
orig_model = model

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
#精度缩放，主要针对float16，在训练的过程中，指数部分精度降低
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizer(weight_decay, learning_rate, (beta1, beta2), device_type)
if resume:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

raw_model.estimate_params()

#train.py loop
running_mfu = -1.0

#data path
code_dir = "data/code"
nl_dir = "data/nl"

dataloader_resume_state_dict = None if not resume else meta_data["dataloader_state_dict"]
train_loader = pre_train_data_loader_best_fit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", 
                                              device=device, 
                                              code_dir=code_dir,
                                              nl_dir= nl_dir,
                                              resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: pre_train_data_loader_best_fit(tokenizer, args.device_batch_size, 
                                                          args.max_seq_len, 
                                                          split="val", device=device,
                                                          code_dir= code_dir, nl_dir= nl_dir)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

ema_beta = 0.9
total_training_time = 0

min_bpb = float("-inf")

#使用动态学习曲线，为了加速收敛，增强训练稳定性，提高模型的泛化能力
def get_lr(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / (warmup_steps + 1)
    
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5*(1+ math.cos(math.pi * decay_ratio))
    lr = min_lr + (max_lr - min_lr) * coeff
    
    return lr

while True:
    last_step = step == max_iters
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if step % args.eval_step == 0:
        model.eval()
        token_bytes = tokenizer.get_token_bytes()
        val_bpb = evaluate_bpb(model,build_val_loader(),eval_iters,token_bytes,device_type)
        print0(f"step {step} | Validation bpb: {val_bpb:.4f}")

        if val_bpb < min_bpb:
            min_bpb = val_bpb

        if wandb_log:
            wandb.log({
                "iter": step,
                "train/loss": smooth_train_loss, 
                "min_bpb": min_bpb,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        model.train()

    if last_step or (step> 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(out_dir,
                        step,
                        orig_model.state_dict(),
                        optimizer.state_dict(),
                        {
                            'step': step,
                            'dataloader_state_dict':dataloader_state_dict,
                            'total_batch_size': total_batch_size,
                            'max_seq_len':args.max_seq_len,
                            'device_batch_size':args.device_batch_size,
                            'smooth_train_loss': smooth_train_loss,
                        },ddp_rank)
    
    if step == 0 and eval_only:
        break

    if last_step:
        break

    #梯度累计
    torch.cuda.synchronize()
    t0 = time.time() 
    for micro_step in range(accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == accumulation_steps - 1)

        with ctx:
            logits, loss = model(x, y)
            # 损失缩放
            train_loss = loss.detach()
            loss = loss / accumulation_steps
        #require_backward_grad_sync 决定是否同步，默认为TRUE，避免频繁同步，这里在最后一步进行同步
        scaler.scale(loss).backward()
        x, y,dataloader_state_dict = next(train_loader)
    
    #normal = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1.0)
    # 梯度裁剪，避免loss剧烈震荡
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if step > 10:
        total_training_time+= dt

    #ema的目的是，让smooth_train_loss下降更加平稳，在真实训练中，loss会震荡，无法看到明确的下降的趋势，使用ema 就是去除噪声
    #ema_beta 占比系数
    smooth_train_loss = smooth_train_loss * ema_beta + (1-ema_beta)* train_loss_f
    #debiased_smooth_loss 是EMA 偏差修正，冷启动导致EMA偏低，影响数据观测
    #EMA_t ≈ (1 - β^t) × 真实平均值
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    #这里可以这样计算原因是，每一setp时间都处理total_token_size大小的token
    tok_per_sec = int(total_batch_size / dt)
    mfu = orig_model.estimate_mfu(batch_size * accumulation_steps, dt,ddp_world_size)
    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step} | loss: {debiased_smooth_loss:.6f} | lr: {lr:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m")

    step += 1

    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 100 == 0:
         torch.cuda.empty_cache()
         gc.collect()

if ddp:
    destroy_process_group()