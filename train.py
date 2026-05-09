# coding=utf-8
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
#import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import math
import numpy as np
import time
import gc
from model import GPTConfig, GPTModule
from contextlib import nullcontext

#I/O
out_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'

#learing_rate
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 600
max_steps = 6000

#data
dataset = 'code_pretrain_data'
batch_size = 16
block_size = 1024

#adawm optiomzer
beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1 #0.1
max_lr = max_lr * (batch_size / 32) ** 0.5
learning_rate = max_lr


#model 270M 可以改变数值 改变模型大小
# N = 20 * P  训练数据一般模型大小的20倍。
n_layer = 16
n_head = 16
n_embd = 1024
dropout = 0.1

#system
device = 'cuda' 

grad_clip = 0.5
max_iters = 6000
compile = True

#默认CPU 如果支持GPU 就使用GPU计算
#device = 'cpu'
#if torch.cuda.is_available():
    #device = 'cuda'

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


# model = GPTModule(GPTConfig())#GPTModule.from_pretrained('gpt2')
# model.to(device)
total_batch_size = 524288#0.5M: data one step data size
accumulation_steps = total_batch_size // (batch_size*block_size)

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert accumulation_steps % ddp_world_size == 0
    accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


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


data_path = os.path.join('data',dataset)

#数据权重
TRAIN_SOURCES = [
    ('train.bin', 0.55),           # 代码
    ('fineweb_train.bin', 0.25),        # 中文
    ('wiki_train.bin', 0.2),      # wiki
]

VAL_SOURCES = [
    ('val.bin', 0.55),
    ('fineweb_val.bin', 0.25),
    ('wiki_val.bin', 0.2),
]

_data_cache = {}

def _load_memmap(filename):
    if filename not in _data_cache:
        path = os.path.join(data_path, filename)
        _data_cache[filename] = np.memmap(path, dtype=np.uint16, mode='r')
    return _data_cache[filename]


def _init_sources(sources):
    mmaps = []
    weights = []
    for filename, weight in sources:
        mm = _load_memmap(filename)
        mmaps.append(mm)
        weights.append(weight)
    
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()
    cum_probs = np.cumsum(weights)
    
    max_indices = [len(mm) - block_size for mm in mmaps]
    
    return mmaps, cum_probs, max_indices


_train_mmaps, _train_cum, _train_max = _init_sources(TRAIN_SOURCES)
_val_mmaps, _val_cum, _val_max = _init_sources(VAL_SOURCES)

_OFFSET_X = np.arange(block_size)
_OFFSET_Y = np.arange(1, block_size + 1)

def get_batch(split):
    if split == 'train':
        mmaps, cum_probs, max_indices = _train_mmaps, _train_cum, _train_max
    else:
        mmaps, cum_probs, max_indices = _val_mmaps, _val_cum, _val_max


    r = np.random.random(batch_size)
    src_idx = np.searchsorted(cum_probs, r)

    x_np = np.empty((batch_size, block_size), dtype=np.int64)
    y_np = np.empty((batch_size, block_size), dtype=np.int64)

    for si, mm in enumerate(mmaps):
        mask = (src_idx == si)
        count = int(mask.sum())
        if count == 0:
            continue

        ixs = np.random.randint(0, max_indices[si], size=count)
        
        x_np[mask] = mm[ixs[:, None] + _OFFSET_X]
        y_np[mask] = mm[ixs[:, None] + _OFFSET_Y]

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    if device == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

#评估函数
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#torch.cuda.empty_cache()
# B = 4
# T = 1024
#ataload = DataLoad(B=B,T=T)

#提高矩阵计算速度，使用TF32 来替代float32 指数部分相同，位数部门只保留10位
torch.set_float32_matmul_precision('high')
#model = torch.compile(model) 

iter_num = 0
best_val_loss = 1e9

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  vocab_size=None, dropout=dropout)
meta_vocab_size = None

#根据参数选择是从头开始训练，还是在中断点继续训练
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 64000 (63992 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 64000
    gptconf = GPTConfig(**model_args)
    model = GPTModule(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'base.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPTModule(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    #加载参数
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
#精度缩放，主要针对float16，在训练的过程中，指数部分精度降低
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizer(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory#torch.optim.AdamW(model.parameters(), lr=3e-4,betas=(0.9,0.95),eps=1e-8)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

model.estimate_params()

#train.py loop
running_mfu = -1.0
raw_model = model.module if ddp else model
x, y= get_batch('train')
local_iter_num = 0

#查看训练效率
t0 = time.time()

while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'base.pt'))

    if iter_num == 0 and eval_only:
        break

    #accumulated_loss = 0.0
    #梯度累计
    t0 = time.time() 
    for micro_step in range(accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == accumulation_steps - 1)
        
        with ctx:
            logits, loss = model(x, y)
            # 损失缩放
            loss = loss / accumulation_steps
        
        x, y = get_batch('train')
        scaler.scale(loss).backward()
    
    #normal = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1.0)
    # 梯度裁剪，避免loss剧烈震荡
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    raw_model.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    #token_pre_sec = (batch_size * block_size * accumulation_steps) / (t1-t0)
    #print(f"setp{iter_num} | loss:{accumulated_loss.item():.4f} | dt: {dt:.2f}ms | norm:{normal:.4f} | tok/sec:{token_pre_sec:.2f}")
    iter_num += 1
    local_iter_num += 1

    if iter_num == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif iter_num % 100 == 0:
         torch.cuda.empty_cache()
         gc.collect()

#结束
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()