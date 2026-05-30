import inspect
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


#因为国内网络问题，只能访问镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 64000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    lora_rank: int = 0
    lora_alpha:float = 1.0
    # MoE 相关配置（预留，基础版本不使用）
    num_experts: int = 8
    top_k: int = 2
    aux_loss_coef: float = 0.01

def apply_rope_emb(x, cos, sin):
    """顺时针旋转实现"""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = (-sin) * x1 + x2 * cos
    return torch.cat([y1, y2], dim=3)

#如果不支持torch 不支持RMSNorm的话，请用下面版本
#y = (x * (mean(x**2) + eps)^2) * weight
class RMSNormFast(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self,x):
        orig_type = x.dtype
        x_f32 = x.float()

        var = (x_f32.pow(2).mean(dim = -1, keepdim = True))

        #数值稳定，如果数值很大超过范围用这个版本
        # max_abs = x_f32.abs().max(dim = -1,keepdim = True)
        # scale = torch.clamp(max_abs,min=1.0)
        # x_scaled = x_f32 / scale
        # var = (x_scaled.pow(2).mena(dim = -1, keepdim = True)) * (scale **2)

        x_f32.mul_(torch.rsqrt(var+ self.eps))

        return x_f32.to(orig_type) * self.weight

class RopeEmbedding(nn.Module):
    def __init__(self, dim, device=None, max_seq=2048, base=100000, dtype=torch.float32):
        super().__init__()  # 修复：加括号
        self.dim = dim
        self.max_seq = max_seq
        channel_range = torch.arange(0, dim, 2, device=device, dtype=dtype)
        inv_freqs = 1.0 / (base ** (channel_range / dim))
        self.inv_freqs = inv_freqs

        t = torch.arange(max_seq, device=device, dtype=dtype)
        freqs = torch.outer(t, inv_freqs)

        cos, sin = freqs.cos(), freqs.sin()
        self.register_buffer("cos", cos[None, None, :, :], persistent=False)
        self.register_buffer("sin", sin[None, None, :, :], persistent=False)

    @torch.no_grad()
    def forward(self, seq, device=None):
        if seq <= self.max_seq:
            return (self.cos[:, :, :seq, :],
                    self.sin[:, :, :seq, :])

        # 大于最大缓存长度，一般用于推理阶段，需要重新计算
        t = torch.arange(seq, device=device, dtype=self.inv_freqs.dtype)
        freqs = torch.outer(t, self.inv_freqs)
        cos, sin = freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]
        cos = cos.to(device)
        sin = sin.to(device)
        return cos, sin


class LoRALinear(nn.Module):
    """LoRA 低秩适配线性层
    W_new = W_0 + (alpha/r) * B @ A
    """
    def __init__(self, in_embd, out_embd, lora_rank, config):
        super().__init__()
        self.rank = lora_rank
        self.in_embd = in_embd
        self.out_embd = out_embd
        self.dropout = nn.Dropout(config.dropout)

        self.linear = nn.Linear(in_embd, out_embd, bias=False)

        if self.rank > 0:
            self.lora_A = nn.Linear(in_embd, self.rank, bias=False)
            self.lora_B = nn.Linear(self.rank, out_embd, bias=False)
            # 修复：scaling 应使用 alpha / rank，而非固定 1.0
            lora_alpha = getattr(config, 'lora_alpha', 1.0)
            self.scaling = lora_alpha / self.rank
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            # 冻结基座权重
            self.linear.weight.requires_grad = False
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        out = self.linear(x) #out shape ->(in_embd, out_embd)
        if self.rank > 0:
            #lora_shape -> 
            lora_out = self.lora_B(self.dropout(self.lora_A(x)))
            out = out + lora_out * self.scaling
        return out

class SelfAttention(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.lora_rank = config.lora_rank
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.rope = rope

        if self.lora_rank == 0:
            # 标准 QKV 合并投影
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            # LoRA 分支：Q, V 使用 LoRA，K 使用普通线性（或 rank=0 的 LoRA）
            self.c_attn = None
            self.q_proj = LoRALinear(config.n_embd, config.n_embd, self.lora_rank, config)
            self.k_proj = LoRALinear(config.n_embd, config.n_embd, 0, config)  # rank=0 等价普通 Linear
            self.v_proj = LoRALinear(config.n_embd, config.n_embd, self.lora_rank, config)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # FlashAttention 检测
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size()
        if self.lora_rank == 0:
            qkv = self.c_attn(x)  # (B, T, 3C)
            q, k, v = qkv.split(self.n_embd, dim=2)  # (B, T, C) each
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        # 拆分为多头 (B, H, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        cos, sin = self.rope(T, device=x.device)
        q = apply_rope_emb(q, cos, sin)
        k = apply_rope_emb(k, cos, sin)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    @torch.no_grad()
    def load_from_c_attn(self, c_attn_weight, c_attn_bias=None):
        """从合并的 c_attn 权重加载到分离的 Q/K/V 投影"""
        if self.lora_rank == 0:
            raise ValueError("当前为普通 Attention，无需分离加载")

        # 修复：统一获取 in_features
        base_proj = getattr(self, 'q_proj')
        base_linear = base_proj.linear if hasattr(base_proj, 'linear') else base_proj
        C = base_linear.in_features

        expected_shape = (3 * C, C)
        assert c_attn_weight.shape == expected_shape,f"权重形状不匹配: {c_attn_weight.shape} != {expected_shape}"

        q_w, k_w, v_w = c_attn_weight.chunk(3, dim=0)
        q_b = k_b = v_b = None
        if c_attn_bias is not None:
            expected_bias_shape = (3 * C,)
            assert c_attn_bias.shape == expected_bias_shape,f"bias 形状不匹配: {c_attn_bias.shape} != {expected_bias_shape}"
            q_b, k_b, v_b = c_attn_bias.chunk(3, dim=0)

        # 修复：zip 顺序与名称严格对应 [q, k, v]
        for proj_name, w, b in zip(["q_proj", "k_proj", "v_proj"],
                                     [q_w, k_w, v_w],
                                     [q_b, k_b, v_b]):
            base_proj = getattr(self, proj_name)
            base = base_proj.linear if hasattr(base_proj, 'linear') else base_proj
            base.weight.copy_(w)
            if b is not None:
                if base.bias is not None:
                    base.bias.copy_(b)
                else:
                    raise ValueError(f"{proj_name} 未启用 bias，但提供了 c_attn_bias")

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        # silu -> x * Sigmoid(x)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.silu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        # 修复：传入 config.n_embd 而非 config 对象
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = SelfAttention(config, rope)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # 标准 GPT 残差连接
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 修复：先创建 wte，再创建 rope，避免未定义引用
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.rope = RopeEmbedding(
            config.n_embd // config.n_head,
            device=self.wte.weight.device,
            max_seq=config.block_size * 2  # 给一定余量
        )

        # 构建 transformer 各组件
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config, self.rope) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：lm_head 与 wte 共享
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        wte_embd = self.wte(idx)
        x = self.drop(wte_embd)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction
            )

        return logits, loss
    
    def configure_optimizer(self, weight_decay,learning_rate,betas,device_type):
        dict_param = {pn:p for pn, p in self.named_parameters() if p.requires_grad}
        decay_param = []
        nodecay_param = []
        special_param = []

        for n, p in dict_param.items():
            if p.dim() >= 2:
                decay_param.append(p)
            else:
                nodecay_param.append(p)

        optim_groups = [
            {'params': decay_param, 'weight_decay': weight_decay,'lr':learning_rate},
            {'params': nodecay_param, 'weight_decay':0.0, 'lr':learning_rate},
        ]

        try:
            #如果显卡显存不够，可以使用bitsandbytes 能节省显存，从而能够支持训练更多的数据
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(optim_groups,lr = learning_rate, betas = betas)
            print("using 8 bit AdamW")
        except ImportError:
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused = True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups,learning_rate,betas,**extra_args)
            print("using torch.optim AdamW")
        return optimizer
    
    def get_detect_gpu(self):
        """自动检测 GPU 类型"""
        if not torch.cuda.is_available():
            return 'CPU'
        gpu_name = torch.cuda.get_device_name(0)
        if 'A100' in gpu_name: return 'A100'
        if 'A800' in gpu_name:return 'A800'
        if 'H100' in gpu_name: return 'H100'
        if 'V100' in gpu_name: return 'V100'
        if 'A10' in gpu_name: return 'A10'
        if '4090' in gpu_name: return 'RTX4090'
        if '4060' in gpu_name: return 'RTX4060'
        return 'Unknown'  # 默认
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #ROPE 替代 wpe embedding
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter, dt, world_size=None):
        """支持多卡的 MFU 计算"""
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        
        # 2. 计算每次前向+反向的 FLOPs
        # 参考 PaLM 论文 Appendix B: https://arxiv.org/abs/2204.02311
        # 6(2次前向+ 4次反向)
        #12*L*H*Q*T 计算推导如下：
        # Q@K^T  2* H*Q*T 
        #sofamax@v  ----> 2* H*Q*T 
        # 如果是1 token  那么需要的次数 q@k = q1*k1 + ......+ qn*kn 那么处理一个1个token 2次(一次乘法+ 一次加法)
        #扩展T个，那么需要2*T， 扩展到多头 2*H*Q*T
        #sofamax@v  ----> 2* H*Q*T 前向 2次
        #前向传播 4* H*Q*T
        #反向传播 8* H*Q*T 前向的2倍
        #B T C 这里把C 进行切分 划分 head 和 c//head大小，因为为了提高训练速度，把一个大的维度企划分多个
        #这里解释下为什么反向是正向的传播的2倍
        #前向传播 y= x@W 只执行一次
        #反应传播
        #这里需要计算dq 和 dk 的梯度  dq = ds @ k  dk = ds^T @q  --- > 4* H*Q*T
        #sofamax@v 反向 计算 d(权重) 和 dv --->4* H*Q*T
        #如果采用梯度累计，需要*梯度累计步数，因为每一次都是一次计算
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        flops_achieved = flops_per_iter * (1.0/dt)
        
        # GPU 峰值表
        gpu_peak_flops = {
            'A100': 312e12,
            'A800': 312e12,
            'H100': 989e12,
            'V100': 125e12,
            'A10': 125e12,
            'RTX4090': 82.6e12,
            'RTX4060': 48.7e12,
        }
        gpu_type = self.get_detect_gpu()
        single_gpu_peak = gpu_peak_flops.get(gpu_type, 312e12)
        total_peak = single_gpu_peak * world_size
        
        mfu = flops_achieved / total_peak
        return mfu
    
    # 评估模型大小
    def estimate_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        print(f"Parameters: {total_params / 1e6:.2f} M")
        print(f"Model size: {param_size / 1e9:.2f} GB (parameters only)")
    
    def generate(self, idx, max_new_tokens, temperature=0.2, top_k=None, eos_token_id=None):
        """
        工业级代码生成器（防幻觉/防内存泄漏/支持提前停止）
        """
        # 1. 类型/设备对齐
        device = next(self.parameters()).device
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long, device=device)
        else:
            idx = idx.long().to(device)
        
        # 2. 强制评估模式
        was_training = self.training
        self.eval()
        
        generated_len = 0
        
        try:
            with torch.no_grad():
                for step in range(max_new_tokens):
                    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                    logits, _ = self(idx_cond)

                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"[FATAL] step {step}, logits has nan/inf!")
                        print(f"  idx_cond min/max: {idx_cond.min()}/{idx_cond.max()}")
                        print(f"  idx_cond shape: {idx_cond.shape}")
                        # 检查 embedding 层
                        emb = self.wte(idx_cond)
                        print(f"  emb nan: {torch.isnan(emb).any()}")
                        # 直接 argmax 一个安全值，避免崩溃
                        idx_next = torch.zeros((idx.size(0), 1), dtype=torch.long, device=idx.device)
                        idx = torch.cat((idx, idx_next), dim=1)
                        continue  # 跳过这一步
                    
                    # Greedy or sampling
                    if temperature == 0.0 or temperature < 1e-6:
                        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    else:
                        logits = logits[:, -1, :] / temperature
                        
                        # Top-K
                        if top_k is not None and top_k > 0:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits = torch.where(logits < v[:, [-1]], torch.tensor(-1e9, device=logits.device), logits)
                        
                        probs = F.softmax(logits, dim=-1)
                        
                        # 防异常
                        if torch.isnan(probs).any() or (probs < 0).any():
                            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                        else:
                            idx_next = torch.multinomial(probs, num_samples=1)
                    
                    idx = torch.cat((idx, idx_next), dim=1)
                    generated_len += 1
                    
                    # EOS 检查
                    if eos_token_id is not None and (idx_next == eos_token_id).any():
                        break
                        
        finally:
            # 确保恢复训练状态（即使异常）
            if was_training:
                self.train()
        
        return idx 

#简单的 数据加载器
import tiktoken
class DataLoad:
    def __init__(self,B,T):
        super().__init__()
        self.B = B
        self.T = T
        with open('input.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_pos =0

    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_pos:self.current_pos + B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        self.current_pos = self.current_pos + B*T
        if self.current_pos + (B*T +1) > len(self.tokens):
            self.current_pos = 0
        return x,y







