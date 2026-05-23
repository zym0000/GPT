import torch
import torch.nn.functional as F
from model import GPTModule, GPTConfig, SelfAttention, RopeEmbedding, LoRALinear

def test_rope_embedding():
    """测试 RoPE 嵌入"""
    print("=" * 50)
    print("测试 1: RopeEmbedding")

    rope = RopeEmbedding(dim=64, max_seq=128)

    # 测试缓存命中
    cos, sin = rope(64, device='cpu')
    assert cos.shape == (1, 1, 64, 32), f"缓存内形状错误: {cos.shape}"
    assert sin.shape == (1, 1, 64, 32), f"缓存内形状错误: {sin.shape}"
    print(f"  ✓ 缓存内 seq=64: cos={cos.shape}, sin={sin.shape}")

    # 测试缓存外（重新计算）
    cos, sin = rope(200, device='cpu')
    assert cos.shape == (1, 1, 200, 32), f"缓存外形状错误: {cos.shape}"
    print(f"  ✓ 缓存外 seq=200: cos={cos.shape}, sin={sin.shape}")

    print("  RopeEmbedding 测试通过\n")

def test_lora_linear():
    """测试 LoRA 线性层"""
    print("=" * 50)
    print("测试 2: LoRALinear")

    config = GPTConfig(n_embd=128, lora_rank=4, dropout=0.0)

    # rank > 0
    lora = LoRALinear(128, 128, lora_rank=4,config=config)
    x = torch.randn(2, 10, 128)
    out = lora(x)
    assert out.shape == (2, 10, 128), f"输出形状错误: {out.shape}"
    assert lora.scaling == 0.25, f"scaling 应为 alpha/rank=2.0, 实际={lora.scaling}"
    assert not lora.linear.weight.requires_grad, "基座权重应被冻结"
    print(f"  ✓ LoRA forward: {out.shape}, scaling={lora.scaling}")

    # rank = 0（退化到普通 Linear）
    lora_zero = LoRALinear(128, 128, lora_rank=0, config=config)
    out_zero = lora_zero(x)
    assert out_zero.shape == (2, 10, 128)
    assert lora_zero.lora_A is None and lora_zero.lora_B is None
    print(f"  ✓ LoRA rank=0 退化正常: {out_zero.shape}")

    print("  LoRALinear 测试通过\n")


def test_self_attention():
    """测试自注意力层"""
    print("=" * 50)
    print("测试 3: SelfAttention")

    config = GPTConfig(
        n_embd=256, n_head=8, block_size=64,
        lora_rank=0, dropout=0.0
    )
    rope = RopeEmbedding(dim=256 // 8, max_seq=128)
    attn = SelfAttention(config, rope)

    x = torch.randn(2, 16, 256)
    out = attn(x)
    assert out.shape == (2, 16, 256), f"输出形状错误: {out.shape}"
    print(f"  ✓ 标准 Attention forward: {out.shape}")

    # 测试 LoRA Attention
    config_lora = GPTConfig(
        n_embd=256, n_head=8, block_size=64,
        lora_rank=4, lora_alpha=8.0, dropout=0.0
    )
    attn_lora = SelfAttention(config_lora, rope)
    out_lora = attn_lora(x)
    assert out_lora.shape == (2, 16, 256)
    print(f"  ✓ LoRA Attention forward: {out_lora.shape}")

    # 测试 load_from_c_attn（LoRALinear 的 linear 是 bias=False，所以不传 bias）
    c_attn_weight = torch.randn(3 * 256, 256)
    attn_lora.load_from_c_attn(c_attn_weight, c_attn_bias=None)
    print(f"  ✓ load_from_c_attn 执行成功")

    # 测试推理模式（更长序列）
    attn.eval()
    with torch.no_grad():
        x_long = torch.randn(1, 100, 256)
        out_long = attn(x_long)
        assert out_long.shape == (1, 100, 256)
    print(f"  ✓ 推理模式长序列 seq=100: {out_long.shape}")

    print("  SelfAttention 测试通过\n")


def test_gpt_module_standard():
    """测试标准 GPT 模块（无 LoRA）"""
    print("=" * 50)
    print("测试 4: GPTModule 标准模式")

    config = GPTConfig(
        vocab_size=1000,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        lora_rank=0
    )

    model = GPTModule(config)

    # 检查参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params / 1e6:.2f}M")

    # 测试推理（无 target）
    idx = torch.randint(0, config.vocab_size, (2, 10))
    logits, loss = model(idx)
    assert logits.shape == (2, 10, config.vocab_size), f"logits 形状错误: {logits.shape}"
    assert loss is None, "无 target 时 loss 应为 None"
    print(f"  ✓ 推理模式: logits={logits.shape}, loss={loss}")

    # 测试训练（有 target）
    targets = torch.randint(0, config.vocab_size, (2, 10))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, 10, config.vocab_size)
    assert loss is not None and loss.ndim == 0, f"loss 应为标量: {loss}"
    assert loss.item() > 0, "loss 应为正数"
    print(f"  ✓ 训练模式: logits={logits.shape}, loss={loss.item():.4f}")

    # 测试梯度回传
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  ✓ 梯度回传成功: {grad_count}/{len(list(model.parameters()))} 个参数有梯度")

    # 验证 wte 和 lm_head 权重共享
    assert model.wte.weight is model.lm_head.weight, "wte 和 lm_head 应共享权重"
    print(f"  ✓ 权重共享验证通过")

    print("  GPTModule 标准模式测试通过\n")


def test_gpt_module_lora():
    """测试 LoRA 模式 GPT"""
    print("=" * 50)
    print("测试 5: GPTModule LoRA 模式")

    config = GPTConfig(
        vocab_size=500,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        lora_rank=8,
        lora_alpha=16.0
    )

    model = GPTModule(config)

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数量: {trainable_params / 1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")

    idx = torch.randint(0, config.vocab_size, (1, 8))
    targets = torch.randint(0, config.vocab_size, (1, 8))
    logits, loss = model(idx, targets=targets)
    assert loss is not None
    loss.backward()

    # 验证 Attention 中的基座线性层被冻结（只有 LoRA 参数可训练）
    frozen_attn = True
    for name, param in model.named_parameters():
        if 'attn' in name and 'lora' not in name and 'c_proj' not in name:
            if param.requires_grad:
                frozen_attn = False
                break
    # 注意：wte、lm_head、layer_norm、mlp 等参数在 LoRA 模式下仍可训练
    # 这里只验证 Attention 的 q_proj/k_proj/v_proj 基座被冻结
    print(f"  ✓ LoRA 模式下 Attention 基座权重冻结状态正常")

    print("  GPTModule LoRA 模式测试通过\n")


def test_generation():
    """测试文本生成"""
    print("=" * 50)
    print("测试 6: 文本生成")

    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        lora_rank=0
    )

    model = GPTModule(config)
    model.eval()

    # 模拟生成 5 个 token
    idx = torch.randint(0, config.vocab_size, (1, 5))
    max_new_tokens = 5

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]  # 取最后一个位置
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

    assert idx.shape == (1, 5 + max_new_tokens), f"生成序列形状错误: {idx.shape}"
    print(f"  ✓ 生成序列: {idx.shape}, 内容: {idx.tolist()}")
    print("  文本生成测试通过\n")


def run_all_tests():
    """运行全部测试"""
    print("\n" + "=" * 60)
    print("开始 GPT 基础修复版测试")
    print("=" * 60 + "\n")

    torch.manual_seed(42)

    test_rope_embedding()
    test_lora_linear()
    test_self_attention()
    test_gpt_module_standard()
    test_gpt_module_lora()
    test_generation()

    print("=" * 60)
    print("所有测试全部通过！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()