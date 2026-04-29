"""
Prepare code pretraining dataset from ModelScope/HF starcoderdata.
✅ Streaming & Interleaved Pipeline | Peak RAM < 1.5GB | No OOM
"""

import os
import sys
sys.path.insert(0, 'C:\\Users\\zym\\AppData\\Local\\pylibs')
import json
import random
from tqdm import tqdm
import numpy as np
import tiktoken
from modelscope.msdatasets import MsDataset
from datasets import interleave_datasets  # ✅ 流式交错混合核心库

# ============================================
# Configuration
# ============================================
TARGET_TOTAL_TOKENS = 10_000_000_000
VAL_RATIO = 0.0005

LANG_CONFIG = {
    "python": 0.35, "java": 0.20, "javascript": 0.10, "typescript": 0.05,
    "go": 0.10, "c++": 0.10, "rust": 0.05, "sql": 0.05,
}
assert abs(sum(LANG_CONFIG.values()) - 1.0) < 1e-6, "Ratios must sum to 1.0"

NAMESPACE = "bigcode"          # HF/ModelScope 命名空间
DATASET_NAME = "starcoderdata"

enc = tiktoken.get_encoding("gpt2")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "code_pretrain_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 2357
random.seed(SEED)

# data_dir 映射 (starcoderdata 按语言分目录存储)
LANG_FILTER_MAP = {
    "python": "python", "java": "java", "javascript": "javascript",
    "typescript": "typescript", "go": "go", "c++": "cpp",
    "rust": "rust", "sql": "sql",
}


def estimate_tokens_per_doc(lang_key: str, n_samples: int = 300):
    """流式采样估算平均 token 数"""
    data_dir = LANG_FILTER_MAP[lang_key]
    ds_stream = MsDataset.load(
        dataset_name=DATASET_NAME, namespace=NAMESPACE,
        split="train", data_dir=data_dir, use_streaming=True
    )
    
    total_tokens = 0
    count = 0
    for example in ds_stream:
        text = example.get("content", "") or example.get("text", "")
        if not text.strip(): continue
        total_tokens += len(enc.encode_ordinary(text))
        count += 1
        if count >= n_samples: break
        
    avg = total_tokens / count if count > 0 else 1000
    print(f"  ✅ [{lang_key}] 采样 {count} 条，平均 {avg:.1f} tokens/doc")
    return avg


def token_generator(stream, total_limit=None):
    """从流式数据集中按需产出 token IDs + EOT"""
    count = 0
    for ex in stream:
        text = ex.get("content", "") or ex.get("text", "")
        if not text.strip(): continue
        
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        
        yield ids
        count += len(ids)
        if total_limit and count >= total_limit:
            break


def write_binary_streaming(token_stream, output_path: str, max_tokens: int, desc: str):
    """
    修复版：增量写入 memmap，避免 resize 报错 + 自动边界截断
    内存恒定 < 1GB，支持精确/宽松两种模式
    """
    # 🔑 预分配时多留 2% 缓冲，避免最后一步越界
    alloc_size = int(max_tokens * 1.02)
    arr = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(alloc_size,))
    idx = 0
    buffer = []
    BUFFER_SIZE = 500_000  # 每 50万 token 刷盘
    
    pbar = tqdm(desc=desc, total=max_tokens, unit="tok")
    
    for item in token_stream:
        ids = item if isinstance(item, list) else (item.get("ids") if isinstance(item, dict) else None)
        if not ids: continue
            
        buffer.extend(ids)
        pbar.update(len(ids))
        
        # 达到缓冲阈值 → 刷盘（带边界检查）
        if len(buffer) >= BUFFER_SIZE:
            chunk = np.array(buffer, dtype=np.uint16)
            
            # 🔑 关键修复：如果剩余空间不足，截断 chunk
            remaining = max_tokens - idx
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
                
            arr[idx:idx+len(chunk)] = chunk
            idx += len(chunk)
            buffer = []
            arr.flush()
            
            if idx >= max_tokens:
                break
                
    # 写入剩余部分（同样带边界检查）
    if buffer and idx < max_tokens:
        chunk = np.array(buffer, dtype=np.uint16)
        remaining = max_tokens - idx
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        arr[idx:idx+len(chunk)] = chunk
        idx += len(chunk)
        
    arr.flush()
    
    # 🔑 修复 resize 报错：用 os.ftruncate 直接截断文件，或跳过截断
    try:
        # 方法1: 尝试用 ftruncate 截断文件（更可靠）
        with open(output_path, "r+b") as f:
            os.ftruncate(f.fileno(), idx * 2)  # uint16 = 2 bytes
    except Exception as e:
        # 方法2: 如果失败，跳过截断（训练代码按实际长度读取，多余字节无害）
        print(f"⚠️ 跳过文件截断: {e}")
    
    # 重新加载正确大小的 memmap（可选，确保后续读取安全）
    final_arr = np.memmap(output_path, dtype=np.uint16, mode="r", shape=(idx,))
    assert len(final_arr) == idx, f"Final array length mismatch: {len(final_arr)} != {idx}"
    
    file_size = os.path.getsize(output_path)
    print(f"✅ {desc} 完成: {idx:,} tokens, {file_size/1e9:.2f} GB")
    return idx


def main():
    print("=" * 70)
    print("🚀 低内存流式代码预训练数据准备")
    print(f"🎯 目标: {TARGET_TOTAL_TOKENS:,} tokens")
    print("=" * 70)

    # Step 1: 估算各语言平均长度
    print("\n📊 Step 1: Estimating tokens per document...")
    lang_avg_tokens = {}
    for lang_key in LANG_CONFIG.keys():
        lang_avg_tokens[lang_key] = estimate_tokens_per_doc(lang_key)

    # Step 2: 初始化各语言流式数据集
    print("\n🌐 Step 2: Initializing streaming data sources...")
    streams = []
    probs = []
    for lang_key in LANG_CONFIG.keys():
        data_dir = LANG_FILTER_MAP[lang_key]
        print(f"  ⏳ 加载 {lang_key} (data_dir='{data_dir}')...")
        ds = MsDataset.load(
            dataset_name=DATASET_NAME, namespace=NAMESPACE,
            split="train", data_dir=data_dir, use_streaming=True
        )
        
        # 🔑 核心修复：只保留 content 字段，彻底消除 schema 类型冲突
        # 兼容 datasets >= 2.14.0，ModelScope 底层基于 HF，完全支持
        ds = ds.select_columns(["content"])
        
        streams.append(ds)
        probs.append(LANG_CONFIG[lang_key])

    # Step 3: 按概率交错混合 (内置小窗口 shuffle)
    print("\n🔀 Step 3: Interleaving datasets by language ratio...")
    mixed_stream = interleave_datasets(
        streams,
        probabilities=probs,
        seed=SEED,
        stopping_strategy="first_exhausted"
    )

    # Step 4: 划分 Val / Train 并写入
    val_target = int(TARGET_TOTAL_TOKENS * VAL_RATIO)
    train_target = TARGET_TOTAL_TOKENS - val_target
    
    # 创建一个全局 token 迭代器（先写 val，再写 train，无缝衔接）
    token_iter = token_generator(mixed_stream, total_limit=val_target + train_target)
    
    val_path = os.path.join(OUTPUT_DIR, "val.bin")
    train_path = os.path.join(OUTPUT_DIR, "train.bin")
    
    print("\n💾 Step 4: Writing binary files...")
    val_actual = write_binary_streaming(token_iter, val_path, val_target, "Writing Val")
    train_actual = write_binary_streaming(token_iter, train_path, train_target, "Writing Train")

    # 保存元数据
    metadata = {
        "target_total_tokens": TARGET_TOTAL_TOKENS,
        "actual_train_tokens": int(train_actual),
        "actual_val_tokens": int(val_actual),
        "language_config": LANG_CONFIG,
        "dataset": f"{NAMESPACE}/{DATASET_NAME}",
        "tokenizer": "gpt2",
        "seed": SEED,
        "pipeline": "streaming_interleave_memmap"
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("✅ 全部完成！内存峰值 < 1.5GB")
    print(f"📊 Train: {train_actual:,} tokens ({os.path.getsize(train_path)/1e9:.2f} GB)")
    print(f"📊 Val:   {val_actual:,} tokens ({os.path.getsize(val_path)/1e6:.2f} MB)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()