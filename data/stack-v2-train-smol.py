"""
Prepare code pretraining dataset from ModelScope/HF starcoderdata.
Fixed: SSL/HuggingFace blocked -> Use ModelScope CDN
Fixed: Precise Val/Train split | Windows Compatible | Robust Buffer Handling
Added: EOS token appended to every document
"""

import os
import sys
import time
import traceback

# ===== 全局异常捕获（防止 Windows 终端闪退）=====
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("\n未捕获的异常:", flush=True)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    input("\n按回车键退出...")

sys.excepthook = handle_exception
# =================================================

print(f"脚本启动 [{time.strftime('%H:%M:%S')}]", flush=True)

# 如需本地 pylibs，保留；否则可注释
sys.path.insert(0, r'C:\\Users\\zym\\AppData\\Local\\pylibs')

import json
from tqdm import tqdm
import numpy as np

#从 ModelScope 加载 tokenizer，走国内镜像，绕过 HuggingFace SSL 问题
from modelscope import AutoTokenizer
from modelscope.msdatasets import MsDataset
from datasets import interleave_datasets

# ============================================
# Configuration
# ============================================
DEBUG_MODE = False  # True=只跑100万token快速验证

if DEBUG_MODE:
    TARGET_TOTAL_TOKENS = 1_000_000
    VAL_RATIO = 0.001
    ESTIMATE_SAMPLES = 50
    BUFFER_SIZE = 50_000
else:
    TARGET_TOTAL_TOKENS = 12_000_000_000
    VAL_RATIO = 0.001
    ESTIMATE_SAMPLES = 500
    BUFFER_SIZE = 500_000

LANG_CONFIG = {
    "python": 0.25,
    "c": 0.2,
    "javascript": 0.15,
    "go": 0.20,
    "c++": 0.20,
}
assert abs(sum(LANG_CONFIG.values()) - 1.0) < 1e-6

NAMESPACE = "bigcode"
DATASET_NAME = "starcoderdata"

# ModelScope 上的模型 ID（注意大小写与 HuggingFace 不同）
ENCODING = "01-ai/Yi-6B"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "code_pretrain_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 2357

LANG_FILTER_MAP = {
    "python": "python", "java": "java", "javascript": "javascript",
    "go": "go", "c++": "cpp", "c": "c"
}

# ============================================
# Step 0: Load Tokenizer from ModelScope
# ============================================
print(f"[{time.strftime('%H:%M:%S')}] 从 ModelScope 加载 tokenizer: {ENCODING} ...", flush=True)
try:
    tokenizer = AutoTokenizer.from_pretrained(
        ENCODING,
        trust_remote_code=True,
        use_fast=True
    )
except Exception as e:
    print(f"Tokenizer 加载失败: {e}", flush=True)
    raise

print(f"Tokenizer 加载成功 | vocab_size: {len(tokenizer)}", flush=True)

eos_id = tokenizer.eos_token_id
if eos_id is None:
    print("tokenizer 没有 eos_token_id，将不使用 EOS", flush=True)
else:
    print(f"EOS token id: {eos_id} ({tokenizer.eos_token})", flush=True)

# ============================================
# Helper Functions
# ============================================

def estimate_tokens_per_doc(lang_key: str, n_samples: int = ESTIMATE_SAMPLES):
    """快速估算平均 token 数，用于预估时间"""
    data_dir = LANG_FILTER_MAP[lang_key]
    print(f"  [{time.strftime('%H:%M:%S')}] 估算 {lang_key} (data_dir='{data_dir}')...", flush=True)
    try:
        ds = MsDataset.load(
            dataset_name=DATASET_NAME, namespace=NAMESPACE,
            split="train", data_dir=data_dir, use_streaming=True
        )
        total_tokens = 0
        count = 0
        for example in ds:
            text = example.get("content", "") or example.get("text", "")
            if text.strip():
                total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
                total_tokens += 1 if eos_id is not None else 0
                count += 1
                if count >= n_samples:
                    break
        avg = total_tokens / count if count > 0 else 1000
        print(f"  [{lang_key}] Avg tokens/doc (w/ EOS): {avg:.1f}", flush=True)
        return avg
    except Exception as e:
        print(f"  [{lang_key}] Estimate failed: {e}", flush=True)
        return 1000

def write_buffer_to_memmap(arr, buffer, idx, max_idx):
    """将 buffer 写入 memmap,处理边界截断"""
    if not buffer:
        return idx, []
    
    chunk = np.array(buffer, dtype=np.uint16)
    remaining = max_idx - idx
    
    if remaining <= 0:
        return idx, []
    
    if len(chunk) > remaining:
        chunk_to_write = chunk[:remaining]
        remaining_buffer = chunk[remaining:].tolist()
    else:
        chunk_to_write = chunk
        remaining_buffer = []
        
    arr[idx : idx + len(chunk_to_write)] = chunk_to_write
    arr.flush()
    return idx + len(chunk_to_write), remaining_buffer

def main():
    print("=" * 70, flush=True)
    print("Code Pretrain Data Prep (ModelScope CDN + EOS Version)", flush=True)
    print(f"Target: {TARGET_TOTAL_TOKENS:,} tokens", flush=True)
    print(f"Debug mode: {DEBUG_MODE}", flush=True)
    print(f"EOS token: {eos_id}", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Estimate
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 1: Estimating...", flush=True)
    for lang in LANG_CONFIG:
        estimate_tokens_per_doc(lang)

    # Step 2: Load Streams
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 2: Loading streams from ModelScope...", flush=True)
    streams = []
    probs = []
    for lang_key in LANG_CONFIG:
        data_dir = LANG_FILTER_MAP[lang_key]
        print(f"  [{time.strftime('%H:%M:%S')}] Loading {lang_key} (data_dir='{data_dir}')...", flush=True)
        
        try:
            ds = MsDataset.load(
                dataset_name=DATASET_NAME, namespace=NAMESPACE,
                split="train", data_dir=data_dir, use_streaming=True
            )
            ds = ds.select_columns(["content"])
            streams.append(ds)
            probs.append(LANG_CONFIG[lang_key])
            print(f"  {lang_key} stream loaded", flush=True)
        except Exception as e:
            print(f"  {lang_key} 加载失败: {e}", flush=True)
            raise

    print(f"[{time.strftime('%H:%M:%S')}] Interleaving {len(streams)} streams...", flush=True)
    mixed_stream = interleave_datasets(
        streams, probabilities=probs, seed=SEED, stopping_strategy="first_exhausted"
    )
    print(f"Interleave done", flush=True)

    # Step 3: Prepare Files
    val_target = int(TARGET_TOTAL_TOKENS * VAL_RATIO)
    train_target = TARGET_TOTAL_TOKENS - val_target
    
    alloc_factor = 1.06 
    train_path = os.path.join(OUTPUT_DIR, "train.bin")
    val_path = os.path.join(OUTPUT_DIR, "val.bin")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 3: Writing memmap files...", flush=True)
    print(f"   Val Target:   {val_target:,}", flush=True)
    print(f"   Train Target: {train_target:,}", flush=True)
    print(f"   Output dir:   {OUTPUT_DIR}", flush=True)

    print(f"   Pre-allocating val memmap ({int(val_target * alloc_factor * 2 / 1e6):.0f} MB)...", flush=True)
    arr_val = np.memmap(val_path, dtype=np.uint16, mode='w+', shape=(int(val_target * alloc_factor)))
    
    print(f"   Pre-allocating train memmap ({int(train_target * alloc_factor * 2 / 1e9):.2f} GB)...", flush=True)
    arr_train = np.memmap(train_path, dtype=np.uint16, mode='w+', shape=(int(train_target * alloc_factor)))
    print(f"   Memmap ready", flush=True)

    idx_val = 0
    idx_train = 0
    buf_val = []
    buf_train = []
    
    pbar = tqdm(total=TARGET_TOTAL_TOKENS, desc="Total Tokens", unit="tok")
    
    n_docs = 0
    print(f"[{time.strftime('%H:%M:%S')}] Starting iteration...", flush=True)
    
    for ex in mixed_stream:
        text = ex.get("content", "") or ex.get("text", "")
        if not text.strip(): 
            continue
        
        if len(text) > 500_000: 
            continue 
        
        ids = tokenizer.encode(text, add_special_tokens=False)
        
        # ==================== EOS ====================
        if eos_id is not None:
            ids.append(eos_id)
        # =============================================
        
        n_tok = len(ids)
        n_docs += 1
        
        if idx_val < val_target:
            buf_val.extend(ids)
            if len(buf_val) >= BUFFER_SIZE:
                idx_val, buf_val = write_buffer_to_memmap(arr_val, buf_val, idx_val, val_target)
        else:
            if idx_train < train_target:
                buf_train.extend(ids)
                if len(buf_train) >= BUFFER_SIZE:
                    idx_train, buf_train = write_buffer_to_memmap(arr_train, buf_train, idx_train, train_target)
            else:
                break
                
        pbar.update(n_tok)
        
        if n_docs % 1000 == 0:
            pbar.write(f"[{time.strftime('%H:%M:%S')}] Processed {n_docs:,} docs | val={idx_val:,} | train={idx_train:,}")
        
        if idx_val >= val_target and idx_train >= train_target:
            break

    # Flush remaining buffers
    idx_val, _ = write_buffer_to_memmap(arr_val, buf_val, idx_val, val_target)
    idx_train, _ = write_buffer_to_memmap(arr_train, buf_train, idx_train, train_target)
    
    pbar.close()

    # Step 4: Finalize Files
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 4: Finalizing files...", flush=True)
    
    del arr_val
    del arr_train
    
    for path, final_idx in [(val_path, idx_val), (train_path, idx_train)]:
        actual_size = final_idx * 2
        try:
            with open(path, "r+b") as f:
                f.truncate(actual_size)
        except Exception as e:
            print(f"Truncate warning for {path}: {e}", flush=True)
            
    print(f"   Val:   {idx_val:,} tokens ({os.path.getsize(val_path)/1e6:.2f} MB)", flush=True)
    print(f"   Train: {idx_train:,} tokens ({os.path.getsize(train_path)/1e9:.2f} GB)", flush=True)

    # Save Metadata
    meta = {
        "target_total_tokens": TARGET_TOTAL_TOKENS,
        "actual_train_tokens": int(idx_train),
        "actual_val_tokens": int(idx_val),
        "language_config": LANG_CONFIG,
        "tokenizer": ENCODING,
        "seed": SEED,
        "add_eos": eos_id is not None,
        "eos_token_id": eos_id,
        "debug_mode": DEBUG_MODE
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"\n[{time.strftime('%H:%M:%S')}] Done!", flush=True)

if __name__ == "__main__":
    main()