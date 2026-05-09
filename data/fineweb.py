import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, r'C:\\Users\\zym\\AppData\\Local\\pylibs')
from modelscope import AutoTokenizer

DATA_DIR = 'your data download path....'
OUTPUT_PATH = "E:/project/gpt/GPT/data/fineweb"
TOKENIZER_NAME = "01-ai/Yi-6B"
TRAIN_RATIO = 0.9
MAX_LEN = 4095

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
eos_id = tokenizer.eos_token_id
print(f"EOS token id: {eos_id}")

# 获取所有parquet文件
parquet_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
print(f"Found {len(parquet_files)} files")

# 第一次遍历：统计token数
print("\nCounting tokens...")
total_tokens = 0
for filename in parquet_files:
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_parquet(filepath)
    for text in df["text"]:
        if not isinstance(text, str) or len(text) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        #if len(ids) > MAX_LEN:
            # continue
        total_tokens += len(ids) + 1  # +1 for EOS
    print(f"  {filename}: {total_tokens} tokens so far")

print(f"\nTotal tokens: {total_tokens}")
split_idx = int(total_tokens * TRAIN_RATIO)
print(f"Split: train={split_idx}, val={total_tokens - split_idx}")

# 第二次遍历：写入数据
print("\nWriting data...")
train_file = open(os.path.join(OUTPUT_PATH, "fineweb_train.bin"), "wb")
val_file = open(os.path.join(OUTPUT_PATH, "fineweb_val.bin"), "wb")

val_needed = total_tokens - split_idx
val_count = 0

for filename in parquet_files:
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_parquet(filepath)
    print(f"Processing {filename}...")

    for text in df["text"]:
        if not isinstance(text, str) or len(text) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > MAX_LEN:
            continue
        ids.append(eos_id)
        tokens = np.array(ids, dtype=np.uint16)

        if val_count < val_needed:
            n = min(len(tokens), val_needed - val_count)
            val_file.write(tokens[:n].tobytes())
            train_file.write(tokens[n:].tobytes())
            val_count += n
        else:
            train_file.write(tokens.tobytes())

    print(f"  val_count: {val_count}/{val_needed}")

train_file.close()
val_file.close()
print("\nDone!")