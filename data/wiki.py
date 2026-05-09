import json
import numpy as np
import os

#由于网络原因，先把数据下载本地，在转成 预训练数据.bin格式
DATA_PATH = "your download local path ...."
OUTPUT_PATH = "E:/project/gpt/GPT/data/code_pretrain_data"
TOKENIZER_NAME = "01-ai/Yi-6B"
TRAIN_RATIO = 0.9

import sys
# 如需本地 pylibs，保留；否则可注释
sys.path.insert(0, r'C:\\Users\\zym\\AppData\\Local\\pylibs')

from modelscope import AutoTokenizer

def process_file_stream(filepath, tokenizer):
    eos_id = tokenizer.eos_token_id
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            if len(text) < 50:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(eos_id)
            yield np.array(ids, dtype=np.uint16)

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"EOS token id: {tokenizer.eos_token_id}")

    print("Counting tokens...")
    total_tokens = 0
    for filename in ['wikipedia-zh-cn-20260201.json']:
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_PATH, filename)
            with open(filepath, "r",encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if len(item.get("text", "")) >= 50:
                        total_tokens += len(tokenizer.encode(item["text"], add_special_tokens=False)) + 1

    print(f"Total: {total_tokens} tokens")
    split_idx = int(total_tokens * TRAIN_RATIO)

    train_file = open(os.path.join(OUTPUT_PATH, "wiki_train.bin"), "wb")
    val_file = open(os.path.join(OUTPUT_PATH, "wiki_val.bin"), "wb")
    val_needed = total_tokens - split_idx
    val_count = 0

    for filename in ['wikipedia-zh-cn-20260201.json']:
        if filename.endswith(".json"):
            for tokens in process_file_stream(os.path.join(DATA_PATH, filename), tokenizer):
                if val_count < val_needed:
                    n = min(len(tokens), val_needed - val_count)
                    val_file.write(tokens[:n].tobytes())
                    train_file.write(tokens[n:].tobytes())
                    val_count += n
                else:
                    train_file.write(tokens.tobytes())

    train_file.close()
    val_file.close()
    print("Done!")

if __name__ == "__main__":
    main()