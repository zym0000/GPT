# GPT Code Pre-Training

A GPT language model implemented from scratch in PyTorch, pre-trained on multi-language code data.

## Project Structure

```
GPT/
├── model.py              # GPT model architecture (from scratch)
├── train.py              # Training script
├── sample.py             # Text generation / inference script
├── input.txt             # Simple text sample (for DataLoad testing)
├── data/
│   ├── code_pretrain_data/
│   │   ├── train.bin     # Training data (binary, tokenized)
│   │   ├── val.bin       # Validation data (binary, tokenized)
│   │   └── metadata.json  # Dataset metadata
│   └── stack-v2-train-smol.py  # Data preprocessing pipeline
└── out/
    └── ckpt.pt           # Model checkpoint (after training)
```

## Model Architecture

Implemented from scratch, matching GPT-2 configuration:

| Parameter    | Value  |
|--------------|--------|
| n_layer      | 12     |
| n_head       | 12     |
| n_embd       | 768    |
| block_size   | 1024   |
| vocab_size   | 50304  |

**Key features:**
- FlashAttention support (with fallback to standard attention)
- Weight tying between token embeddings and language model head
- 8-bit AdamW optimizer support (via bitsandbytes)
- Multi-GPU distributed training (DDP)

## Setup

```bash
pip install torch numpy tiktoken
```

For 8-bit optimizer:
```bash
pip install bitsandbytes
```

For data preprocessing (optional):
```bash
pip install modelscope datasets tqdm
```

## Data Preprocessing

```bash
python data/stack-v2-train-smol.py
```

This downloads code data from HuggingFace starcoderdata, tokenizes it with GPT-2's tiktoken encoder, and saves binary `.bin` files.

**Supported languages and ratios:**
- Python: 35%
- Java: 20%
- Go: 10%
- JavaScript: 10%
- C++: 10%
- Rust: 5%
- TypeScript: 5%
- SQL: 5%

## Training

```bash
python train.py
```

**Key hyperparameters (in `train.py`):**
- Max learning rate: `6e-4` (cosine decay to `6e-5`)
- Warmup steps: 2000
- Max steps: 600000
- Batch size: 16
- Block size: 1024
- Weight decay: 0.1
- Gradient clip: 1.0
- Mixed precision: bfloat16 (or float16 if bf16 unsupported)

**Features:**
- Gradient accumulation (total batch ~524k tokens)
- Automatic checkpoint saving (best val loss)
- Resume training from checkpoint
- MFU (Model FLOPs Utilization) monitoring
- Wandb logging (set `wandb_log = True`)

## Inference / Sampling

```bash
python sample.py
```

Generates text from the trained model checkpoint (`out/ckpt.pt`).

**Parameters (in `sample.py`):**
- `temperature`: 0.9 (controls randomness)
- `top_k`: 200 (limits vocabulary)
- `max_new_tokens`: 500
- `num_samples`: 10

**Example output:**

```java
switch (g) {
    case G_DEFAULT_TARGET:
        goto IL_00106;
        return 0;
    case G_REQUIRED:
        goto IL_00103;
        return 0;
    // ...
}

public static O = 0 as Vector2f;
public static O *= 0 -> Vector2f;
public static O *= 0 -> (Vector2f + Vector2f) : Vector2f;
// ...
```

The model is pre-training on mixed code from 8 languages (Python, Java, JavaScript, TypeScript, Go, C++, Rust, SQL). Output quality depends on training steps and data scale.

## License

MIT License
