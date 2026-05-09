# GPT Code Pre-Training

A GPT language model implemented from scratch in PyTorch, pre-trained on multi-language code data.

## Model Architecture

Implemented from scratch. Model configuration:

| Parameter  | Value   |
|------------|---------|
| n_layer    | 16      |
| n_head     | 16      |
| n_embd     | 1024    |
| block_size | 1024    |
| vocab_size | 64000   |

**Key features:**
- FlashAttention support (with fallback to standard attention)
- Weight tying between token embeddings and language model head
- 8-bit AdamW optimizer support (via bitsandbytes)
- Multi-GPU distributed training (DDP)

## Model Size

| Metric        | Value   |
|---------------|---------|
| Parameters    | 270M    |
| Memory (fp32) | 1.07 GB |

Memory calculation:
```python
total_num = sum(p.numel() for p in self.parameters())
dtype = next(self.parameters()).dtype
dtype_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 1
memery_size = total_num * dtype_size / 1e9
```

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

This downloads code data from HuggingFace starcoderdata, tokenizes it with 01-ai/Yi-6B tiktoken encoder, and saves binary `.bin` files.

**Supported languages and ratios:**

| Language    | Ratio |
|-------------|-------|
| python      | 0.25  |
| c           | 0.20  |
| javascript  | 0.15  |
| go          | 0.20  |
| c++         | 0.20  |

## Pre-Training

```bash
python train.py
```

**Key hyperparameters (in `train.py`):**
- Max learning rate: `6e-4` (cosine decay to `6e-5`)
- Warmup steps: 600
- Max steps: 6000
- Batch size: 32
- Block size: 1024
- Weight decay: 0.1
- Gradient clip: 1.0
- Dropout: 0.0
- Mixed precision: bfloat16 (or float16 if bf16 unsupported)
- Square Root Scaling Rule

**Features:**
- Gradient accumulation (total batch ~524k tokens)
- Automatic checkpoint saving (best val loss)
- Resume training from checkpoint
- MFU (Model FLOPs Utilization) monitoring
- Wandb logging (set `wandb_log = True`)

### Pre-Training Data

**Code data:**
```bash
python data/stack-v2-train-smol.py
```

**English data:** `fineweb-CC-MAIN-2024-10-1B-en`
```bash
python data/fineweb.py
```

**Chinese data:** `wikipedia-zh-cn-20260201`
```bash
python data/wiki.py
```

### Pre-Training Result

![Training loss curve](image/image.png)

Evaluation schedule: Starting from step 2600, train/val loss evaluated every 200 steps.

| Steps   | Observation                                          |
|---------|------------------------------------------------------|
| 2600-2800 | Both losses decreased with gap < 0.2 → continued training |
| 3000 | train: 2.2794, val: 2.4493, gap: 0.170 — val loss plateaued, applied regularization (dropout + hyperparameter adjustment) |
| 3000-3200 | Both losses declined; gap reduced to 0.113 → strong learning signal |
| 3200-3400 | Train loss decreased but val loss increased → overfitting |

→ Training stopped at step 3200; checkpoint with best generalization retained.

## Inference / Sampling

```bash
python sample.py
```

Generates text from the trained model checkpoint (`out/base.pt`).

**Parameters (in `sample.py`):**
- `temperature`: 0.9 (controls randomness)
- `top_k`: 200 (limits vocabulary)
- `max_new_tokens`: 500
- `num_samples`: 10

**Example output:**
```cpp
#include <memory>

using namespace std;
template<class T>
void Copy<T>::operator new(T&& other) {
        other.s = m_s;
        other.a = 1.0;
        other.b = 1.0;
        other.c = 0.03;
        for(int i = 0; i < m_s; i++)
        {
                m_s[i] = other[i];
                m_s[i] = m_s[i];
        }
}
```

```python
# #### data of size
# def data():
#     s = 96
#     for i, text in enumerate(data.split(';'), 96):
#         all_words_in_words[i] = total_sum(text)

    return all_words_in_words


def extract_features(features_dict):
    """
    extract features and extract mean and use them for all words
    """
    labels = labels.split(';')
    # extract mean and use them for all words
    features = []
    for text in features:
        l = len(text)
        mean = 0
        counts = max(counts, 0)
        # extract counts and use them for all words
        counts += l * counts + mean
        labels.append(l)
        labels.append(lx.Label())

    labels = labels.split(',')
    # extract mean and use them for all words
    labels = labels.split(';')
    if len(labels) == 1:
        labels = labels.split(',')

    features = []
    for text in labels:
        l = len(text)
        mean = 0
        counts = max(counts, 0)
        labels.append(l)
        labels.append(lx.Label())
        labels.append(l)
        labels.append(lx.Label())

    _, _, _ = s.get_num_words()
    words = torch.maximum(100, words.max(100), dim=-1).unsqueeze(0).expand(0)
    features = torch.cat(features, l)
    labels = torch.cat(labels, dim=-1).
```

## Remaining Issues

**MFU Degradation During Training:**

Training一段时间后，MFU 下降，每步训练时间增加：

| Iteration | Loss    | Time (ms) | MFU (%) |
|-----------|---------|-----------|---------|
| 2605      | 2.2898  | 5659.88   | 53.55   |
| 2606      | 2.4253  | 5680.17   | 53.54   |
| 2607      | 1.9189  | 5765.15   | 53.44   |
| 2608      | 2.2020  | 5922.03   | 53.21   |
| 2609      | 2.3840  | 6010.25   | 52.94   |
| 2610      | 2.3588  | 6197.13   | 52.53   |
| 2611      | 2.5115  | 6325.67   | 52.07   |
| 2612      | 2.4407  | 6556.29   | 51.49   |
| 2613      | 2.4101  | 6498.14   | 51.00   |
| 2614      | 2.4459  | 6627.08   | 50.48   |
| 2615      | 2.2298  | 6690.62   | 49.96   |

**Problem:** 随着训练进行，每 step 的计算效率（MFU）逐渐下降，训练时间相应增加。

## License

MIT License
