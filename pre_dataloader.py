import torch
import random

from dataset import mixed_document_batches

def sliding_window_split(tokens, chunk_size=2048, overlap=256):
    doc_len = len(tokens)
    if doc_len <= chunk_size:
        return [(tokens, 0)]
    
    start = 0
    chunks = []
    step = chunk_size - overlap
    while start < doc_len:
        end = min(start + chunk_size, doc_len)
        chunk = tokens[start:end]
        chunks.append((chunk, start))
        if end == doc_len:
            break
        start += step
    return chunks

def mark_overlap_tokens(chunk_length, chunk_start, total_size, overlap, overlap_weight=0.5):
    weights = [1.0] * chunk_length
    if total_size <= chunk_length:
        return weights
    
    # 前面重叠（防越界）
    if chunk_start > 0:
        for i in range(min(overlap, chunk_length)):
            weights[i] = overlap_weight
    
    # 后面重叠（防越界）
    if chunk_length + chunk_start < total_size:
        back_start = max(0, chunk_length - overlap)
        for i in range(back_start, chunk_length):
            weights[i] = overlap_weight

    return weights

def pre_train_data_loader_best_fit(
    tokenizer, B, T, split, device="cuda", buffer_size=1000,
    code_dir=None,
    nl_dir=None,
    code_col="content",
    nl_col="text",
    resume_state_dict=None,
    code_ratio=0.85,
    seed=42,
):
    FIM_PREFIX_ID, FIM_SUFFIX_ID, FIM_MIDDLE_ID = tokenizer.get_fim_ids()
    EOS = tokenizer.get_eos_id()

    def apply_fim_to_document(token, is_code=False, fim_rate=0.5):
        # 非代码数据,直接返回 token + EOS
        if not is_code or random.random() > fim_rate:
            return token + [EOS]
        
        # 代码数据做 FIM：PSM 和 SPMv2 各 50%
        use_psm = random.random() < 0.5
        length = len(token)
        if length < 10:
            return token + [EOS]
        
        split1 = random.randint(1, max(1, length // 3))
        min_split2 = min(split1 + length // 5, length - 1)
        min_split2 = max(min_split2, split1 + 1)
        split2 = random.randint(min_split2, length - 1)

        prefix = token[:split1]
        middle = token[split1:split2]
        suffix = token[split2:]

        if use_psm:
            # PSM: <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle <EOS>
            return (
                [FIM_PREFIX_ID] + prefix +
                [FIM_SUFFIX_ID] + suffix +
                [FIM_MIDDLE_ID] + middle +
                [EOS]
            )
        else:
            # SPMv2: <fim_prefix><fim_suffix> suffix <fim_middle> prefix+middle <EOS>
            return (
                [FIM_PREFIX_ID, FIM_SUFFIX_ID] + suffix + 
                [FIM_MIDDLE_ID] + prefix + middle +
                [EOS]
            )

    use_cuda = device == 'cuda'
    
    # 需要确保 mixed_document_batches 已在同模块或已 import
    document_iter = mixed_document_batches(
        split=split,
        resume_state_dict=resume_state_dict,      # 如需断点续训，从外部传入
        tokenizer_batch_size=1024,   # 每次读取一批原始文本
        code_dir=code_dir,
        nl_dir=nl_dir,
        code_col=code_col,
        nl_col=nl_col,
        code_ratio=code_ratio,
        seed=seed,
    )
    
    # 状态变量（用于保存 checkpoint）
    pq_idx, rg_idx, epoch = 0, 0, 1
    doc_buffer = []
    weights_buffer = []

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch

        doc_batch, indices = next(document_iter)
        
        # 解包状态（混合迭代器返回 dict）
        pq_idx = indices["pq_idx"]
        rg_idx = indices["rg_idx"]
        epoch = indices["epoch"]
        
        # 判断当前 batch 来源：代码 or NL
        is_code = (indices.get("source") == "code")
        # 分词
        token_list = tokenizer.get_doc_batch_tokens(doc_batch)
        for tokens in token_list:
            chunks = sliding_window_split(tokens, T - 4)
            total_size = len(tokens)
            for chunk_token, chunk_start in chunks:
                fim_token = apply_fim_to_document(chunk_token, is_code=is_code)
                doc_buffer.append(fim_token)

    row_capacity = T + 1
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B*T].view(B, T)
    cpu_targets = cpu_buffer[B*T:].view(B, T)
    inputs = gpu_buffer[:B*T].view(B, T)
    targets = gpu_buffer[B*T:].view(B, T)
    mask = torch.ones((B, row_capacity), dtype=torch.bool) 
    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                
                remaining = row_capacity - pos
                
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx,pos:pos+doc_len] = torch.tensor(doc, dtype=torch.long)
                    mask[row_idx,pos:pos+doc_len] = False
                    pos += doc_len
                else:
                    # 这里为了保证FIM完整性，不使用截断，使用填充 因为input 不能为 -1 -100这种特殊,所以填充0
                    #mask 用来标记是否是填充，来表示哪些不参与计算。
                    row_buffer[row_idx, pos:pos+remaining] = torch.full((remaining,),0,dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        targets_tmp = row_buffer[:, 1:].clone()
        targets_tmp[mask[:, 1:]] = -1
        cpu_targets.copy_(targets_tmp)
        
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict