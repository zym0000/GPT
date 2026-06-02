import os
import re
import random
from typing import Optional, Callable, Dict, Any, Tuple, List
import pyarrow.parquet as pq
from common import get_ddp_info

def clean_starcoder_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    
    text = re.sub(
        r'^(?:<{1,2}reponame>[^<\n]*)?(?:<{1,2}filename>[^<\n]*)?(?:<{1,2}gh_stars>[^<\n]*)?\n?',
        '',
        text,
        count=1
    )
    
    text = re.sub(r'\|?<\|endoftext\|>\|?', '', text)
    
    return text

def _list_parquet_files(data_dir: str) -> List[str]:
    """列出目录下所有 parquet 文件，返回完整路径"""
    if not os.path.exists(data_dir):
        return []
    
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    return [os.path.join(data_dir, f) for f in files]


def _document_batches_single(
    split: str,
    resume_state_dict: Optional[Dict[str, Any]],
    tokenizer_batch_size: int,
    data_dir: str,
    column_name: str = "text",
    clean_fn: Optional[Callable[[str], str]] = None,
    source_name: str = "default",
    seed: int = 42,
):
    
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_ddp_info()

    print(f"data_dir : {data_dir}, split : {split}")
    
    # 列出 train 和 val 的 parquet
    train_paths = _list_parquet_files(data_dir)
    val_dir = os.path.join(data_dir, "val")
    val_paths = _list_parquet_files(val_dir)

    # 根据 split 选择路径
    if split == "train":
        paths = train_paths
        assert len(paths) != 0, f"No train parquet files found in {data_dir}"
    else:
        paths = val_paths
        assert len(paths) != 0, f"No val parquet files found in {val_dir}"

    # Resume 状态：val 不支持 resume
    if split == "val":
        resume_pq_idx = 0
        resume_rg_idx = None
        resume_epoch = 1
    else:
        resume_pq_idx = resume_state_dict.get("pq_idx", 0) if resume_state_dict else 0
        resume_rg_idx = resume_state_dict.get("rg_idx") if resume_state_dict else None
        resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict else 1
    
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        
        while pq_idx < len(paths):
            filepath = paths[pq_idx]
            pf = pq.ParquetFile(filepath)

            # Resume 逻辑：train 时处理，val 时跳过
            if split == "train" and first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank

            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column(column_name).to_pylist()

                if clean_fn:
                    batch = [clean_fn(t) for t in batch]
                    batch = [t for t in batch if t.strip()]

                for i in range(0, len(batch), tokenizer_batch_size):
                    sub_batch = batch[i:i+tokenizer_batch_size]
                    if sub_batch:
                        indices = {
                            "source": source_name,
                            "pq_idx": pq_idx,
                            "rg_idx": rg_idx,
                            "epoch": epoch,
                        }
                        yield sub_batch, indices
                
                rg_idx += ddp_world_size
            
            pq_idx += 1
        
        first_pass = False
        epoch += 1

class MixedDocumentBatches:
    def __init__(
        self,
        split: str,
        resume_state_dict: Optional[Dict[str, Any]],
        tokenizer_batch_size: int,
        code_dir: str,
        nl_dir: str,
        code_col: str = "content",
        nl_col: str = "text",
        code_ratio: float = 0.75,
        seed: int = 42,
    ):
        self.code_ratio = code_ratio
        self.rng = random.Random(seed)
        self.tokenizer_batch_size = tokenizer_batch_size

        # 恢复状态
        code_resume = resume_state_dict.get("code") if resume_state_dict else None
        nl_resume = resume_state_dict.get("nl") if resume_state_dict else None

        self.code_state = code_resume or {"pq_idx": 0, "rg_idx": None, "epoch": 1}
        self.nl_state = nl_resume or {"pq_idx": 0, "rg_idx": None, "epoch": 1}

        # 缓存池：上次多取但未用完的文本暂存这里
        self._code_buffer = []
        self._nl_buffer = []

        # 两个独立的文档迭代器
        self.code_iter = _document_batches_single(
            split=split,
            resume_state_dict=code_resume,
            tokenizer_batch_size=tokenizer_batch_size,
            data_dir=code_dir,
            column_name=code_col,
            clean_fn=clean_starcoder_text,
            source_name="code",
            seed=seed,
        )

        self.nl_iter = _document_batches_single(
            split=split,
            resume_state_dict=nl_resume,
            tokenizer_batch_size=tokenizer_batch_size,
            data_dir=nl_dir,
            column_name=nl_col,
            clean_fn=None,
            source_name="nl",
            seed=seed,
        )

    def _take(self, iterator, target_size, state_attr, buffer_attr):
        """
        从缓存 + 迭代器获取恰好 target_size 个文本。
        - iterator: 文档流（_document_batches_single 生成的迭代器）
        - target_size: 本次需要获取的文本条数
        - state_attr: 对应状态的属性名（"code_state" 或 "nl_state"）
        - buffer_attr: 对应缓存的属性名（"_code_buffer" 或 "_nl_buffer"）
        返回: 恰好 target_size 个文本的列表
        """
        texts = []
        buffer = getattr(self, buffer_attr)

        # 1. 先从缓存取
        if buffer:
            need = target_size - len(texts)
            take = min(need, len(buffer))
            texts.extend(buffer[:take])
            setattr(self, buffer_attr, buffer[take:])  # 更新缓存

        # 2. 缓存不够时继续从迭代器拉取
        last_indices = None
        while len(texts) < target_size:
            sub_batch, indices = next(iterator)
            texts.extend(sub_batch)
            last_indices = indices

        # 3. 更新状态：只有实际拉取了新数据才更新（否则保持旧状态）
        if last_indices is not None:
            setattr(self, state_attr, {
                "pq_idx": last_indices["pq_idx"],
                "rg_idx": last_indices["rg_idx"],
                "epoch": last_indices["epoch"],
            })

        # 4. 多余的部分存回缓存，确保不丢失文本
        if len(texts) > target_size:
            extra = texts[target_size:]
            setattr(self, buffer_attr, extra)
            texts = texts[:target_size]

        return texts

    def __iter__(self):
        return self

    def __next__(self):
        # 计算本次混合 batch 中两类数据的数量
        code_size = int(self.tokenizer_batch_size * self.code_ratio)
        nl_size = self.tokenizer_batch_size - code_size

        # 取出所需数量的文本（可能跨 batch，多余部分由 _take 内部缓存）
        code_texts = self._take(
            self.code_iter, code_size, "code_state", "_code_buffer"
        )
        nl_texts = self._take(
            self.nl_iter, nl_size, "nl_state", "_nl_buffer"
        )

        # 合并并记录每条文本的来源
        combined = code_texts + nl_texts
        text_sources = ["code"] * len(code_texts) + ["nl"] * len(nl_texts)

        # 随机打乱（让 code 和 nl 在 batch 内充分混合）
        zipped = list(zip(combined, text_sources))
        self.rng.shuffle(zipped)
        texts, shuffled_sources = zip(*zipped) if zipped else ([], [])

        # 构造返回的 indices，精简字段，带上 resume 状态和每条文本的来源
        merged_indices = {
            "source": "mixed",
            "text_sources": list(shuffled_sources),   # 顺序与 texts 一致
            "resume_state": {
                "code": self.code_state,
                "nl": self.nl_state,
            },
        }
        return list(texts), merged_indices

def mixed_document_batches(
    split: str,
    resume_state_dict: Optional[Dict[str, Any]],
    tokenizer_batch_size: int,
    code_dir: str,
    nl_dir: str,
    code_col: str = "content",
    nl_col: str = "text",
    code_ratio: float = 0.75,
    seed: int = 42,
):
    """
        code_dir, nl_dir: 两类数据的目录
        code_col, nl_col: 列名（代码默认 "content"，NL 默认 "text"）
        code_ratio: 代码数据占比（0.0 ~ 1.0）
    """
    mixer = MixedDocumentBatches(
        split=split,
        resume_state_dict=resume_state_dict,
        tokenizer_batch_size=tokenizer_batch_size,
        code_dir=code_dir,
        nl_dir=nl_dir,
        code_col=code_col,
        nl_col=nl_col,
        code_ratio=code_ratio,
        seed=seed,
    )

    yield from mixer