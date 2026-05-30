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

    print(f"data_dir : {data_dir}")
    # 列出该目录下的 parquet
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    
    assert len(parquet_paths) != 0, f"No parquet files found in {data_dir}"

    # train/val 切分（最后一个文件做 val）
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    # Resume 状态
    #pg 取那个文件
    #rg pg下那个文件 row_id
    resume_pq_idx = resume_state_dict.get("pq_idx", 0) if resume_state_dict else 0
    resume_rg_idx = resume_state_dict.get("rg_idx") if resume_state_dict else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # 无限循环 epoch
        pq_idx = resume_pq_idx if first_pass else 0
        
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)

            # Resume 逻辑：如果是恢复点所在的文件，从恢复位置继续
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 避免重复
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # 只处理一次
            else:
                rg_idx = ddp_rank

            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column(column_name).to_pylist()

                # 清洗（如果是代码数据）
                if clean_fn:
                    batch = [clean_fn(t) for t in batch]
                    batch = [t for t in batch if t.strip()]

                # 按 tokenizer_batch_size 切分 yield
                for i in range(0, len(batch), tokenizer_batch_size):
                    sub_batch = batch[i:i+tokenizer_batch_size]
                    if sub_batch:  # 非空才 yield
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
        self.nl_ratio = 1.0 - code_ratio
        self.rng = random.Random(seed)
        self.tokenizer_batch_size = tokenizer_batch_size

        # 分别创建两个底层迭代器
        # resume_state_dict 结构示例：
        #   {"code": {"pq_idx": 0, "rg_idx": 5, "epoch": 1}, "nl": {"pq_idx": 1, "rg_idx": 3, "epoch": 2}}
        code_resume = resume_state_dict.get("code") if resume_state_dict else None
        nl_resume = resume_state_dict.get("nl") if resume_state_dict else None

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

    def __iter__(self):
        return self

    def __next__(self):
        """
        按 code_ratio 决定从哪个数据源取下一个 batch。
        两个底层迭代器都是无限的（各自循环 epoch），所以不会耗尽。
        """
        # 按比例采样
        if self.rng.random() < self.code_ratio:
            return next(self.code_iter)
        else:
            return next(self.nl_iter)

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