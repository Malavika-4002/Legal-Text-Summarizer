#src/summarizer/abstractive_dataset.py

import json
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from typing import List, Dict


class AbstractiveSummaryDataset(Dataset):
    """
    Dataset for extract-then-abstract legal summarization.

    Each sample:
        input  -> extractive-compressed legal text
        target -> gold abstractive summary

    Output per item:
        src_ids  : LongTensor
        src_len  : int
        tgt_ids  : LongTensor
    """

    def __init__(
        self,
        jsonl_path: str,
        sp_model_path: str,
        max_src_len: int | None = None,
        max_tgt_len: int | None = None,
    ):
        self.samples = []
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Load SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)

        # 🔒 FIXED SPECIAL TOKEN IDS (DO NOT USE sp.bos_id())
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        # Load JSONL dataset
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append({
                    "input": obj["input"],
                    "target": obj["target"]
                })

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text: str, max_len: int | None):
        """
        Encode text using SentencePiece and add BOS/EOS.
        Applies optional truncation.
        """
        ids = self.sp.encode(text, out_type=int)

        # Add BOS and EOS explicitly
        ids = [self.bos_id] + ids + [self.eos_id]

        if max_len is not None:
            ids = ids[:max_len]
            if ids[-1] != self.eos_id:
                ids[-1] = self.eos_id

        return ids

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        src_ids = self.encode_text(
            sample["input"], self.max_src_len
        )
        tgt_ids = self.encode_text(
            sample["target"], self.max_tgt_len
        )

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "src_len": len(src_ids),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }


# =========================================================
# Collate Function (Batch Padding)
# =========================================================

def abstractive_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int
):
    """
    Pads variable-length sequences for batching.
    """

    batch_size = len(batch)

    src_lens = [item["src_len"] for item in batch]
    max_src = max(src_lens)
    max_tgt = max(len(item["tgt_ids"]) for item in batch)

    src_batch = torch.full(
        (batch_size, max_src),
        pad_id,
        dtype=torch.long
    )

    tgt_batch = torch.full(
        (batch_size, max_tgt),
        pad_id,
        dtype=torch.long
    )

    for i, item in enumerate(batch):
        src_len = item["src_len"]
        tgt_len = len(item["tgt_ids"])

        src_batch[i, :src_len] = item["src_ids"]
        tgt_batch[i, :tgt_len] = item["tgt_ids"]

    return {
        "src_ids": src_batch,
        "src_lengths": torch.tensor(src_lens, dtype=torch.long),
        "tgt_ids": tgt_batch
    }
