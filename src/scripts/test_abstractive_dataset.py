#src/scripts/test_abstractive_dataset.py

from torch.utils.data import DataLoader
from src.summarizer.abstractive_dataset import (
    AbstractiveSummaryDataset,
    abstractive_collate_fn
)

DATASET_PATH = "data/processed/abstractive_dataset.jsonl"
SP_MODEL_PATH = "models/tokenizer/spm.model"

dataset = AbstractiveSummaryDataset(
    jsonl_path=DATASET_PATH,
    sp_model_path=SP_MODEL_PATH,
    max_src_len=512,
    max_tgt_len=150
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda b: abstractive_collate_fn(b, dataset.pad_id)
)

batch = next(iter(loader))

print("SRC IDS SHAPE :", batch["src_ids"].shape)
print("SRC LENGTHS  :", batch["src_lengths"])
print("TGT IDS SHAPE:", batch["tgt_ids"].shape)

print("\nSample SRC IDs:", batch["src_ids"][0][:20])
print("Sample TGT IDs:", batch["tgt_ids"][0][:20])
