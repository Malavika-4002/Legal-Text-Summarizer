# src/scripts/train_abstractive.py

import os
import sys
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm

# Fix Python Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

# 1. FIXED IMPORT: Changed AbstractiveDataset to AbstractiveSummaryDataset
from src.summarizer.model_seq2seq import (
    Encoder,
    DecoderWithAttention,
    BahdanauAttention,
    Seq2Seq
)
from src.summarizer.abstractive_dataset import AbstractiveSummaryDataset, abstractive_collate_fn

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EMB_DIM = 256
HIDDEN_DIM = 256
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
TEACHER_FORCING_RATIO = 0.5

MAX_SRC_LEN = 512
MAX_TGT_LEN = 150

# 2. FIXED PATH: Match your directory structure
DATA_PATH = "data/processed/abstractive_dataset.jsonl"
TOKENIZER_PATH = "models/tokenizer/spm.model" 
CHECKPOINT_DIR = "models/seq2seq_summarizer" # renamed from t5 as you are using Seq2Seq

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.manual_seed(42)
random.seed(42)

# LOAD TOKENIZER
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)
VOCAB_SIZE = sp.get_piece_size()
PAD_ID = 0 # Match your dataset.py

print(f"Vocabulary size: {VOCAB_SIZE}")

# 3. FIXED INITIALIZATION: Pass the path string, not the object 'sp'
train_dataset = AbstractiveSummaryDataset(
    jsonl_path=DATA_PATH,
    sp_model_path=TOKENIZER_PATH,
    max_src_len=MAX_SRC_LEN,
    max_tgt_len=MAX_TGT_LEN
)

# 4. FIXED COLLATE: Use the standalone function 'abstractive_collate_fn'
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: abstractive_collate_fn(b, PAD_ID)
)

print(f"Training samples: {len(train_dataset)}")

# MODEL INITIALIZATION
encoder = Encoder(
    vocab_size=VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    pad_id=PAD_ID
)

attention = BahdanauAttention(hidden_dim=HIDDEN_DIM)

decoder = DecoderWithAttention(
    vocab_size=VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    attention=attention,
    pad_id=PAD_ID
)

model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    pad_id=PAD_ID,
    device=DEVICE
).to(DEVICE)

# LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TRAINING LOOP
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in progress:
        src_ids = batch["src_ids"].to(DEVICE)
        src_lengths = batch["src_lengths"].to(DEVICE)
        tgt_ids = batch["tgt_ids"].to(DEVICE)

        optimizer.zero_grad()

        # Output shape: [B, TGT_LEN, VOCAB_SIZE]
        outputs = model(
            src_ids=src_ids,
            src_lengths=src_lengths,
            tgt_ids=tgt_ids,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )

        # We skip the first token (<BOS>) for loss calculation
        loss = criterion(
            outputs[:, 1:].reshape(-1, VOCAB_SIZE),
            tgt_ids[:, 1:].reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch} | Average Loss: {avg_loss:.4f}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"seq2seq_epoch_{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)

print("Training completed.")