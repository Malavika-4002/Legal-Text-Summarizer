import torch
from src.summarizer.model_seq2seq import (
    Encoder,
    DecoderWithAttention,
    BahdanauAttention,
    Seq2Seq
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB = 8000
EMB = 256
HID = 256
PAD = 0

B = 2
SRC_LEN = 12
TGT_LEN = 8

# -------- Model --------
encoder = Encoder(VOCAB, EMB, HID).to(DEVICE)

attention = BahdanauAttention(HID).to(DEVICE)

decoder = DecoderWithAttention(
    vocab_size=VOCAB,
    emb_dim=EMB,
    hidden_dim=HID,
    attention=attention,
    pad_id=PAD
).to(DEVICE)

model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    pad_id=PAD,
    device=DEVICE
).to(DEVICE)

# -------- Dummy data --------
src = torch.randint(1, VOCAB, (B, SRC_LEN)).to(DEVICE)
src_lengths = torch.tensor([SRC_LEN, SRC_LEN - 2]).to(DEVICE)
tgt = torch.randint(1, VOCAB, (B, TGT_LEN)).to(DEVICE)

# -------- Forward pass --------
out = model(
    src,
    src_lengths,
    tgt,
    teacher_forcing_ratio=1.0
)

print("Seq2Seq output:", out.shape)
