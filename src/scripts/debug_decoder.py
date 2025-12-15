import torch
from src.summarizer.model_seq2seq import Encoder, BahdanauAttention, DecoderWithAttention

BATCH = 2
SRC_LEN = 10
VOCAB_SIZE = 8000
EMB_DIM = 128
HIDDEN_DIM = 256

pad_id = 0

# Dummy input
src = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
lengths = torch.tensor([10, 7])
input_token = torch.randint(1, VOCAB_SIZE, (BATCH,))

# Models
encoder = Encoder(
    vocab_size=VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    pad_id=pad_id
)

attention = BahdanauAttention(HIDDEN_DIM)

decoder = DecoderWithAttention(
    vocab_size=VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    attention=attention,
    pad_id=pad_id
)

# Forward
encoder_outputs, h, c = encoder(src, lengths)

logits, h2, c2, attn = decoder(
    input_token, h, c, encoder_outputs
)

print("Logits:", logits.shape)
print("Hidden:", h2.shape)
print("Cell:", c2.shape)
print("Attention:", attn.shape)
print("Attention sum:", attn.sum(dim=1))
