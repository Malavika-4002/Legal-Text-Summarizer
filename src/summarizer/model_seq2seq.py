#model_seq2seq.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# =========================================================
# Encoder
# =========================================================
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=pad_id
        )

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def forward(self, input_ids, lengths):
        """
        input_ids: (B, src_len)
        lengths:   (B,)
        """

        embedded = self.embedding(input_ids)  # (B, T, E)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_outputs, (h_n, c_n) = self.lstm(packed)

        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )
        # outputs: (B, T, H)

        return outputs, h_n, c_n


# =========================================================
# Bahdanau (Additive) Attention
# =========================================================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.attn_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,     # (B, H)
        encoder_outputs: torch.Tensor,    # (B, T, H)
        mask: torch.Tensor | None = None
    ):
        B, T, H = encoder_outputs.size()

        dec = decoder_hidden.unsqueeze(1).expand(B, T, H)

        energy = torch.tanh(
            self.attn_enc(encoder_outputs) +
            self.attn_dec(dec)
        )

        scores = self.attn_v(energy).squeeze(-1)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=1)  # (B, T)

        context = torch.bmm(
            attn_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)  # (B, H)

        return context, attn_weights


# =========================================================
# Decoder with Attention
# =========================================================
class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        attention: BahdanauAttention,
        pad_id: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # REQUIRED attributes (were missing before)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=pad_id
        )

        self.attention = attention

        self.lstm = nn.LSTM(
            input_size=emb_dim + hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        """
        input_token: (B,)
        hidden,cell: (1, B, H)
        encoder_outputs: (B, T, H)
        """

        embedded = self.dropout(
            self.embedding(input_token).unsqueeze(1)
        )  # (B, 1, E)

        context, attn_weights = self.attention(
            hidden.squeeze(0),
            encoder_outputs
        )

        context = context.unsqueeze(1)  # (B, 1, H)

        lstm_input = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.lstm(
            lstm_input, (hidden, cell)
        )

        output = output.squeeze(1)
        context = context.squeeze(1)

        logits = self.fc_out(
            torch.cat((output, context), dim=1)
        )

        return logits, hidden, cell, attn_weights


# =========================================================
# Seq2Seq Wrapper (with Teacher Forcing)
# =========================================================
class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: DecoderWithAttention,
        pad_id: int,
        device: torch.device
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.device = device

        # Sanity check
        assert encoder.lstm.hidden_size == decoder.hidden_dim

    def forward(
        self,
        src_ids,          # (B, src_len)
        src_lengths,      # (B,)
        tgt_ids,          # (B, tgt_len)
        teacher_forcing_ratio: float = 0.5
    ):
        B, tgt_len = tgt_ids.shape
        vocab_size = self.decoder.vocab_size

        encoder_outputs, h_n, c_n = self.encoder(
            src_ids,
            src_lengths
        )

        input_token = tgt_ids[:, 0]  # <bos>
        hidden = h_n
        cell = c_n

        outputs = torch.zeros(
            B, tgt_len, vocab_size,
            device=self.device
        )

        for t in range(1, tgt_len):
            logits, hidden, cell, _ = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs
            )

            outputs[:, t] = logits

            use_teacher = random.random() < teacher_forcing_ratio
            top_pred = logits.argmax(1)

            input_token = tgt_ids[:, t] if use_teacher else top_pred

        return outputs
