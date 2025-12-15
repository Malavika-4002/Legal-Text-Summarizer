import torch
from src.summarizer.attention import BahdanauAttention

def main():
    B, T, H = 2, 10, 256

    encoder_outputs = torch.randn(B, T, H)
    decoder_hidden = torch.randn(B, H)

    attention = BahdanauAttention(hidden_dim=H)

    context, weights = attention(decoder_hidden, encoder_outputs)

    print("Context:", context.shape)
    print("Attention weights:", weights.shape)
    print("Attention sum (per batch):", weights.sum(dim=1))

if __name__ == "__main__":
    main()
