import torch
from src.summarizer.model_seq2seq import Encoder

def main():
    B, T = 2, 10
    vocab_size = 100
    emb_dim = 128
    hidden_dim = 256

    x = torch.randint(0, vocab_size, (B, T))
    lengths = torch.tensor([10, 7])

    encoder = Encoder(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        pad_id=0,
    )

    outputs, h, c = encoder(x, lengths)

    print("Encoder outputs:", outputs.shape)
    print("Hidden state:", h.shape)
    print("Cell state:", c.shape)

if __name__ == "__main__":
    main()
