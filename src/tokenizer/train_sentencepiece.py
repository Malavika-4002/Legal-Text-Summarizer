# src/tokenizer/train_sentencepiece.py

import sentencepiece as spm
import json
import os

INPUT_JSONL = "data/processed/abstractive_dataset.jsonl"
MODEL_PREFIX = "models/tokenizer/spm"
VOCAB_SIZE = 8000   # ✅ reduced vocab size (CRITICAL)


def load_training_texts(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            inp = obj.get("input", "")
            tgt = obj.get("target", "")
            texts.append(inp)
            texts.append(tgt)
    return texts


def save_tmp_corpus(texts, out_path="data/processed/tmp_corpus.txt"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print("Saving temporary training corpus...")
    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            t = t.replace("\n", " ").strip()
            f.write(t + "\n")
    return out_path


def train_tokenizer():
    print("Loading training texts from:", INPUT_JSONL)
    texts = load_training_texts(INPUT_JSONL)

    corpus_path = save_tmp_corpus(texts)

    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,

        # ✅ REQUIRED FOR SEQ2SEQ
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        # ✅ REQUIRED FOR YOUR DATASET
        user_defined_symbols=["<s>", "</s>"]
    )

    print("Tokenizer training completed.")
    print("Files created:")
    print(f" - {MODEL_PREFIX}.model")
    print(f" - {MODEL_PREFIX}.vocab")


if __name__ == "__main__":
    train_tokenizer()
