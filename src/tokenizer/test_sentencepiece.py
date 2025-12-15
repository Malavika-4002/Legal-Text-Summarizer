import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("models/tokenizer/spm.model")

text = "<s> The court considered the application </s>"
ids = sp.encode(text)
decoded = sp.decode(ids)

print("Original:", text)
print("Token IDs:", ids)
print("Decoded :", decoded)
