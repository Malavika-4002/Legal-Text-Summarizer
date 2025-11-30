from src.sbd.inference import get_sentences

text = "The court ruled in favor of Mr. Smith on Jan 5. This is a new sentence."
print("--- Testing Phase 1 ---")
sentences = get_sentences(text)
for i, s in enumerate(sentences):
    print(f"[{i}] {s}")