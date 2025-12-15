#build_abstarctive_dataset.py

import json
import pandas as pd
from tqdm import tqdm

from src.summarizer.sentence_splitter import split_into_sentences
from src.summarizer.feature_builder import build_tfidf_vectorizer, compute_combined_scores
from src.summarizer.extractive_ranker import mmr_select, get_sentence_vectors_tf

INPUT_CSV = "data/raw/summarizer-train.csv"
OUTPUT_JSONL = "data/processed/abstractive_dataset.jsonl"

def generate_extracts():
    df = pd.read_csv(INPUT_CSV)

    all_sents = []
    for text in tqdm(df["Text"].astype(str)):
        all_sents.extend(split_into_sentences(text))

    tfidf = build_tfidf_vectorizer(all_sents)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = str(row["Text"])
            summary = str(row["Summary"])

            sents = split_into_sentences(text)
            if not sents:
                continue

            n = len(sents)
            top_k = min(10, max(4, n // 5))

            scores = compute_combined_scores(sents, tfidf)
            vecs = get_sentence_vectors_tf(sents, tfidf)
            selected = mmr_select(
                sents, vecs, scores, top_k=top_k, diversity=0.35
            )

            compressed = " ".join(
                f"<s> {sents[i]} </s>" for i in sorted(selected)
            )

            out.write(json.dumps({
                "input": compressed,
                "target": summary
            }) + "\n")

    print("Saved →", OUTPUT_JSONL)

if __name__ == "__main__":
    generate_extracts()
