import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring

from src.summarizer.sentence_splitter import split_into_sentences
from src.summarizer.feature_builder import build_tfidf_vectorizer, compute_combined_scores
from src.summarizer.extractive_ranker import mmr_select, get_sentence_vectors_tf


def extractive_summary(text, tfidf):
    sents = split_into_sentences(text)
    if not sents:
        return ""

    n = len(sents)
    top_k = min(10, max(4, n // 5))

    scores = compute_combined_scores(sents, tfidf)
    vecs = get_sentence_vectors_tf(sents, tfidf)
    chosen = mmr_select(sents, vecs, scores, top_k=top_k, diversity=0.35)

    return " ".join(sents[i] for i in sorted(chosen))


def evaluate(df, tfidf):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    agg = scoring.BootstrapAggregator()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pred = extractive_summary(row["Text"], tfidf)
        ref = str(row["Summary"])
        agg.add_scores(scorer.score(ref, pred))

    return agg.aggregate()


# -------- PRESENTATION ONLY (NEW) --------
def print_rouge_table(results):
    print("\n================ EXTRACTIVE ROUGE SCORES ================\n")
    print(f"{'Metric':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)

    for metric in ["rouge1", "rouge2", "rougeL"]:
        mid = results[metric].mid
        print(
            f"{metric.upper():<10} "
            f"{mid.precision:>10.3f}"
            f"{mid.recall:>10.3f}"
            f"{mid.fmeasure:>10.3f}"
        )

    print("\n========================================================\n")
# -----------------------------------------


def main(csv):
    df = pd.read_csv(csv)

    corpus = []
    for txt in df["Text"].astype(str):
        corpus.extend(split_into_sentences(txt))

    tfidf = build_tfidf_vectorizer(corpus)

    results = evaluate(df, tfidf)
    print_rouge_table(results)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/raw/summary.csv")
    args = p.parse_args()
    main(args.csv)
