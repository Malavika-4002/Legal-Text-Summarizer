#predict.py

from src.summarizer.sentence_splitter import split_into_sentences
from src.summarizer.feature_builder import build_tfidf_vectorizer, compute_combined_scores
from src.summarizer.extractive_ranker import mmr_select, get_sentence_vectors_tf

def extractive_predict(text, tfidf_vectorizer, role_predictor=None):
    sents = split_into_sentences(text)
    if not sents:
        return ""

    n = len(sents)
    top_k = min(10, max(4, n // 5))

    scores = compute_combined_scores(sents, tfidf_vectorizer, role_predictor)
    vecs = get_sentence_vectors_tf(sents, tfidf_vectorizer)
    chosen = mmr_select(sents, vecs, scores, top_k=top_k, diversity=0.35)

    return " ".join(sents[i] for i in sorted(chosen))
