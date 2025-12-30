#feature_builder.py

from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def normalize_sentence(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s([.,;:])', r'\1', s)
    s = re.sub(r'[^a-z0-9.,;:()\- ]', '', s)
    return s.strip()

def build_tfidf_vectorizer(corpus: List[str], max_features: int = 20000):
    corpus_norm = [normalize_sentence(s) for s in corpus]
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf.fit(corpus_norm)
    return tfidf

def sentence_tfidf_scores(sentences: List[str], tfidf):
    if not sentences:
        return np.array([])
    norm_sents = [normalize_sentence(s) for s in sentences]
    mat = tfidf.transform(norm_sents)
    scores = np.asarray(mat.sum(axis=1)).ravel()
    if scores.max() > 0:
        scores /= scores.max()
    return scores

def position_score(index: int, total: int, first_k=3, last_k=3):
    if index < first_k:
        return 1.0
    if index >= total - last_k:
        return 0.9
    mid = total / 2
    return max(0.1, 1.0 - abs(index - mid) / mid)

def length_score(sent: str, min_len=8, max_len=60):
    ln = len(sent.split())
    if ln < min_len:
        return ln / min_len
    if ln > max_len:
        return max_len / ln
    return 1.0

DEFAULT_ROLE_WEIGHTS = {
    "RATIO": 3.0, "ANALYSIS": 2.5, "ISSUE": 2.2, "FAC": 1.5,
    "PREAMBLE": 0.3, "NONE": 0.5
}

def get_role_weights(sentences, role_predictor=None):
    if role_predictor:
        labels = role_predictor(sentences)
        return [DEFAULT_ROLE_WEIGHTS.get(l.upper(), 1.0) for l in labels]
    return [1.0] * len(sentences)

def compute_combined_scores(
    sentences: List[str],
    tfidf,
    role_predictor=None,
    weights=(0.5, 0.3, 0.2)
):
    tfidf_scores = sentence_tfidf_scores(sentences, tfidf)
    role_w = get_role_weights(sentences, role_predictor)
    combined = {}

    tfidf_w, role_wt, pos_w = weights
    total = len(sentences)
    max_role = max(role_w) if role_w else 1.0

    for i, s in enumerate(sentences):
        score = (
            tfidf_w * tfidf_scores[i]
            + role_wt * (role_w[i] / max_role)
            + pos_w * position_score(i, total) * length_score(s)
        )
        combined[i] = float(score)

    return combined
