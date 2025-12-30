#extractive_ranker.py

from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mmr_select(
    sentences: List[str],
    sentence_vectors,
    scores: Dict[int, float],
    top_k: int = 5,
    diversity: float = 0.35
):
    n = len(sentences)
    if n == 0:
        return []

    idxs = list(scores.keys())
    ranked_by_score = sorted(idxs, key=lambda i: scores[i], reverse=True)
    selected = []

    if not ranked_by_score:
        return selected

    # normalize scores
    sc_arr = np.array([scores[i] for i in idxs])
    if sc_arr.max() > 0:
        sc_arr = sc_arr / sc_arr.max()
    score_map = {i: sc_arr[idxs.index(i)] for i in idxs}

    if sentence_vectors is None or len(sentence_vectors) == 0:
        return ranked_by_score[:top_k]

    sims = cosine_similarity(sentence_vectors)
    unselected = set(idxs)

    first = ranked_by_score[0]
    selected.append(first)
    unselected.remove(first)

    while len(selected) < top_k and unselected:
        mmr_values = {}
        for j in unselected:
            sim_to_selected = max(sims[j][s] for s in selected)
            mmr_values[j] = (
                (1 - diversity) * score_map.get(j, 0)
                - diversity * sim_to_selected
            )
        pick = max(mmr_values.items(), key=lambda x: x[1])[0]
        selected.append(pick)
        unselected.remove(pick)

    return selected

def get_sentence_vectors_tf(sentences, tfidf_vectorizer):
    if not sentences:
        return None
    return tfidf_vectorizer.transform(sentences).toarray()
