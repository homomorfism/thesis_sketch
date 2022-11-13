__all__ = ['recall', 'f_score', 'ndcg_score']

import numpy as np
from sklearn.metrics import ndcg_score as sklearn_ndcg


def recall(
        retrieved_docs_ids: list[list[int]],
        relevant_docs_ids: list[list[int]],
):
    scores = []
    for retrieved, relevant in zip(retrieved_docs_ids, relevant_docs_ids):
        score = len(set(retrieved).intersection(relevant)) / len(relevant)
        scores.append(score)

    return sum(scores) / len(scores)


def f_score(
        retrieved_docs_scores: np.array,
        relevant_docs_scores: np.array,
        threshold: float
):
    scores = []
    for retrieved_scores, relevant_scores in zip(retrieved_docs_scores, relevant_docs_scores):
        retrieved_ids = set([ii for ii, score in enumerate(retrieved_scores) if score >= threshold])
        relevant_ids = set([ii for ii, score in enumerate(relevant_scores) if score >= threshold])

        tp = len(retrieved_ids.intersection(relevant_ids))
        fp = len(retrieved_ids.difference(relevant_ids))
        fn = len(relevant_ids.difference(retrieved_ids))

        score = 2 * tp / (2 * tp + fp + fn)
        scores.append(score)

    return sum(scores) / len(scores)


def ndcg_score(
        retrieved_docs_scores: np.array,
        relevant_docs_scores: np.array,
        k: int
):
    score = sklearn_ndcg(
        y_true=relevant_docs_scores,
        y_score=retrieved_docs_scores,
        k=k,
    )

    return score
