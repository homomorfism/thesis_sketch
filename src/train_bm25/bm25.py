__all__ = ['BM25']

import numpy as np
from rank_bm25 import BM25Okapi


class BM25:
    def __init__(self, article_sentences: list[str]):
        tokenized_corpus = [article.split(" ") for article in article_sentences]
        self.model = BM25Okapi(tokenized_corpus)

    def predict(self, queries: list[str], top_k: int):
        predictions = []
        for text in queries:
            query = text.split(" ")
            scores = self.model.get_scores(query)
            top_n_indices = np.argsort(scores)[::-1][:top_k].tolist()
            predictions.append(top_n_indices)
        return predictions

    def get_scores(self, queries: list[str]) -> np.array:
        predictions = []
        for text in queries:
            query = text.split(" ")
            scores = self.model.get_scores(query).tolist()
            predictions.append(scores)
        return np.asarray(predictions, dtype=float)
