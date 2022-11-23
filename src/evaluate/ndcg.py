import numpy as np
from sklearn.metrics import ndcg_score

from src.evaluate.base_evaluator import BaseEvaluator


class NDCGEvaluator(BaseEvaluator):
    def __init__(self, top_k: list[int]):
        super(NDCGEvaluator, self).__init__()

        self.top_k = top_k

    def evaluate(
            self,
            pred_relevancy_scores: list[np.ndarray],
            true_relevancy: list[np.ndarray]
    ) -> dict:
        results = {}

        for k in self.top_k:
            score = ndcg_score(true_relevancy, pred_relevancy_scores, k=k)

            results[f'ndcg_at_{k}'] = score

        return results
