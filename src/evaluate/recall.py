import numpy as np

from src.evaluate.base_evaluator import BaseEvaluator


def calculate_recall(retrieved_scores, true_scores, k):
    top_k_pred_indices = np.argsort(retrieved_scores)[::-1][:k]
    retrieved = set(top_k_pred_indices)
    relevant = set(*(true_scores == 1).nonzero())

    score = len(retrieved.intersection(relevant)) / len(relevant)
    return score


class RecallEvaluator(BaseEvaluator):
    def __init__(self, top_k: list[int]):
        super(RecallEvaluator, self).__init__()

        self.top_k = top_k

    def evaluate(
            self,
            pred_relevancy_scores: list[np.ndarray],
            true_relevancy: list[np.ndarray]
    ) -> dict:
        results = {}

        for k in self.top_k:
            scores = []
            for retrieved, relevant in zip(pred_relevancy_scores, true_relevancy):
                scores.append(calculate_recall(retrieved, relevant, k))

            results[f'recall_at_{k}'] = np.mean(scores)

        return results
