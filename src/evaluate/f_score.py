import numpy as np

from src.evaluate.base_evaluator import BaseEvaluator


def f_score(retrieved, relevant, threshold):
    retrieved_ids = set([ii for ii, score in enumerate(retrieved) if score >= threshold])
    relevant_ids = set(*(relevant == 1).nonzero())

    tp = len(retrieved_ids.intersection(relevant_ids))
    fp = len(retrieved_ids.difference(relevant_ids))
    fn = len(relevant_ids.difference(retrieved_ids))

    score = 2 * tp / (2 * tp + fp + fn)
    return score


class FScoreEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
        self.thresholds = np.linspace(0.5, 1.0, num=20)

    # It tries all the thresholds and selects max
    def evaluate(self, pred_relevancy: list, true_relevancy: list) -> dict:
        scores_at_threshold = []

        for threshold in self.thresholds:
            scores = []
            for retrieved, relevant in zip(pred_relevancy, true_relevancy):
                score = f_score(retrieved, relevant, threshold)
                scores.append(score)

            scores_at_threshold.append(np.mean(scores))

        max_score = max(scores_at_threshold)
        return {
            'f_score': max_score
        }
