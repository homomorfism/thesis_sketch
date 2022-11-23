import numpy as np
from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split

from src.data_reader import DatasetType, read_data
from src.evaluate.base_evaluator import BaseEvaluator
from src.evaluate.f_score import FScoreEvaluator
from src.evaluate.ndcg import NDCGEvaluator
from src.evaluate.recall import RecallEvaluator
from src.models.bm_25 import BM25
from src.utils import timer
from tqdm import tqdm

seed_everything(seed=42)


def train(
        train_queries: list[dict],
        val_queries: list[dict],
        articles: list[dict],
        model: BM25,
        evaluators: list[BaseEvaluator]
):
    with timer("Training BM25"):
        model.fit(train_queries, articles)

    article_name_to_id = {}
    for ii, article_item in enumerate(articles):
        name = article_item['article_name']
        article_name_to_id[name] = ii


    val_predictions = []
    val_relevancy = []

    for query_item in tqdm(val_queries, desc="Generating predictions"):
        predictions: np.ndarray = model.predict(query_item['query'])
        predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))
        val_predictions.append(predictions)

        relevancy_mask = np.zeros(shape=len(article_name_to_id), dtype=float)
        for article_name in query_item['articles']:
            index = article_name_to_id[article_name]
            relevancy_mask[index] = 1.

        val_relevancy.append(relevancy_mask)

    metrics = {}
    for evaluator in evaluators:
        metrics |= evaluator.evaluate(val_predictions, val_relevancy)

    return metrics


def main():
    collie_queries, collie_articles = read_data(DatasetType.COLLIE)
    collie_train_queries, collie_val_queries = train_test_split(collie_queries, test_size=0.2)

    # vietnam_queries, vietnam_articles = read_data(DatasetType.VIETNAM)
    # vietnam_train_queries, vietnam_val_queries = train_test_split(vietnam_queries, test_size=0.2)

    bm25_model = BM25()

    top_k = [2, 3, 5, 10]
    evaluators = [
        FScoreEvaluator(),
        NDCGEvaluator(top_k=top_k),
        RecallEvaluator(top_k=top_k),
    ]

    collie_metrics = train(
        train_queries=collie_train_queries,
        val_queries=collie_val_queries,
        articles=collie_articles,
        model=bm25_model,
        evaluators=evaluators
    )
    print("Collie metrics")
    print(collie_metrics)

    # vietnam_metrics = train(
    #     train_queries=vietnam_train_queries,
    #     val_queries=vietnam_val_queries,
    #     articles=vietnam_articles,
    #     model=bm25_model,
    #     evaluators=evaluators
    # )
    #
    # print("Vietnam metrics")
    # print(vietnam_metrics)


if __name__ == '__main__':
    main()
