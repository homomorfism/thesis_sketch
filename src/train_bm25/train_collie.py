import numpy as np

from src.data_reader import Dataset, read_data
from src.metrics import f_score, ndcg_score, recall
from src.train_bm25.bm25 import BM25


def main():
    query_article, articles = read_data(Dataset.COLLIE)

    article_name_to_id = {
        item['article_name']: ii for ii, item in enumerate(articles)
    }

    relevant_article_ids = []
    relevant_scores = []
    for query_item in query_article:
        relevant_ids = [
            article_name_to_id[name] for name in query_item['articles']
        ]

        relevance_mask = np.zeros(shape=len(articles), dtype=float)
        relevance_mask[relevant_ids] = 1.
        relevant_scores.append(relevance_mask)

        relevant_article_ids.append(relevant_ids)

    relevant_scores = np.asarray(relevant_scores, dtype=float)

    article_content = [item['article_content'] for item in articles]
    query_content = [item['query'] for item in query_article]
    model = BM25(article_content)

    scores = model.get_scores(query_content)
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    print("calculating recall...")
    top_k_values = [2, 3, 5, 10]
    for k in top_k_values:
        y_pred = model.predict(query_content, top_k=k)
        score = recall(retrieved_docs_ids=y_pred, relevant_docs_ids=relevant_article_ids)

        print(f"recall(k={k}) = {round(score, 4)}")

    print("calculating ndcg...")
    for k in top_k_values:
        score = ndcg_score(
            retrieved_docs_scores=scores,
            relevant_docs_scores=relevant_scores,
            k=k
        )

        print(f"ndcg(k={k}) = {round(score, 4)}")

    print("Calculating f score...")
    thresholds = np.linspace(0.1, 1.0, num=10)
    for thresh in thresholds:
        score = f_score(
            retrieved_docs_scores=scores,
            relevant_docs_scores=relevant_scores,
            threshold=thresh
        )

        print(f"F score (thresh={round(thresh, 2)}) = {round(score, 4)}")


if __name__ == '__main__':
    main()
