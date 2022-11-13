import json
from enum import Enum
from pathlib import Path


class Dataset(Enum):
    COLLIE = 0
    VIETNAM = 1


def read_data(dataset_type: Dataset):
    if dataset_type == Dataset.COLLIE:
        query_article_path = Path("../../data/cleaned/COLLIE/query_article.json")
        with open(query_article_path) as f:
            query_article = json.load(f)

        article_path = Path("../../data/cleaned/COLLIE/articles.json")
        with open(article_path) as f:
            articles = json.load(f)

        return query_article, articles
    #
    # elif dataset_type == Dataset.VIETNAM:
    #     query_article_path = Path("../../data/preprocessed/VietnamTranslated/query_article.json")
    #     with open(query_article_path) as f:
    #         query_article = json.load(f)
    #
    #     article_path = Path("../../data/preprocessed/VietnamTranslated/articles.json")
    #     with open(article_path) as f:
    #         articles = json.load(f)
    #
    #     return query_article, articles

    else:
        raise RuntimeError(f"Unknown dataset type: {dataset_type}")
