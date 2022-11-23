import json
from enum import Enum
from pathlib import Path


class DatasetType(Enum):
    COLLIE = 0
    VIETNAM = 1


def read_data(dataset_type: DatasetType):
    if dataset_type == DatasetType.COLLIE:
        query_article_path = Path("data/cleaned/COLLIE/query_article.json")
        with open(query_article_path) as f:
            query_article = json.load(f)

        article_path = Path("data/cleaned/COLLIE/articles.json")
        with open(article_path) as f:
            articles = json.load(f)

        return query_article, articles

    elif dataset_type == DatasetType.VIETNAM:
        query_article_path = Path("data/cleaned/Vietnam/query_article.json")
        with open(query_article_path) as f:
            query_article = json.load(f)

        article_path = Path("data/cleaned/Vietnam/articles.json")
        with open(article_path) as f:
            articles = json.load(f)

        return query_article, articles

    else:
        raise RuntimeError(f"Unknown dataset type: {dataset_type}")
