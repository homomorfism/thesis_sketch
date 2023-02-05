import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.pipelines.base_model import BaseModel


class BM25(BaseModel):
    def __init__(self):
        super().__init__()
        self.model: BM25Okapi | None = None

    def fit(self, query_item_data: list[dict], articles_data: list[dict]) -> None:
        article_content = [item["article_content"].split(" ") for item in articles_data]

        self.model = BM25Okapi(article_content)

    def predict(self, query: str) -> np.ndarray:
        assert self.model is not None, "Model should be trained!"

        return self.model.get_scores(query.split(" "))

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path, config: object = None):
        with open(path, "rb") as f:
            model = pickle.load(f)

        obj = cls()
        obj.model = model

        return obj
