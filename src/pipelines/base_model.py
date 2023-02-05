from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    name: str

    def __init__(self, config: object = None):
        pass

    @abstractmethod
    def fit(self, query_item_data: list[dict], articles_data: list[dict]) -> None:
        pass

    @abstractmethod
    def predict(self, query: str) -> np.ndarray:
        """Return not normalized score"""
        pass

    @abstractmethod
    def save(self, path: Path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path, config: object = None):
        pass
