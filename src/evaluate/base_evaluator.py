from abc import ABC, abstractmethod

import pandas as pd


class BaseEvaluator(ABC):
    name: str

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, pred_relevancy: list, true_relevancy: list) -> pd.DataFrame:
        pass
