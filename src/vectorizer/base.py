# src/vectorizer/base.py

from abc import ABC, abstractmethod
import pandas as pd


class Vectorizer(ABC):
    """
    Abstract base class for vectorizers. This class enforces the implementation
    of the `fit_transform` method in subclasses.
    """

    @abstractmethod
    def fit_transform(self, text_series: pd.Series) -> None:
        """
        Transform the text data into vectors.

        Args:
            text_series (pd.Series): Series containing text data.
        """
        pass
