# src/vectorizer/base.py

from abc import ABC, abstractmethod
import pandas as pd
import joblib


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

    def save(self, file_name: str) -> None:
        """
        Save the vectorizer to a file.

        Args:
            file_name (str): Path to the file where the vectorizer should be saved.
        """
        joblib.dump(self, file_name)

    def load(self, file_name: str) -> None:
        """
        Load a vectorizer from a file.

        Args:
            file_name (str): Path to the file where the vectorizer is saved.
        """
        pass
