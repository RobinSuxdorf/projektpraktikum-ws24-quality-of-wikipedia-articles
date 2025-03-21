# src/models/base.py

from abc import ABC, abstractmethod
import joblib


class Model(ABC):
    """
    Abstract base class for machine learning models. This class enforces the implementation
    of the `fit` and `predict` methods in subclasses.
    """

    @abstractmethod
    def fit(self, features: any, labels: any) -> None:
        """
        Fit the model to the training data.

        Args:
            features (array-like or similar): Input data features. Typically a 2D array or similar structure.
            labels (array-like or similar): Target labels corresponding to the input data features.
        """
        pass

    @abstractmethod
    def predict(self, features: any) -> any:
        """
        Make predictions using the fitted model.

        Args:
            features (array-like or similar): Input data for prediction. Typically a 2D array or similar structure.

        Returns:
            array-like: Predicted labels corresponding to the input features.
        """
        pass

    def save(self, file_name: str) -> None:
        """
        Save the model to a file.

        Args:
            file_name (str): Path to the file where the model should be saved.
        """
        joblib.dump(self, file_name)

    def load(self, file_name: str) -> None:
        """
        Load a model from a file.

        Args:
            file_name (str): Path to the file where the model is saved.
        """
        pass
