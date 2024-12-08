# src/models/base.py

from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for machine learning models. This class enforces the implementation
    of the `fit` and `predict` methods in subclasses.
    """

    @abstractmethod
    def fit(self, features, labels):
        """
        Fit the model to the training data.

        Args:
            features (array-like or similar): Input data features. Typically a 2D array or similar structure.
            labels (array-like or similar): Target labels corresponding to the input data features.
        """
        pass

    @abstractmethod
    def predict(self, features):
        """
        Make predictions using the fitted model.

        Args:
            features (array-like or similar): Input data for prediction. Typically a 2D array or similar structure.

        Returns:
            array-like: Predicted labels corresponding to the input features.
        """
        pass
