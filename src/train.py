# src/train.py

import logging
from sklearn.model_selection import train_test_split
from src.models import (
    Model,
    NaiveBayes,
    MultilabelNaiveBayes,
    LinearSupportVectorMachine,
    MultilabelLinearSupportVectorMachine,
)

logger = logging.getLogger(__name__)


def train_model(features, labels, model_config: dict) -> Model:
    """
    Train a machine learning model based on the provided configuration.

    Args:
        features (array-like): Input features for training the model.
        labels (array-like): Target labels corresponding to the input features.
        model_config (dict): Configuration dictionary for the model.

    Returns:
        Model: An instance of the trained model.
    """
    type = model_config.get("type")

    logger.info("Splitting data into training and testing sets.")
    x_train, _, y_train, _ = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    if type == "naive_bayes":
        if y_train.shape[1] == 1:
            logger.info("Training a binary Naive Bayes model.")
            model = NaiveBayes(model_config)
            y_train = y_train.values.ravel()  # Flatten labels for binary classification
            model.fit(x_train, y_train)
        else:
            logger.info("Training a multilabel Naive Bayes model.")
            model = MultilabelNaiveBayes(model_config)
            model.fit(x_train, y_train)
    elif type == "linear_svm":
        if y_train.shape[1] == 1:
            logger.info("Training a binary SVM model.")
            model = LinearSupportVectorMachine(model_config)
            y_train = y_train.values.ravel()  # Flatten labels for binary classification
            model.fit(x_train, y_train)
        else:
            logger.info("Training a multilabel SVM model.")
            model = MultilabelLinearSupportVectorMachine(model_config)
            model.fit(x_train, y_train)

    logger.info("Model training complete.")
    return model
