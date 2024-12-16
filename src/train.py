# src/train.py

import logging
from sklearn.model_selection import train_test_split
from enum import StrEnum
from src.models import Model, NaiveBayes, MultilabelNaiveBayes

logger = logging.getLogger(__name__)


class ModelType(StrEnum):
    NAIVE_BAYES = "naive_bayes"


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
    logger.info(f"Training model with {model_config}.")

    model_type = model_config.get("type")
    test_size = model_config.get("test_size")
    random_state = model_config.get("random_state", None)

    x_train, _, y_train, _ = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    if model_type == ModelType.NAIVE_BAYES:
        if y_train.shape[1] == 1:
            logger.info("Training a binary Naive Bayes model.")
            model = NaiveBayes(model_config)
            y_train = y_train.values.ravel()  # Flatten labels for binary classification
        else:
            logger.info("Training a multilabel Naive Bayes model.")
            model = MultilabelNaiveBayes(model_config)
    else:
        logger.error(
            f"Invalid model type '{model_type}'. Supported types: {[mt for mt in ModelType]}."
        )

    model.fit(x_train, y_train)

    logger.info("Model training complete.")
    return model
