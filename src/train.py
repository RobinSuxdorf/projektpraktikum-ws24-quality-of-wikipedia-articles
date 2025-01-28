# src/train.py

import logging
from enum import StrEnum
from src.models import (
    Model,
    NaiveBayes,
    NaiveBayesGridSearch,
    MultilabelNaiveBayes,
    MultilabelNaiveBayesGridSearch,
    LinearSupportVectorMachine,
    MultilabelLinearSupportVectorMachine,
    Logistic_Regression,
    MultilabelLogisticRegression,
)

logger = logging.getLogger(__name__)


class ModelType(StrEnum):
    NAIVE_BAYES = "naive_bayes"
    LINEAR_SVM = "linear_svm"
    LOGISTIC_REGRESSION = "logistic_regression"


def train_model(x_train, y_train, model_config: dict) -> Model:
    """
    Train a machine learning model based on the provided configuration.

    Args:
        x_train (array-like): Input features for training the model.
        y_train (array-like): Target labels corresponding to the input features.
        model_config (dict): Configuration dictionary for the model.

    Returns:
        Model: An instance of the trained model.
    """
    logger.info(f"Training model with {model_config}.")

    model_type = model_config.get("type")
    grid_search = model_config.get("grid_search")

    binary_classification = True if y_train.shape[1] == 1 else False
    if binary_classification:
        logger.info("Flatten labels for binary classification.")
        y_train = y_train.values.ravel()

    if model_type == ModelType.NAIVE_BAYES:
        if binary_classification and grid_search:
            logger.info("Training a binary Naive Bayes model with grid search.")
            model = NaiveBayesGridSearch(model_config)
        elif binary_classification and not grid_search:
            logger.info("Training a binary Naive Bayes model.")
            model = NaiveBayes(model_config)
        elif not binary_classification and grid_search:
            logger.info("Training a multilabel Naive Bayes model with grid search.")
            model = MultilabelNaiveBayesGridSearch(model_config)
        elif not binary_classification and not grid_search:
            logger.info("Training a multilabel Naive Bayes model.")
            model = MultilabelNaiveBayes(model_config)
    elif model_type == ModelType.LINEAR_SVM:
        if binary_classification:
            logger.info("Training a binary SVM model.")
            model = LinearSupportVectorMachine(model_config)
        else:
            logger.info("Training a multilabel SVM model.")
            model = MultilabelLinearSupportVectorMachine(model_config)
    elif model_type == ModelType.LOGISTIC_REGRESSION:
        if binary_classification:
            logger.info("Training a Logistic Regression model.")
            model = Logistic_Regression(model_config)
        else:
            logger.info("Training a multilabel Logistic Regression model.")
            model = MultilabelLogisticRegression(model_config)
    else:
        logger.error(
            f"Invalid model type '{model_type}'. Supported types: {[mt for mt in ModelType]}."
        )

    model.fit(x_train, y_train)

    logger.info("Model training complete.")
    return model
