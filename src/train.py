"""Module for training models using various algorithms with optional grid search optimization.

Author: Sebastian Bunge
"""

import logging
from enum import StrEnum
from src.models import (
    Model,
    NaiveBayes,
    NaiveBayesGridSearch,
    MultilabelNaiveBayes,
    MultilabelNaiveBayesGridSearch,
    LinearSupportVectorMachine,
    LinearSupportVectorMachineGridSearch,
    MultilabelLinearSupportVectorMachine,
    MultilabelLinearSupportVectorMachineGridSearch,
    SupportVectorMachine,
    MultilabelSupportVectorMachine,
    Logistic_Regression,
    LogisticRegressionGridSearch,
    MultilabelLogisticRegression,
    MultilabelLogisticRegressionGridSearch,
)

logger = logging.getLogger(__name__)


# Define the supported model types
class ModelType(StrEnum):
    NAIVE_BAYES = "naive_bayes"
    LINEAR_SVM = "linear_svm"
    SVM = "svm"
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

    # Load common configuration parameters
    model_type = model_config.get("type")
    grid_search = model_config.get("grid_search")

    # If the target labels are binary, flatten the array
    binary_classification = True if y_train.shape[1] == 1 else False
    if binary_classification:
        logger.info("Flatten labels for binary classification.")
        y_train = y_train.values.ravel()

    # Define a factory for choosing the model class based on the configuration
    # (model_type, binary_classification, grid_search) -> ModelClass
    model_factory = {
        (ModelType.NAIVE_BAYES, True, True): NaiveBayesGridSearch,
        (ModelType.NAIVE_BAYES, True, False): NaiveBayes,
        (ModelType.NAIVE_BAYES, False, True): MultilabelNaiveBayesGridSearch,
        (ModelType.NAIVE_BAYES, False, False): MultilabelNaiveBayes,
        (ModelType.LINEAR_SVM, True, True): LinearSupportVectorMachineGridSearch,
        (ModelType.LINEAR_SVM, True, False): LinearSupportVectorMachine,
        (ModelType.LINEAR_SVM, False, True): MultilabelLinearSupportVectorMachineGridSearch,
        (ModelType.LINEAR_SVM, False, False): MultilabelLinearSupportVectorMachine,
        (ModelType.SVM, True, True): SupportVectorMachine,
        (ModelType.SVM, True, False): SupportVectorMachine,
        (ModelType.SVM, False, True): MultilabelSupportVectorMachine,
        (ModelType.SVM, False, False): MultilabelSupportVectorMachine,
        (ModelType.LOGISTIC_REGRESSION, True, True): LogisticRegressionGridSearch,
        (ModelType.LOGISTIC_REGRESSION, True, False): Logistic_Regression,
        (ModelType.LOGISTIC_REGRESSION, False, True): MultilabelLogisticRegressionGridSearch,
        (ModelType.LOGISTIC_REGRESSION, False, False): MultilabelLogisticRegression,
    }
    model_class = model_factory.get((model_type, binary_classification, grid_search))

    # Initialize the model
    if model_class:
        model_name = model_class.__name__
        logger.info(f"Training a {model_name} model.")
        model = model_class(model_config)
    else:
        logger.error(
            f"Invalid model type '{model_type}'. Supported types: {[mt for mt in ModelType]}."
        )

    # Train the model
    model.fit(x_train, y_train)

    logger.info("Model training complete.")
    return model
