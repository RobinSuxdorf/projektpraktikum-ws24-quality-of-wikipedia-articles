"""Author: Johannes Krämer"""

import logging
import pandas as pd
import numpy as np
import sklearn.metrics as skm
from sklearn.base import BaseEstimator
from typing import Any


def evaluate_model(X_test: Any, y_test: Any, model: BaseEstimator) -> None:
    """
    Evaluates the given model using the test data and logs the results.

    Parameters:
    X_test (Any): Test feature matrix.
    y_test (Any): Test labels.
    model (BaseEstimator): The machine learning model to be evaluated.

    Returns:
    None
    """
    logging.info("%s - Evaluating model", model)
    y_pred = model.predict(X_test)
    confusion_matrix = skm.confusion_matrix(y_test, y_pred)
    logging.info("%s - Confusion Matrix:\n%s", model, confusion_matrix)
    classification_report = skm.classification_report(y_test, y_pred, zero_division=0)
    logging.info("%s - Classification Report:\n%s", model, classification_report)
    accuracy = skm.accuracy_score(y_test, y_pred)
    logging.info("%s - Accuracy: %s", model, accuracy)


def evaluate_model_with_categories(
    X_test: Any, y_test: pd.DataFrame, model: BaseEstimator
) -> None:
    """
    Evaluates the given model using the test data and logs the results.

    Parameters:
    X_test (Any): Test feature matrix.
    y_test (pd.DataFrame): Test labels.
    model (BaseEstimator): The machine learning model to be evaluated.

    Returns:
    None
    """
    logging.info("%s - Evaluating model", model)
    y_pred = model.predict(X_test)
    if not isinstance(y_pred, pd.DataFrame):
        y_pred = pd.DataFrame(y_pred, columns=y_test.columns)
    accuracies = []
    for column in y_test.columns:
        logging.info("%s - Evaluating output: %s", model, column)
        confusion_matrix = skm.confusion_matrix(y_test[column], y_pred[column])
        logging.info(
            "%s - Confusion Matrix for %s:\n%s", model, column, confusion_matrix
        )
        classification_report = skm.classification_report(
            y_test[column], y_pred[column], zero_division=0
        )
        logging.info(
            "%s - Classification Report for %s:\n%s",
            model,
            column,
            classification_report,
        )
        accuracy = skm.accuracy_score(y_test[column], y_pred[column])
        logging.info("%s - Accuracy for %s: %s", model, column, accuracy)
        accuracies.append(accuracy)
    overall_accuracy = np.mean(accuracies)
    logging.info("%s - Overall Accuracy: %s", model, overall_accuracy)
