import logging
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
    logging.info(
        "%s - Confusion Matrix:\n%s", model, skm.confusion_matrix(y_test, y_pred)
    )
    logging.info(
        "%s - Classification Report:\n%s",
        model,
        skm.classification_report(y_test, y_pred, zero_division=0),
    )
    logging.info("%s - Accuracy: %s", model, skm.accuracy_score(y_test, y_pred))
