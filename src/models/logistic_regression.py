# src/models/logistic_regression.py

import logging
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from .base import Model

logger = logging.getLogger(__name__)


class Logistic_Regression(Model):
    """
    Logistic Regression model for binary classification.
    """

    def __init__(self, model_config):
        penalty = model_config.get("penalty")
        solver = model_config.get("solver")
        max_iter = model_config.get("max_iter")

        self._model = LogisticRegression(
            penalty=penalty, solver=solver, max_iter=max_iter
        )

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class MultilabelLogisticRegression(Model):
    """
    Multilabel Logistic Regression model using OneVsRestClassifier with optional oversampling.
    """

    def __init__(self, model_config):
        penalty = model_config.get("penalty")
        solver = model_config.get("solver")
        max_iter = model_config.get("max_iter")
        random_state = model_config.get("random_state", None)
        oversampling = model_config.get("oversampling")

        if oversampling:
            self._model = OneVsRestClassifier(
                Pipeline(
                    [
                        ("oversample", RandomOverSampler(random_state=random_state)),
                        (
                            "classifier",
                            LogisticRegression(
                                penalty=penalty, solver=solver, max_iter=max_iter
                            ),
                        ),
                    ]
                ),
                n_jobs=-1,
            )
        else:
            self._model = OneVsRestClassifier(
                LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter),
                n_jobs=-1,
            )

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)
