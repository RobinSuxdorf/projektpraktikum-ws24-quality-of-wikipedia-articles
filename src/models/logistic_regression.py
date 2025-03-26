"""Model class definitions for Logistic Regression models.

Authors: Alexander Kunze, Sebastian Bunge
"""

import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from .base import Model

logger = logging.getLogger(__name__)


class Logistic_Regression(Model):
    """
    Logistic Regression classifier based on LogisticRegression from scikit-learn
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


class LogisticRegressionGridSearch(Model):
    """
    Logistic Regression classifier based on the LogisticRegression model from scikit-learn with GridSearchCV for hyperparameter tuning
    """

    def __init__(self, model_config):
        self._param_grid = model_config.get("param_grid")
        self._model = LogisticRegression()

    def fit(self, features, labels):
        grid_search = GridSearchCV(
            self._model, self._param_grid, scoring="recall_macro"
        )
        grid_search.fit(features, labels)
        self._model = grid_search.best_estimator_
        logger.info(f"Trained LogisticRegression model with {grid_search.best_params_}")

    def predict(self, features):
        return self._model.predict(features)


class MultilabelLogisticRegression(Model):
    """
    Logistic Regression classifier for multilabel based on LogisticRegression from scikit-learn using OneVsRestClassifier
    """

    def __init__(self, model_config):
        penalty = model_config.get("penalty")
        solver = model_config.get("solver")
        max_iter = model_config.get("max_iter")
        self._model = OneVsRestClassifier(
            LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter),
            n_jobs=-1,
        )

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class MultilabelLogisticRegressionGridSearch(Model):
    """
    Logistic Regression classifier for multilabel based on the LogisticRegression model from scikit-learn using OneVsRestClassifier with GridSearchCV for hyperparameter tuning
    """

    def __init__(self, model_config):
        self._param_grid = model_config.get("param_grid")
        self._model = OneVsRestClassifier(LogisticRegression(), n_jobs=-1)

    def fit(self, features, labels):
        param_grid = {
            f"estimator__{key}": value for key, value in self._param_grid.items()
        }
        grid_search = GridSearchCV(self._model, param_grid, scoring="recall_macro")
        grid_search.fit(features, labels)
        self._model = grid_search.best_estimator_
        logger.info(f"Trained LogisticRegression model with {grid_search.best_params_}")

    def predict(self, features):
        return self._model.predict(features)
