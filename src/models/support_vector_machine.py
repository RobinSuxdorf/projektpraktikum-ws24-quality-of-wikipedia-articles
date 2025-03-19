# src/models/support_vector_machine.py

import logging
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from .base import Model

logger = logging.getLogger(__name__)


class LinearSupportVectorMachine(Model):
    """
    Support Vector Machine classifier based on the LinearSVC model from scikit-learn
    """

    def __init__(self, model_config):
        C = model_config.get("C")
        loss = model_config.get("loss")
        self._model = LinearSVC(C=C, loss=loss)

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class LinearSupportVectorMachineGridSearch(Model):
    """
    Support Vector Machine classifier based on the LinearSVC model from scikit-learn with GridSearchCV for hyperparameter tuning
    """

    def __init__(self, model_config):
        self._param_grid = model_config.get("param_grid")
        self._model = LinearSVC()

    def fit(self, features, labels):
        grid_search = GridSearchCV(
            self._model, self._param_grid, scoring="recall_macro"
        )
        grid_search.fit(features, labels)
        self._model = grid_search.best_estimator_
        logger.info(f"Trained LinearSVC model with {grid_search.best_params_}")

    def predict(self, features):
        return self._model.predict(features)


class MultilabelLinearSupportVectorMachine(Model):
    """
    Support Vector Machine classifier for multilabel based on the LinearSVC model from scikit-learn
    """

    def __init__(self, model_config):
        C = model_config.get("C")
        loss = model_config.get("loss")
        self._model = OneVsRestClassifier(LinearSVC(C=C, loss=loss), n_jobs=-1)

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class MultilabelLinearSupportVectorMachineGridSearch(Model):
    """
    Support Vector Machine classifier for multilabel based on the LinearSVC model from scikit-learn with GridSearchCV for hyperparameter tuning
    """

    def __init__(self, model_config):
        self._param_grid = model_config.get("param_grid")
        self._model = OneVsRestClassifier(LinearSVC(), n_jobs=-1)

    def fit(self, features, labels):
        param_grid = {
            f"estimator__{key}": value for key, value in self._param_grid.items()
        }
        grid_search = GridSearchCV(self._model, param_grid, scoring="recall_macro")
        grid_search.fit(features, labels)
        self._model = grid_search.best_estimator_
        logger.info(f"Trained LinearSVC model with {grid_search.best_params_}")

    def predict(self, features):
        return self._model.predict(features)


class SupportVectorMachine(Model):
    """
    Support Vector Machine classifier based on the SVC model from scikit-learn
    """

    def __init__(self, model_config):
        C = model_config.get("C")
        kernel = model_config.get("kernel")
        gamma = model_config.get("gamma")
        self._model = SVC(C=C, kernel=kernel, gamma=gamma)

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class MultilabelSupportVectorMachine(Model):
    """
    Support Vector Machine classifier for multilabel based on the SVC model from scikit-learn
    """

    def __init__(self, model_config):
        C = model_config.get("C")
        kernel = model_config.get("kernel")
        gamma = model_config.get("gamma")
        self._model = OneVsRestClassifier(
            SVC(C=C, kernel=kernel, gamma=gamma), n_jobs=-1
        )

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)
