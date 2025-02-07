# src/models/support_vector_machine.py

import logging
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC, LinearSVC
from .base import Model

logger = logging.getLogger(__name__)


class LinearSupportVectorMachine(Model):
    """
    A Support Vector Machine classifier based on the LinearSVC model from scikit-learn.
    """

    def __init__(self, model_config):
        C = model_config.get("C")
        self._model = LinearSVC(C=C)

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class LinearSupportVectorMachineGridSearch(Model):
    """
    A Support Vector Machine classifier based on the LinearSVC model from scikit-learn.
    """

    def __init__(self, model_config):
        self._param_grid = model_config.get("param_grid")
        self._model = LinearSVC()

    def fit(self, features, labels):
        grid_search = GridSearchCV(self._model, self._param_grid, n_jobs=-1)
        grid_search.fit(features, labels)
        self._model = grid_search.best_estimator_
        logger.info(f"Trained LinearSVC model with {grid_search.best_params_}")

    def predict(self, features):
        return self._model.predict(features)


class MultilabelLinearSupportVectorMachine(Model):
    """
    A multilabel Support Vector Machine classifier based on the LinearSVC model from scikit-learn.
    """

    def __init__(self, model_config):
        C = model_config.get("C")
        self._model = MultiOutputClassifier(LinearSVC(C=C), n_jobs=-1)

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class SupportVectorMachine(Model):
    """
    A Support Vector Machine classifier based on the SVC model from scikit-learn.
    """

    def __init__(self, model_config):
        C = model_config.get("C")
        kernel = model_config.get("kernel")
        self._model = SVC(C=C, kernel=kernel, gamma="scale")

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class MultilabelSupportVectorMachine(Model):
    """
    A multilabel Support Vector Machine classifier based on the SVC model from scikit-learn.
    """

    def __init__(self, model_config):
        C = model_config.get("C")
        kernel = model_config.get("kernel")
        self._model = MultiOutputClassifier(SVC(C=C, kernel=kernel, gamma="scale"))

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)
