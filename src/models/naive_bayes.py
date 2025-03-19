# src/models/naive_bayes.py

import logging
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from .base import Model

logger = logging.getLogger(__name__)


class NaiveBayes(Model):
    """
    Naive Bayes classifier based on the MultinomialNB model from scikit-learn
    """

    def __init__(self, model_config):
        alpha = model_config.get("alpha")
        fit_prior = model_config.get("fit_prior")
        self._model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class NaiveBayesGridSearch(Model):
    """
    Naive Bayes classifier based on the MultinomialNB model from scikit-learn with GridSearchCV for hyperparameter tuning
    """

    def __init__(self, model_config):
        self._param_grid = model_config.get("param_grid")
        self._model = MultinomialNB()

    def fit(self, features, labels):
        grid_search = GridSearchCV(
            self._model, self._param_grid, scoring="recall_macro"
        )
        grid_search.fit(features, labels)
        self._model = grid_search.best_estimator_
        logger.info(f"Trained Naive Bayes model with {grid_search.best_params_}")

    def predict(self, features):
        return self._model.predict(features)


class MultilabelNaiveBayes(Model):
    """
    Naive Bayes classifier for multilabel based on the MultinomialNB model from scikit-learn using OneVsRestClassifier
    """

    def __init__(self, model_config):
        alpha = model_config.get("alpha")
        fit_prior = model_config.get("fit_prior")
        self._model = OneVsRestClassifier(
            MultinomialNB(alpha=alpha, fit_prior=fit_prior), n_jobs=-1
        )

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class MultilabelNaiveBayesGridSearch(Model):
    """
    Naive Bayes classifier for multilabel based on the MultinomialNB model from scikit-learn using OneVsRestClassifier with GridSearchCV for hyperparameter tuning
    """

    def __init__(self, model_config):
        self._param_grid = model_config.get("param_grid")
        self._model = OneVsRestClassifier(MultinomialNB(), n_jobs=-1)

    def fit(self, features, labels):
        param_grid = {
            f"estimator__{key}": value for key, value in self._param_grid.items()
        }
        grid_search = GridSearchCV(self._model, param_grid, scoring="recall_macro")
        grid_search.fit(features, labels)
        self._model = grid_search.best_estimator_
        logger.info(f"Trained Naive Bayes model with {grid_search.best_params_}")

    def predict(self, features):
        return self._model.predict(features)
