# src/models/naive_bayes.py

import logging
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from .base import Model

logger = logging.getLogger(__name__)


class NaiveBayes(Model):
    """
    A Naive Bayes classifier based on the MultinomialNB model from scikit-learn.
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
    A Naive Bayes classifier using GridSearchCV for hyperparameter tuning.
    """

    def __init__(self, model_config):
        self._param_grid = model_config.get("param_grid")

        self._model = MultinomialNB()

    def fit(self, features, labels):
        grid_search = GridSearchCV(self._model, self._param_grid)
        grid_search.fit(features, labels)
        self._model = grid_search.best_estimator_
        logger.info(
            f"Trained Naive Bayes model parameters: alpha={self._model.alpha}, fit_prior={self._model.fit_prior}"
        )

    def predict(self, features):
        return self._model.predict(features)


class MultilabelNaiveBayes(Model):
    """
    A multilabel Naive Bayes classifier using OneVsRestClassifier with optional oversampling.
    """

    def __init__(self, model_config):
        alpha = model_config.get("alpha")
        fit_prior = model_config.get("fit_prior")
        random_state = model_config.get("random_state", None)
        oversampling = model_config.get("oversampling")

        if oversampling:
            self._model = OneVsRestClassifier(
                Pipeline(
                    [
                        ("oversample", RandomOverSampler(random_state=random_state)),
                        ("classifier", MultinomialNB(alpha=alpha, fit_prior=fit_prior)),
                    ]
                ),
                n_jobs=-1,
            )
        else:
            self._model = OneVsRestClassifier(
                MultinomialNB(alpha=alpha, fit_prior=fit_prior), n_jobs=-1
            )

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class MultilabelNaiveBayesGridSearch(Model):
    """
    A multilabel Naive Bayes classifier using GridSearchCV for hyperparameter tuning.
    """

    def __init__(self, model_config):
        self._param_grid = model_config.get("param_grid")
        random_state = model_config.get("random_state", None)
        oversampling = model_config.get("oversampling")

        if oversampling:
            self._model = OneVsRestClassifier(
                Pipeline(
                    [
                        ("oversample", RandomOverSampler(random_state=random_state)),
                        ("classifier", MultinomialNB()),
                    ]
                ),
                n_jobs=-1,
            )
        else:
            self._model = OneVsRestClassifier(MultinomialNB(), n_jobs=-1)

    def fit(self, features, labels):
        if isinstance(self._model.estimator, Pipeline):
            param_grid = {
                f"estimator__classifier__{key}": value
                for key, value in self._param_grid.items()
            }
        else:
            param_grid = {
                f"estimator__{key}": value for key, value in self._param_grid.items()
            }
        grid_search = GridSearchCV(self._model, param_grid)
        grid_search.fit(features, labels)
        self._model = grid_search.best_estimator_
        if isinstance(self._model.estimator, Pipeline):
            alpha = self._model.estimator.named_steps["classifier"].alpha
            fit_prior = self._model.estimator.named_steps["classifier"].fit_prior
        else:
            alpha = self._model.estimator.alpha
            fit_prior = self._model.estimator.fit_prior
        logger.info(
            f"Trained Multilabel Naive Bayes model parameters: alpha={alpha}, fit_prior={fit_prior}"
        )

    def predict(self, features):
        return self._model.predict(features)
