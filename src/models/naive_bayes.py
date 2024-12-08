# src/models/naive_bayes.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from .base import Model


class NaiveBayes(Model):
    """
    A Naive Bayes classifier based on the MultinomialNB model from scikit-learn.
    """

    def __init__(self, model_config):
        alpha = model_config.get("alpha")

        self._model = MultinomialNB(alpha=alpha)

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)


class MultilabelNaiveBayes(Model):
    """
    A multilabel Naive Bayes classifier using OneVsRestClassifier with optional oversampling.
    """

    def __init__(self, model_config):
        alpha = model_config.get("alpha")
        oversampling = model_config.get("oversampling")

        if oversampling:
            self._model = OneVsRestClassifier(
                Pipeline(
                    [
                        ("oversample", RandomOverSampler(random_state=42)),
                        ("classifier", MultinomialNB(alpha=alpha)),
                    ]
                )
            )
        else:
            self._model = OneVsRestClassifier(MultinomialNB(alpha=alpha))

    def fit(self, features, labels):
        self._model.fit(features, labels)

    def predict(self, features):
        return self._model.predict(features)
