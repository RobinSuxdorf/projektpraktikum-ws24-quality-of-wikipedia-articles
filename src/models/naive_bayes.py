from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from .base import Model


class NaiveBayes(Model):
    def __init__(self, model_config):
        alpha = model_config.get("alpha")
        self._model = MultinomialNB(alpha=alpha)

    def fit(self, x, y):
        self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict(x)


class MultilabelNaiveBayes(Model):
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

    def fit(self, x, y):
        self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict(x)
