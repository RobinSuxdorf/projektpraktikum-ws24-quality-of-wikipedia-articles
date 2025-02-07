# src/vectorizer/sklearn.py

import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .base import Vectorizer

logger = logging.getLogger(__name__)


class Tfidf_Vectorizer(Vectorizer):
    """
    A vectorizer that uses the TF-IDF model to transform text data into vectors.
    """

    def __init__(self, features_config):
        ngram_range = tuple(features_config.get("ngram_range"))
        max_df = features_config.get("max_df")
        min_df = features_config.get("min_df")
        max_features = features_config.get("max_features")
        sublinear_tf = features_config.get("sublinear_tf")
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )

    def fit_transform(self, text_series):
        return self._vectorizer.fit_transform(text_series)


class Count_Vectorizer(Vectorizer):
    """
    A vectorizer that uses the CountVectorizer model to transform text data into vectors.
    """

    def __init__(self, features_config):
        ngram_range = tuple(features_config.get("ngram_range"))
        max_df = features_config.get("max_df")
        min_df = features_config.get("min_df")
        max_features = features_config.get("max_features")
        binary = features_config.get("binary")
        self._vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            binary=binary,
        )

    def fit_transform(self, text_series):
        return self._vectorizer.fit_transform(text_series)


class BagOfWords_Vectorizer(Vectorizer):
    """
    A vectorizer that uses the CountVectorizer model to transform text data into vectors using the Bag of Words model.
    """

    def __init__(self, features_config):
        ngram_range = tuple(features_config.get("ngram_range"))
        max_df = features_config.get("max_df")
        min_df = features_config.get("min_df")
        max_features = features_config.get("max_features")
        self._vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            binary=True,
        )

    def fit_transform(self, text_series):
        return self._vectorizer.fit_transform(text_series)
