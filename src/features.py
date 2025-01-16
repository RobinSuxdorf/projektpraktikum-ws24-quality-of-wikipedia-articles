# src/features.py

import logging
from enum import StrEnum
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np

logger = logging.getLogger(__name__)


class FeatureType(StrEnum):
    TFIDF = "tfidf"
    COUNT = "count"
    WORD2VEC = "word2vec"
    GLOVE = "glove"


class Word2VecVectorizer:
    def __init__(
        self,
        workers,
        vector_size,
        window,
        min_count,
        sg,
        hs,
        negative,
        alpha,
        epochs,
    ):
        self.workers = workers
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.alpha = alpha
        self.epochs = epochs
        self.model = None

    def fit_transform(self, text_series: pd.Series):
        tokenized_texts = [text.split() for text in text_series]
        self.model = Word2Vec(
            tokenized_texts,
            workers=self.workers,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            hs=self.hs,
            negative=self.negative,
            alpha=self.alpha,
            epochs=self.epochs,
        )
        vectors = []
        for tokens in tokenized_texts:
            word_vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
            if word_vecs:
                mean_vector = sum(word_vecs) / len(word_vecs)
                mean_vector[mean_vector < 0] = 0  # Ensure non-negative values
                vectors.append(mean_vector)
            else:
                vectors.append([0] * self.vector_size)
        return vectors


class GloVe:
    def __init__(self, model_name):
        self.embeddings = api.load(model_name)
        self.embedding_dim = self.embeddings.vector_size

    def fit_transform(self, text_series: pd.Series):
        vectors = []
        for text in text_series:
            tokens = text.split()
            token_embs = [self.embeddings[t] for t in tokens if t in self.embeddings]
            if token_embs:
                mean_vector = np.mean(token_embs, axis=0)
                mean_vector[mean_vector < 0] = 0  # Ensure non-negative values
                vectors.append(mean_vector.tolist())
            else:
                vectors.append([0.0] * self.embedding_dim)
        return vectors


def get_features(text_series: pd.Series, features_config: dict):
    """
    Get features based on the provided configuration.

    Args:
        text_series (pd.Series): Series containing text data.
        features_config (dict): Configuration dictionary for the feature extraction.

    Returns:
        features (array-like): Input features for training the model.
    """
    logger.info(f"Getting features with {features_config}.")

    feature_type = features_config.get("type")

    if feature_type == FeatureType.TFIDF:
        logger.info("Using a TF-IDF vectorizer.")
        vectorizer = TfidfVectorizer(
            ngram_range=tuple(features_config.get("ngram_range")),
            max_df=features_config.get("max_df"),
            min_df=features_config.get("min_df"),
            max_features=features_config.get("max_features"),
            sublinear_tf=features_config.get("sublinear_tf"),
        )
    elif feature_type == FeatureType.COUNT:
        logger.info("Using a Count vectorizer.")
        vectorizer = CountVectorizer(
            ngram_range=tuple(features_config.get("ngram_range")),
            max_df=features_config.get("max_df"),
            min_df=features_config.get("min_df"),
            max_features=features_config.get("max_features"),
            binary=features_config.get("binary"),
        )
    elif feature_type == FeatureType.WORD2VEC:
        logger.info("Using a Word2Vec vectorizer.")
        vectorizer = Word2VecVectorizer(
            workers=features_config.get("workers"),
            vector_size=features_config.get("vector_size"),
            window=features_config.get("window"),
            min_count=features_config.get("min_count"),
            sg=features_config.get("sg"),
            hs=features_config.get("hs"),
            negative=features_config.get("negative"),
            alpha=features_config.get("alpha"),
            epochs=features_config.get("epochs"),
        )
    elif feature_type == FeatureType.GLOVE:
        logger.info("Using a GloVe vectorizer.")
        vectorizer = GloVe(model_name=features_config.get("model_name"))
    else:
        logger.error(
            f"Invalid feature extraction type '{feature_type}'. Supported types: {[ft for ft in FeatureType]}."
        )

    features = vectorizer.fit_transform(text_series)

    logger.info("Feature extraction complete.")
    return features
