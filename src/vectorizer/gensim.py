# src/vectorizer/gensim.py

import logging
from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np
from .base import Vectorizer

logger = logging.getLogger(__name__)


class Word2Vec_Vectorizer(Vectorizer):
    """
    A vectorizer that uses the Word2Vec model to transform text data into vectors.
    """

    def __init__(self, features_config):
        self.workers = features_config.get("workers", 3)
        self.vector_size = features_config.get("vector_size")
        self.window = features_config.get("window")
        self.min_count = features_config.get("min_count")
        self.sg = features_config.get("sg")
        self.hs = features_config.get("hs")
        self.negative = features_config.get("negative")
        self.alpha = features_config.get("alpha")
        self.epochs = features_config.get("epochs")

    def fit_transform(self, text_series):
        tokenized_texts = [text.split() for text in text_series]
        self._vectorizer = Word2Vec(
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
            word_vecs = [
                self._vectorizer.wv[t] for t in tokens if t in self._vectorizer.wv
            ]
            if word_vecs:
                mean_vector = np.array(sum(word_vecs) / len(word_vecs))
                # Normalize using min-max scaling to [0,1] range
                vec_min = np.min(mean_vector)
                vec_max = np.max(mean_vector)
                if vec_max > vec_min:  # Avoid division by zero
                    mean_vector = (mean_vector - vec_min) / (vec_max - vec_min)
                vectors.append(mean_vector)
            else:
                # Use the average vector of the entire vocabulary as the default vector
                default_vector = np.mean(self._vectorizer.wv.vectors, axis=0)
                vectors.append(default_vector)
        return vectors


class GloVe_Vectorizer(Vectorizer):
    """
    A vectorizer that uses the GloVe model to transform text data into vectors.
    """

    def __init__(self, features_config):
        model_name = features_config.get("model_name")
        logger.info(f"Loading GloVe model: {model_name}")
        self._vectorizer = api.load(model_name)
        logger.info(f"Successfully loaded GloVe model: {model_name}")

    def fit_transform(self, text_series):
        vectors = []
        for text in text_series:
            tokens = text.split()
            token_embs = [self._vectorizer[t] for t in tokens if t in self._vectorizer]
            if token_embs:
                mean_vector = np.mean(token_embs, axis=0)
                # Normalize using min-max scaling to [0,1] range
                vec_min = np.min(mean_vector)
                vec_max = np.max(mean_vector)
                if vec_max > vec_min:  # Avoid division by zero
                    mean_vector = (mean_vector - vec_min) / (vec_max - vec_min)
                vectors.append(mean_vector.tolist())
            else:
                # Use the average vector of the entire vocabulary as the default vector
                default_vector = np.mean(self._vectorizer.vectors, axis=0)
                vectors.append(default_vector.tolist())
        return vectors
