# src/features.py

import logging
from enum import StrEnum
import pandas as pd
from src.vectorizer import (
    Tfidf_Vectorizer,
    Count_Vectorizer,
    BagOfWords_Vectorizer,
    Word2Vec_Vectorizer,
    GloVe_Vectorizer,
)

logger = logging.getLogger(__name__)


class FeatureType(StrEnum):
    TFIDF = "tfidf"
    COUNT = "count"
    BAGOFWORDS = "bagofwords"
    WORD2VEC = "word2vec"
    GLOVE = "glove"


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
        vectorizer = Tfidf_Vectorizer(features_config)
    elif feature_type == FeatureType.COUNT:
        logger.info("Using a Count vectorizer.")
        vectorizer = Count_Vectorizer(features_config)
    elif feature_type == FeatureType.BAGOFWORDS:
        logger.info("Using a Bag of Words vectorizer.")
        vectorizer = BagOfWords_Vectorizer(features_config)
    elif feature_type == FeatureType.WORD2VEC:
        logger.info("Using a Word2Vec vectorizer.")
        vectorizer = Word2Vec_Vectorizer(features_config)
    elif feature_type == FeatureType.GLOVE:
        logger.info("Using a GloVe vectorizer.")
        vectorizer = GloVe_Vectorizer(features_config)
    else:
        logger.error(
            f"Invalid feature extraction type '{feature_type}'. Supported types: {[ft for ft in FeatureType]}."
        )

    features = vectorizer.fit_transform(text_series)

    logger.info("Feature extraction complete.")
    return features
