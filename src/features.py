# src/features.py

import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from enum import StrEnum

logger = logging.getLogger(__name__)


class FeatureType(StrEnum):
    TFIDF = "tfidf"
    COUNT = "count"


def get_features(text_series: pd.Series, features_config: dict):
    """
    Get features based on the provided configuration.

    Args:
        text_series (pd.Series): Series containing text data.
        features_config (dict): Configuration dictionary for the feature extraction.

    Returns:
        features (array-like): Input features for training the model.
    """
    feature_type = features_config.get("type")
    max_features = features_config.get("max_features")
    ngram_range = tuple(features_config.get("ngram_range"))
    min_df = features_config.get("min_df")
    max_df = features_config.get("max_df")

    if feature_type == FeatureType.TFIDF:
        logger.info("Using a TFIDF vectorizer.")
        sublinear_tf = features_config.get("sublinear_tf")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
        )
    elif feature_type == FeatureType.COUNT:
        logger.info("Using a count vectorizer.")
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
    else:
        logger.error(
            f"Invalid feature extraction type '{feature_type}'. Supported types: {[ft for ft in FeatureType]}."
        )

    features = vectorizer.fit_transform(text_series)

    logger.info("Feature extraction complete.")
    return features
