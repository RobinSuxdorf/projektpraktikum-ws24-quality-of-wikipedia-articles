# src/features.py

import logging
from enum import StrEnum
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
    logger.info(f"Getting features with {features_config}.")

    feature_type = features_config.get("type")
    ngram_range = tuple(features_config.get("ngram_range"))
    max_df = features_config.get("max_df")
    min_df = features_config.get("min_df")
    max_features = features_config.get("max_features")

    if feature_type == FeatureType.TFIDF:
        logger.info("Using a TF-IDF vectorizer.")
        sublinear_tf = features_config.get("sublinear_tf")
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
    elif feature_type == FeatureType.COUNT:
        logger.info("Using a Count vectorizer.")
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
        )
    else:
        logger.error(
            f"Invalid feature extraction type '{feature_type}'. Supported types: {[ft for ft in FeatureType]}."
        )

    features = vectorizer.fit_transform(text_series)

    logger.info("Feature extraction complete.")
    return features
