# src/feature_engineering.py

import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logging.getLogger(__name__)


def get_vectorizer(vectorizer_config: dict):
    """
    Get a vectorizer based on the provided configuration.

    Args:
        vectorizer_config (dict): Configuration dictionary for the vectorizer.

    Returns:
        Vectorizer: Configured vectorizer.
    """
    vectorizer_type = vectorizer_config.get("type")
    max_features = vectorizer_config.get("max_features")
    ngram_range = tuple(vectorizer_config.get("ngram_range"))
    min_df = vectorizer_config.get("min_df")
    max_df = vectorizer_config.get("max_df")

    if vectorizer_type == "tfidf":
        return TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
    elif vectorizer_type == "count":
        return CountVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
    else:
        raise ValueError("Invalid vectorizer type. Please choose 'tfidf' or 'count'.")
