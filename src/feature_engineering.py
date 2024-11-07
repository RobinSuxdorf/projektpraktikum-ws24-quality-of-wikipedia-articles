# src/feature_engineering.py

import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logging.getLogger(__name__)


def get_vectorizer(vectorizer_type, max_features, ngram_range, min_df, max_df):
    """
    Get the specified vectorizer.

    Args:
        vectorizer_type (str): The type of vectorizer to use ('tfidf' or 'count').
        max_features (int): Maximum number of features for the vectorizer.
        ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams to be extracted.
        min_df (float): Minimum document frequency for the terms.
        max_df (float): Maximum document frequency for the terms.

    Returns:
        Vectorizer: Configured vectorizer.
    """
    ngram_range = tuple(map(int, ngram_range.split(",")))
    if vectorizer_type == "count":
        return CountVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
    elif vectorizer_type == "tfidf":
        return TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            # ngram_range=ngram_range,
            # min_df=min_df,
            # max_df=max_df,
        )
    else:
        raise ValueError("Invalid vectorizer type. Please choose 'tfidf' or 'count'.")
