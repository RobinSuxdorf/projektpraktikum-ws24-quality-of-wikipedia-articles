# src/feature_engineering.py

import logging
from sklearn.feature_extraction.text import TfidfVectorizer

logging.getLogger(__name__)


def extract_features(text_series, max_features=1000):
    """
    Extract numerical features from a pandas Series containing text data using TF-IDF Vectorizer.

    Args:
        text_series (pd.Series): A pandas Series containing cleaned text data to transform into features.
        max_features (int, optional): Maximum number of features for the vectorizer. Defaults to 1000.

    Returns:
        tuple: A tuple containing the extracted features (sparse matrix) and the fitted TF-IDF vectorizer.
    """
    logging.info("Extracting features from text data using TF-IDF Vectorizer.")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    features = vectorizer.fit_transform(text_series)
    logging.info("Feature extraction complete.")
    return features, vectorizer
