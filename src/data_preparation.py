# src/data_preparation.py

import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

if not stopwords.words("english"):
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))


def preprocess_text_series(text_series: pd.Series) -> pd.Series:
    """
    Preprocess a pandas Series containing text data by removing non-word characters, converting to lowercase,
    and removing stopwords.

    Args:
        text_series (pd.Series): A pandas Series containing text data to preprocess.

    Returns:
        pd.Series: A pandas Series with cleaned text.
    """
    logger.info("Preprocessing text data.")
    text_series = text_series.str.replace(r"\W", " ", regex=True)
    text_series = text_series.str.lower()
    text_series = text_series.apply(
        lambda x: " ".join([word for word in x.split() if word not in STOPWORDS])
    )
    return text_series
