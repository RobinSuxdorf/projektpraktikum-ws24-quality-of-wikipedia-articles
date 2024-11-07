# src/data_preparation.py

import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

logger = logging.getLogger(__name__)

if not stopwords.words("english"):
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text_series(
    text_series: pd.Series,
    remove_stopwords=True,
    apply_stemming=True,
    remove_numbers=True,
) -> pd.Series:
    """
    Preprocess a pandas Series containing text data by removing non-word characters, converting to lowercase,
    and optionally removing stopwords, applying stemming, and removing numbers.

    Args:
        text_series (pd.Series): A pandas Series containing text data to preprocess.
        remove_stopwords (bool, optional): Whether to remove stopwords. Defaults to True.
        apply_stemming (bool, optional): Whether to apply stemming. Defaults to True.
        remove_numbers (bool, optional): Whether to remove numbers. Defaults to True.

    Returns:
        pd.Series: A pandas Series with cleaned text.
    """
    logger.info("Preprocessing text data.")

    logger.info("Removing non-word characters.")
    text_series = text_series.str.replace(r"\W", " ", regex=True)

    logger.info("Converting text to lowercase.")
    text_series = text_series.str.lower()

    if remove_stopwords:
        logger.info("Removing stopwords.")
        text_series = text_series.apply(
            lambda x: " ".join([word for word in x.split() if word not in STOPWORDS])
        )

    if apply_stemming:
        logger.info("Applying stemming.")
        text_series = text_series.apply(
            lambda x: " ".join([stemmer.stem(word) for word in x.split()])
        )

    if remove_numbers:
        logger.info("Removing numbers.")
        text_series = text_series.apply(lambda x: re.sub(r"\d+", "", x))

    logger.info("Removing leading and trailing whitespace.")
    text_series = text_series.str.strip()

    logger.info("Text data preprocessing complete.")
    return text_series
