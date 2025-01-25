# src/preprocessing.py

import logging
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

try:
    nltk.data.find("stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text_series(
    text_series: pd.Series, preprocessing_config: dict
) -> pd.Series:
    """
    Preprocess a series of text data based on the provided configuration.

    Args:
        text_series (pd.Series): Series containing text data.
        preprocessing_config (dict): Configuration dictionary for preprocessing.

    Returns:
        pd.Series: Preprocessed text data.
    """
    logger.info(f"Preprocessing data with {preprocessing_config}")

    remove_non_word = preprocessing_config.get("remove_non_word")
    convert_lowercase = preprocessing_config.get("convert_lowercase")
    remove_stopwords = preprocessing_config.get("remove_stopwords")
    apply_stemming = preprocessing_config.get("apply_stemming")
    remove_numbers = preprocessing_config.get("remove_numbers")
    remove_whitespace = preprocessing_config.get("remove_whitespace")

    if remove_non_word:
        logger.info("Removing non-word characters.")
        text_series = text_series.str.replace(r"\W", " ", regex=True)

    if convert_lowercase:
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

    if remove_whitespace:
        logger.info("Removing leading and trailing whitespace.")
        text_series = text_series.str.strip()

    logger.info("Text data preprocessing complete.")
    return text_series
