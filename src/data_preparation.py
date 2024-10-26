# src/data_preparation.py

import pandas as pd
import nltk

try:
    from nltk.corpus import stopwords

    if not stopwords.words("english"):
        nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


def preprocess_text_series(text_series):
    """
    Preprocess a pandas Series containing text data by removing non-word characters, converting to lowercase,
    and removing stopwords.

    Args:
        text_series (pd.Series): A pandas Series containing text data to preprocess.

    Returns:
        pd.Series: A pandas Series with cleaned text.
    """
    text_series = text_series.str.replace(r"\W", " ", regex=True)
    text_series = text_series.str.lower()
    text_series = text_series.apply(
        lambda x: " ".join([word for word in x.split() if word not in STOPWORDS])
    )
    return text_series


def prepare_data(promotional_file, good_file, nrows=None):
    """
    Load promotional and non-promotional text data from CSV files, preprocess the text, and combine them into a single dataset.

    Args:
        promotional_file (str): Path to the CSV file containing promotional text data.
        good_file (str): Path to the CSV file containing non-promotional text data.
        nrows (int, optional): Number of rows to read from each CSV file. Defaults to None, meaning all rows are read.

    Returns:
        pd.DataFrame: A pandas DataFrame containing preprocessed text data and corresponding labels (1 for promotional, 0 for non-promotional).
    """
    try:
        promo_df = pd.read_csv(promotional_file, nrows=nrows)
        good_df = pd.read_csv(good_file, nrows=nrows)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error reading files: {e}")

    promo_df["cleaned_text"] = preprocess_text_series(promo_df["text"])
    promo_df["label"] = 1

    good_df["cleaned_text"] = preprocess_text_series(good_df["text"])
    good_df["label"] = 0

    data = pd.concat(
        [promo_df[["cleaned_text", "label"]], good_df[["cleaned_text", "label"]]],
        axis=0,
    ).reset_index(drop=True)

    data = data.sample(frac=1).reset_index(drop=True)

    return data
