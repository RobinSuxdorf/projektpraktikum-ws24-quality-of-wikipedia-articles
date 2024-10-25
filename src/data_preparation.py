# src/data_preparation.py

import pandas as pd
import re
import nltk

# Load stopwords once
try:
    from nltk.corpus import stopwords

    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


def preprocess_text_series(text_series):
    text_series = text_series.str.replace(r"\W", " ", regex=True)
    text_series = text_series.str.lower()
    text_series = text_series.apply(
        lambda x: " ".join([word for word in x.split() if word not in STOPWORDS])
    )
    return text_series


def prepare_data(promotional_file, good_file, nrows=None):
    """
    Loads the first 10 lines of data from the CSV files, cleans up the texts and returns them as a DataFrame.
    """
    promo_df = pd.read_csv(promotional_file, nrows=nrows)
    good_df = pd.read_csv(good_file, nrows=nrows)

    promo_df["cleaned_text"] = preprocess_text_series(promo_df["text"])
    promo_df["label"] = 1

    good_df["cleaned_text"] = preprocess_text_series(good_df["text"])
    good_df["label"] = 0

    data = pd.concat(
        [promo_df[["cleaned_text", "label"]], good_df[["cleaned_text", "label"]]],
        axis=0,
    ).reset_index(drop=True)

    return data
