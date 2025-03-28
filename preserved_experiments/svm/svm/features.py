"""Author: Johannes Krämer"""

import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from typing import Tuple


def extract_features(df: pd.DataFrame) -> Tuple[csr_matrix, pd.Series]:
    """
    Extracts TF-IDF features from the text column of the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the text data and labels.

    Returns:
    tuple: A tuple containing the TF-IDF feature matrix (X) and the labels (y).
    """
    logging.info("Extracting features")
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, sublinear_tf=True)
    X = tfidf.fit_transform(df["text"])
    y = df["label"]
    logging.info("Feature matrix shape: %s", X.shape)
    logging.info("Labels shape: %s", y.shape)
    return X, y


def extract_features_with_categories(
    df: pd.DataFrame,
) -> Tuple[csr_matrix, pd.DataFrame]:
    """
    Extracts TF-IDF features from the text column of the DataFrame and includes additional labels for promotional articles.

    Parameters:
    df (pd.DataFrame): DataFrame containing the text data, labels, and additional features.

    Returns:
    tuple: A tuple containing the TF-IDF feature matrix (X) and the labels DataFrame (y).
    """
    logging.info("Extracting features")
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, sublinear_tf=True)
    X = tfidf.fit_transform(df["text"])
    df["label"] = df["label"].map({"good": 0, "promotional": 1})
    df[["advert", "coi", "fanpov", "pr", "resume"]] = df[
        ["advert", "coi", "fanpov", "pr", "resume"]
    ].fillna(0)
    y = df[["label", "advert", "coi", "fanpov", "pr", "resume"]]
    logging.info("Feature matrix shape: %s", X.shape)
    logging.info("Labels shape: %s", y.shape)
    return X, y


def extract_features_promotional_categories(
    df: pd.DataFrame,
) -> Tuple[csr_matrix, pd.DataFrame]:
    """
    Extracts TF-IDF features from the text column of the DataFrame and includes additional labels for promotional articles.
    Only includes promotional articles and drops the label column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the text data, labels, and additional features.

    Returns:
    tuple: A tuple containing the TF-IDF feature matrix (X) and the labels DataFrame (y).
    """
    logging.info("Extracting features for promotional articles")
    df = df[df["label"] == "promotional"]
    df = df.drop(columns=["label"])
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, sublinear_tf=True)
    X = tfidf.fit_transform(df["text"])
    df[["advert", "coi", "fanpov", "pr", "resume"]] = df[
        ["advert", "coi", "fanpov", "pr", "resume"]
    ].fillna(0)
    y = df[["advert", "coi", "fanpov", "pr", "resume"]]
    logging.info("Feature matrix shape: %s", X.shape)
    logging.info("Labels shape: %s", y.shape)
    return X, y
