import logging
import pandas as pd
import sklearn.feature_extraction as skfe
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
    tfidf = skfe.text.TfidfVectorizer(stop_words="english", max_df=0.7)
    X = tfidf.fit_transform(df["text"])
    y = df["label"]
    logging.info("Feature matrix shape: %s", X.shape)
    return X, y
