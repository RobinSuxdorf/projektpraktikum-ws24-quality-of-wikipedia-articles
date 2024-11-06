# src/data_loader.py

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(
    good_file_path: str,
    promo_file_path: str,
    nrows: int | None = None,
    shuffle: bool = True,
) -> pd.DataFrame:
    """
    Load promotional and non-promotional text data from CSV files and combine them into a single dataset.

    Args:
        good_file_path (str): Path to the CSV file containing non-promotional text data.
        promo_file_path (str): Path to the CSV file containing promotional text data.
        nrows (int, optional): Number of rows to read from each CSV file. Defaults to None, meaning all rows are read.
        shuffle (bool, optional): Whether to shuffle the combined dataset. Defaults to True.

    Returns:
        pd.DataFrame: A pandas DataFrame containing text data and corresponding labels (1 for promotional, 0 for non-promotional).
    """
    logger.info("Loading promotional and non-promotional data.")
    try:
        good_df = pd.read_csv(good_file_path, nrows=nrows)
        promo_df = pd.read_csv(promo_file_path, nrows=nrows)
    except FileNotFoundError as e:
        logger.error(f"Error reading files: {e}")
        raise

    logger.info("Assigning labels to data.")
    good_df["label"] = 0
    promo_df["label"] = 1

    df = pd.concat([good_df, promo_df], axis=0, ignore_index=True)

    if shuffle:
        logger.info("Shuffling data.")
        df = df.sample(frac=1).reset_index(drop=True)

    logger.info("Data loading complete.")
    return df[["text", "label"]]
