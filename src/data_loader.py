# src/data_loader.py

import logging
import pandas as pd
from typing import Literal

logger = logging.getLogger(__name__)


def load_data(
    data_loader_config: dict, usecase: Literal["binary", "multilabel"]
) -> pd.DataFrame:
    """
    Load data for either binary classification or multilabel classification use cases.

    Args:
        data_loader_config (dict): Configuration dictionary for the data loader.
        usecase (Literal["binary", "multilabel"]): The type of classification task.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded text data and corresponding labels.
    """
    if usecase == "binary":
        good_file_path = data_loader_config.get("good_file")
        promo_file_path = data_loader_config.get("promo_file")
        nrows = data_loader_config.get("nrows", None)
        shuffle = data_loader_config.get("shuffle")

        logger.info("Loading non-promotional and promotional data.")
        good_df = pd.read_csv(good_file_path, nrows=nrows)
        promo_df = pd.read_csv(promo_file_path, nrows=nrows)

        logger.info("Assigning binary labels to the data.")
        good_df["label"] = 0
        promo_df["label"] = 1

        df = pd.concat([good_df, promo_df], axis=0, ignore_index=True)

        if shuffle:
            logger.info("Shuffling the combined data.")
            df = df.sample(frac=1).reset_index(drop=True)

        logger.info("Data loading for binary classification complete.")
        return df[["text", "label"]]

    elif usecase == "multilabel":
        promo_file_path = data_loader_config.get("promo_file")
        nrows = data_loader_config.get("nrows", None)
        shuffle = data_loader_config.get("shuffle")

        logger.info("Loading promotional data for multilabel classification.")
        promo_df = pd.read_csv(promo_file_path, nrows=nrows)

        logger.info("Dropping unnecessary columns.")
        promo_df.drop(columns=["url"], inplace=True)

        if shuffle:
            logger.info("Shuffling the promotional data.")
            promo_df = promo_df.sample(frac=1).reset_index(drop=True)

        logger.info("Data loading for multilabel classification complete.")
        return promo_df
