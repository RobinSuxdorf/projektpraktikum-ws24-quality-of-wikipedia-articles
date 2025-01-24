# src/data_loader.py

import logging
from typing import Literal
import pandas as pd
from enum import StrEnum
import numpy as np

logger = logging.getLogger(__name__)


class Usecase(StrEnum):
    BINARY = "binary"
    MULTILABEL = "multilabel"


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
    logger.info(f"Loading data with {data_loader_config}")

    promo_file_path = data_loader_config.get("promo_file")
    nrows = data_loader_config.get("nrows", None)
    shuffle = data_loader_config.get("shuffle")

    if usecase == Usecase.BINARY:
        good_file_path = data_loader_config.get("good_file")
        neutral_file_path = data_loader_config.get("neutral_file")

        logger.info(
            "Loading non-promotional and promotional data for binary classification."
        )
        good_df = pd.read_csv(good_file_path, nrows=nrows)
        good_df["label"] = 0
        promo_df = pd.read_csv(promo_file_path, nrows=nrows)
        promo_df["label"] = 1
        df = pd.concat([good_df, promo_df], axis=0, ignore_index=True)
        if neutral_file_path:
            neutral_df = pd.read_csv(neutral_file_path, nrows=nrows)
            neutral_df["label"] = 2
            df = pd.concat([df, neutral_df], axis=0, ignore_index=True)

        df = df[["text", "label"]]
    elif usecase == Usecase.MULTILABEL:
        logger.info("Loading promotional data for multilabel classification.")
        promo_df = pd.read_csv(promo_file_path, nrows=nrows)

        df = promo_df.drop(columns=["url"])
    else:
        logger.error(
            f"Invalid model type '{usecase}'. Supported types: {[uc for uc in Usecase]}."
        )

    if shuffle:
        logger.info("Shuffling the data.")
        df = df.sample(frac=1).reset_index(drop=True)

    label_change_frac = data_loader_config.get("label_change_frac", 0)
    if label_change_frac > 0:
        logger.info(f"Randomly changing labels for {label_change_frac * 100}% of the data.")
        num_rows_to_change = int(len(df) * label_change_frac)
        rows_to_change = np.random.choice(df.index, num_rows_to_change, replace=False)
        
        possible_labels = [0, 1, 2] if neutral_file_path else [0, 1]
        for row in rows_to_change:
            current_label = df.loc[row, "label"]
            new_label = np.random.choice([label for label in possible_labels if label != current_label])
            df.loc[row, "label"] = new_label

    return df
