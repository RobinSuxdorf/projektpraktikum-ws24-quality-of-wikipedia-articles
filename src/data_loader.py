# src/data_loader.py

import logging
import pandas as pd
from typing import Literal
from enum import StrEnum

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

        logger.info(
            "Loading non-promotional and promotional data for binary classification."
        )
        good_df = pd.read_csv(good_file_path, nrows=nrows)
        promo_df = pd.read_csv(promo_file_path, nrows=nrows)

        good_df["label"] = 0
        promo_df["label"] = 1

        df = pd.concat([good_df, promo_df], axis=0, ignore_index=True)
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

    return df
