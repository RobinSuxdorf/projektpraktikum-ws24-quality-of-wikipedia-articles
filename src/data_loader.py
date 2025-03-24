"""Module for loading data for binary and multilabel classification tasks.

Author: Sebastian Bunge
"""

import logging
from typing import Literal
import pandas as pd
from enum import StrEnum
import numpy as np

logger = logging.getLogger(__name__)


# Define the supported use cases
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

    # Load common configuration parameters
    promo_file_path = data_loader_config.get("promo_file")
    nrows = data_loader_config.get("nrows", None)
    shuffle = data_loader_config.get("shuffle")

    if usecase == Usecase.BINARY:
        # Load additional configuration parameters
        good_file_path = data_loader_config.get("good_file")
        neutral_file_path = data_loader_config.get("neutral_file")

        # Load data and assign labels
        logger.info(
            "Loading non-promotional and promotional data for binary classification."
        )
        good_df = pd.read_csv(good_file_path, nrows=nrows)
        good_df["label"] = 0
        promo_df = pd.read_csv(promo_file_path, nrows=nrows)
        promo_df["label"] = 1

        # Concatenate data
        df = pd.concat([good_df, promo_df], axis=0, ignore_index=True)

        ##################################################################################
        # Add neutral data if available
        # Author: Johannes Krämer
        if neutral_file_path:
            neutral_df = pd.read_csv(neutral_file_path, nrows=nrows)
            neutral_df["label"] = 2
            df = pd.concat([df, neutral_df], axis=0, ignore_index=True)
        ##################################################################################

        # Select only relevant columns
        df = df[["text", "label"]]
    elif usecase == Usecase.MULTILABEL:
        # Load data
        logger.info("Loading promotional data for multilabel classification.")
        promo_df = pd.read_csv(promo_file_path, nrows=nrows)

        # Select only relevant columns
        df = promo_df.drop(columns=["url", "id", "title"], errors="ignore")
    else:
        logger.error(
            f"Invalid model type '{usecase}'. Supported types: {[uc for uc in Usecase]}."
        )

    # Optionally shuffle the data
    if shuffle:
        logger.info("Shuffling the data.")
        df = df.sample(frac=1).reset_index(drop=True)

    ##################################################################################
    # Applies random label flipping on a fraction of the data for simulation of label noise.
    # Author: Johannes Krämer
    label_change_frac = data_loader_config.get("label_change_frac", 0)
    if label_change_frac > 0:
        logger.info(
            f"Randomly changing labels for {label_change_frac * 100}% of the data."
        )
        num_rows_to_change = int(len(df) * label_change_frac)
        rows_to_change = np.random.choice(df.index, num_rows_to_change, replace=False)

        possible_labels = [0, 1, 2] if neutral_file_path else [0, 1]
        for row in rows_to_change:
            current_label = df.loc[row, "label"]
            new_label = np.random.choice(
                [label for label in possible_labels if label != current_label]
            )
            df.loc[row, "label"] = new_label
    ##################################################################################

    return df
