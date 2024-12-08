# src/data_loader.py

import logging
import pandas as pd
from typing import Literal

logger = logging.getLogger(__name__)


def load_data(
    data_loader_config: dict, usecase: Literal["binary", "multilabel"]
) -> pd.DataFrame:
    """_summary_

    Args:
        data_loader_config (dict): _description_
        usecase (Literal[&quot;binary&quot;, &quot;multilabel&quot;]): _description_

    Returns:
        pd.DataFrame: A pandas DataFrame containing text data and corresponding labels.
    """
    if usecase == "binary":
        good_file_path = data_loader_config.get("good_file")
        promo_file_path = data_loader_config.get("promo_file")
        nrows = data_loader_config.get("nrows", None)
        shuffle = data_loader_config.get("shuffle")

        logger.info("Loading promotional and non-promotional data.")
        good_df = pd.read_csv(good_file_path, nrows=nrows)
        promo_df = pd.read_csv(promo_file_path, nrows=nrows)

        logger.info("Assigning labels to data.")
        good_df["label"] = 0
        promo_df["label"] = 1

        df = pd.concat([good_df, promo_df], axis=0, ignore_index=True)

        if shuffle:
            logger.info("Shuffling data.")
            df = df.sample(frac=1).reset_index(drop=True)

        logger.info("Data loading complete.")
        return df[["text", "label"]]

    elif usecase == "multilabel":
        promo_file_path = data_loader_config.get("promo_file")
        nrows = data_loader_config.get("nrows", None)
        shuffle = data_loader_config.get("shuffle")

        logger.info("Loading promotional data.")
        promo_df = pd.read_csv(promo_file_path, nrows=nrows)
        promo_df.drop(columns=["url"], inplace=True)

        if shuffle:
            logger.info("Shuffling data.")
            promo_df = promo_df.sample(frac=1).reset_index(drop=True)

        logger.info("Data loading complete.")
        return promo_df
