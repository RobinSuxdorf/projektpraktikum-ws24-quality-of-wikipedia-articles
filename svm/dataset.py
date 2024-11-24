import os
import logging
import pandas as pd
import kagglehub


def load_data_frame(frac: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """
    Downloads and loads the Wikipedia promotional articles dataset, processes it, and returns a combined DataFrame.

    Parameters:
    frac (float): Fraction of the dataset to sample. Default is 1.0 (use the entire dataset).
    random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
    pd.DataFrame: A DataFrame containing the combined 'good' and 'promotional' articles with labels.
    """
    logging.info("Downloading dataset")
    path_to_dataset = kagglehub.dataset_download(
        "urbanbricks/wikipedia-promotional-articles"
    )
    logging.info("Dataset downloaded to: %s", path_to_dataset)
    good_df = pd.read_csv(os.path.join(path_to_dataset, "good.csv"))
    promo_df = pd.read_csv(os.path.join(path_to_dataset, "promotional.csv"))
    if frac < 1:
        good_df = good_df.sample(frac=frac, random_state=random_state)
        promo_df = promo_df.sample(frac=frac, random_state=random_state)
    good_df["label"] = "good"
    promo_df["label"] = "promotional"
    df = pd.concat([good_df, promo_df])
    logging.info("Dataset shape: %s", df.shape)
    logging.info("Dataset info: %s", df.info())
    return df


