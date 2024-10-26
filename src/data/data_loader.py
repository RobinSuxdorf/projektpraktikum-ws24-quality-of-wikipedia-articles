from typing import Callable
import pandas as pd


def load_data(
    good_file_path: str,
    promo_file_path: str,
    transform: Callable[[pd.Series], str] | None = None,
) -> pd.DataFrame:
    """
    Load, merges and optionally transforms promotional and non-promotional wikipedia articles.

    Args:
        good_file_path (str): Path to the CSV file containing the 'good' articles.
        promo_file_path (str): Path to the CSV file containing the 'promotional' articles.
        transform (Callable[[pd.Series], str] | None): A function that applies
            a transformation to each row of the DataFrame. Expected to return a
            string, which will be used as the value for the "text" column.

    Returns:
        (pd.DataFrame): A combined DataFrame with data from both files, labeled and optionally transformed.
    """
    try:
        good_df = pd.read_csv(good_file_path, index_col=0)
        promo_df = pd.read_csv(promo_file_path, index_col=0)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error reading files: {e}")

    good_df["label"] = 0
    promo_df["label"] = 1

    df = pd.concat((good_df, promo_df), axis=0, ignore_index=True)

    if transform:
        df["text"] = df.apply(transform, axis=1)

    return df[["text", "label"]]
