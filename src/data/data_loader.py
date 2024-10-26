from typing import Callable
import pandas as pd


def load_data(
    good_file_path: str, 
    promo_file_path: str, 
    transform: Callable[[pd.Series], str] | None = None
) -> pd.DataFrame:
    try:
        good_df = pd.read_csv(good_file_path, index_col=0)
        promo_df = pd.read_csv(promo_file_path, index_col=0)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error reading files: {e}")

    good_df["label"] = 0
    promo_df["label"] = 1

    df = pd.concat((good_df, promo_df), axis=0).reset_index(drop=True)

    if transform:
        df["text"] = df.apply(transform, axis=1)

    return df
