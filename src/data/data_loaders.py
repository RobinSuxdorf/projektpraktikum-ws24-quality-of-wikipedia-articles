from typing import Callable
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .wikipedia_article_dataset import WikipediaArticleDataset


def load_data(
    good_file_path: str,
    promo_file_path: str
) -> pd.DataFrame:
    try:
        good_df = pd.read_csv(good_file_path, index_col=0)
        promo_df = pd.read_csv(promo_file_path, index_col=0)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error reading files: {e}")

    good_df["label"] = 0
    promo_df["label"] = 1

    df = pd.concat((good_df, promo_df), axis=0, ignore_index=True)

    df = df.sample(frac=1).reset_index(drop=True)

    return df[["text", "label"]]

def get_data_loaders(
    df: pd.DataFrame,
    encode: Callable[[str], list[int]],
    max_length: int,
    batch_size: int = 16,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> tuple[DataLoader, DataLoader]:
    texts = list(df["text"])
    labels = list(df["label"])

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_dataset = WikipediaArticleDataset(train_texts, train_labels, encode, max_length, device=device)
    test_dataset = WikipediaArticleDataset(test_texts, test_labels, encode, max_length, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader