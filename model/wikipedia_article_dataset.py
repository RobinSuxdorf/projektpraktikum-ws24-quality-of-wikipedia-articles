from typing import Callable
import torch
from torch.utils.data import Dataset


class WikipediaArticleDataset(Dataset):
    def __init__(
        self,
        articles: list[str],
        labels: list[int],
        transform: Callable[[str], list[int]]
    ) -> None:
        self._articles = articles
        self._labels = labels
        self._transform = transform

    def __len__(self) -> int:
        return len(self._articles)

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.Tensor]:
        article = self._articles[idx]
        label = self._labels[idx]

        if self._transform:
            article = self._transform(article)

        return torch.LongTensor(article), torch.Tensor(label)
