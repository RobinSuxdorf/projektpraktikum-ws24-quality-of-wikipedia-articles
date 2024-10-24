from typing import Callable
import torch
from torch.utils.data import Dataset


class WikipediaArticleDataset(Dataset):
    def __init__(
        self,
        articles: list[str],
        labels: list[int],
        tokenizer: Callable[[str], list[int]] # TODO: Change to transform
    ) -> None:
        self._articles = [tokenizer(article) for article in articles]
        self._labels = labels

    def __len__(self) -> int:
        return len(self._articles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.LongTensor(self._articles[idx]), torch.Tensor(self._labels[idx])
