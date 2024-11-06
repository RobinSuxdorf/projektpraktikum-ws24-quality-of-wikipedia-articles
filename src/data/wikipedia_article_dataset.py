from typing import Callable
import torch
from torch.utils.data import Dataset


class WikipediaArticleDataset(Dataset):
    def __init__(
        self,
        articles: list[str],
        labels: list[int],
        encode: Callable[[str], list[int]],
        max_length: int
    ) -> None:
        self._articles = articles
        self._labels = labels
        self._encode = encode
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._articles)

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.Tensor]:
        article = self._articles[idx]
        label = self._labels[idx]

        encoded = self._encode(article)

        encoded = encoded[:self._max_length]

        if len(encoded) < self._max_length:
            encoded += [0] * (self._max_length - len(encoded))

        return torch.tensor(encoded), torch.tensor(label, dtype=torch.float)
