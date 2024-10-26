from typing import Callable
import torch
from torch.utils.data import Dataset


class WikipediaArticleDataset(Dataset):
    """
    A custom Dataset for Wikipedia articles and their labels, with optional transformations applied to each article.

    Args:
        articles (list[str]): A list of articles represented as strings.
        labels (list[int]): A list of integer labels corresponding to each article.
        encode (Callable[[str], list[int]]): A function for encoding the articles.
        transform (Callable[[str], str] | None): A function for transforming the articles.
    """

    def __init__(
        self,
        articles: list[str],
        labels: list[int],
        encode: Callable[[str], list[int]],
        transform: Callable[[str], str] | None = None,
    ) -> None:
        self._articles = articles
        self._labels = labels
        self._encode = encode
        self._transform = transform

    def __len__(self) -> int:
        """
        Returns the number of articles in the dataset.

        Returns:
            int: The total number of articles.
        """
        return len(self._articles)

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.Tensor]:
        """
        Retrieves a transformed article and its label as tensors.

        Args:
            idx (int): The index of the article to retrieve.

        Returns:
            tuple[torch.LongTensor, torch.Tensor]: A tuple containing the article as tensor and the label as tensor.
        """
        article = self._articles[idx]
        label = self._labels[idx]

        if self._transform:
            article = self._transform(article)

        encoded = self._encode(article)

        return torch.LongTensor(encoded), torch.Tensor(label)
