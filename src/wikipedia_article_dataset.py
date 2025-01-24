from collections.abc import Callable
import torch
from torch.utils.data import Dataset


def text_to_tensor(text: str, encode: Callable[[str], list[int]], max_length: int, device: torch.device) -> torch.Tensor:
    """
    Converts a text string into a padded tensor of token indices for model input.

    Args:
        text (str): The input text to be tokenized and converted to a tensor.
        encode (Callable[[str], list[int]]): A function that tokenizes the input text.
        max_length (int): The maxmimum length of the resulting tensor. If the tokenized sequence is shorter, it will be padded with zeros. If it's longer, it will be truncated.
        device (torch.device): The target device where the resulting tensor will be stored (e.g., "cpu" or "cuda").

    Returns:
        torch.Tensor: A 1D tensor of shape `(max_length,)` containing token IDs. Shorter sequences are zero-padded, and longer sequences are truncated.
    """
    encoded_article = encode(text)

    encoded_article = encoded_article[: max_length]

    if len(encoded_article) < max_length:
        encoded_article += [0] * (max_length - len(encoded_article))

    return torch.tensor(
        encoded_article, dtype=torch.long, device=device
    )

class WikipediaArticleDataset(Dataset):
    """
    A PyTorch Dataset for handling Wikipedia articles and their corresponding labels.
    """

    def __init__(
        self,
        articles: list[str],
        labels: list[int],
        encode: Callable[[str], list[int]],
        max_length: int,
        device: torch.device | None = None,
    ) -> None:
        """
        Initializes the WikipediaArticleDataset

        Args:
            articles (list[str]): The list of articles.
            labels (list[int]): The list of labels corresponding to the articles.
            encode (Callable[[str], list[int]]): A function to encode an article into a list of integers, e.g. a tokenizer.
            max_length (int): The maximum length of the encoded articles.
            device (torch.device | None, optional): The device to place tensors on. Defaults to GPU if available.

        Raises:
            ValueError: If the number of articles and labels do not match.
        """
        if len(articles) != len(labels):
            raise ValueError("The nuber of articles must match the number of labels.")

        self._articles = articles
        self._labels = labels
        self._encode = encode
        self._max_length = max_length
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset, i.e. the number of articles.
        """
        return len(self._articles)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the encoded article and its label at the specified index.

        Args:
            index (int): The index of the article to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the encoded article tensor and the label tensor.
        """
        article = self._articles[index]
        label = self._labels[index]

        article_tensor = text_to_tensor(article, self._encode, self._max_length, self._device)

        label_tensor = torch.tensor(label, dtype=torch.float, device=self._device)
        return article_tensor, label_tensor
