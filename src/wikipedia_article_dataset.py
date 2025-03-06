import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix

class WikipediaArticleDataset(Dataset):
    """
    A PyTorch Dataset for handling Wikipedia articles and their corresponding labels.
    """

    def __init__(
        self,
        articles: csr_matrix,
        labels: list[int],
        device: torch.device | None = None,
    ) -> None:
        """
        Initializes the WikipediaArticleDataset

        Args:
            articles (csr_matrix): The list of articles as compressed sparse row matrix.
            labels (list[int]): The list of labels corresponding to the articles.
            device (torch.device | None, optional): The device to place tensors on. Defaults to GPU if available.

        Raises:
            ValueError: If the number of articles and labels do not match.
        """
        if articles.shape[0] != len(labels):
            raise ValueError("The number of articles must match the number of labels.")

        self._articles = articles
        self._labels = labels
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset, i.e. the number of articles.
        """
        return self._articles.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the encoded article and its label at the specified index.

        Args:
            index (int): The index of the article to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the encoded article tensor and the label tensor.
        """
        article = self._articles[index].toarray().squeeze()
        article_tensor = torch.tensor(article, dtype=torch.float, device=self._device)

        label = self._labels[index]  
        label_tensor = torch.tensor(label, dtype=torch.float, device=self._device)

        return article_tensor, label_tensor
