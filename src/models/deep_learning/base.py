"""Base class for neural network models.

Author: Robin Suxdorf
"""

import os
from collections.abc import Callable

from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from src.wikipedia_article_dataset import WikipediaArticleDataset
from src.models.base import Model


class BaseNeuralNetworkModel(Model):
    """
    A base class for neural network models that provides functionality for training,
    prediction, saving and loading models.
    """

    def __init__(
        self,
        neural_network: nn.Module,
        criterion: nn.Module,
        predict_fn: Callable[[torch.Tensor], torch.Tensor],
        label_dtype: torch.dtype,
    ) -> None:
        """
        Initializes the neural network model.

        Args:
            neural_network (nn.Module): The neural network model.
            criterion (nn.Module): The loss function used for training.
            predict_fn (Callable[[torch.Tensor], torch.Tensor]): Function to process model output into predictions.
            label_dtype (torch.dtype): Data type for labels.
        """
        self._model = neural_network
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._criterion = criterion
        self._predict_fn = predict_fn
        self._label_dtype = label_dtype

    def _train_one_epoch(
        self, train_dataloader: DataLoader, optimizer: optim.Optimizer
    ) -> float:
        """
        Trains the model for a single epoch.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            optimizer (optim.Optimizer): Optimizer used for training.

        Returns:
            float: Average loss for the epoch.
        """
        self._model.train()
        total_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(self._device)
            labels = labels.to(self._device).type(self._label_dtype)

            optimizer.zero_grad()

            logits = self._model(inputs)

            loss = self._criterion(logits, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_dataloader)

    def fit(
        self,
        features: csr_matrix,
        labels: list[int],
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
    ) -> None:
        """
        Trains the neural network model.

        Args:
            features (csr_matrix): Sparse matrix containing input features.
            labels (list[int]): List of labels corresponding to input features.
            learning_rate (float): Learning rate for the optimizer.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        train_dataset = WikipediaArticleDataset(
            features,
            labels,
            device=self._device,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            avg_loss = self._train_one_epoch(train_dataloader, optimizer)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def predict(self, features: csr_matrix) -> list:
        tensors = [
            torch.tensor(
                article.toarray().squeeze(), dtype=torch.float, device=self._device
            )
            for article in features
        ]
        input_batch = torch.stack(tensors)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(input_batch)
            predictions = self._predict_fn(logits)

        return predictions.cpu().tolist()

    def save(self, file_name: str) -> None:
        torch.save(self._model.state_dict(), file_name)

    def load(self, file_name: str) -> None:
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Model file '{file_name}' does not exist.")
        self._model.load_state_dict(torch.load(file_name, map_location=self._device))
