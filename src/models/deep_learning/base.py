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
    def __init__(
        self,
        neural_network: nn.Module,
        criterion: nn.Module,
        predict_fn: Callable[[torch.Tensor], torch.Tensor],
        label_dtype: torch.dtype,
    ) -> None:
        self._model = neural_network
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._criterion = criterion
        self._predict_fn = predict_fn
        self._label_dtype = label_dtype

    def _train_one_epoch(
        self, train_dataloader: DataLoader, optimizer: optim.Optimizer
    ) -> float:
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
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def predict(self, features) -> list:
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
