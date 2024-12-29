from .base import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class CNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: list[int],
        num_classes: int,
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        conv_x = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_x, dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return torch.sigmoid(x)

class CNNModel(Model):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: list[int],
        num_classes: int,
        dropout: float = 0.5
    ) -> None:
        self._model = CNN(
            vocab_size,
            embedding_dim,
            num_filters,
            filter_sizes,
            num_classes,
            dropout
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._criterion = nn.BCELoss()
        self._optimizer = None

    def fit(self, features: any, labels: any) -> None:
        pass

    def predict(self, features: any) -> any:
        pass

    def save(self, file_name: str) -> None:
        torch.save(self._model.state_dict(), file_name)

    def load(self, file_name: str) -> None:
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Model file '{file_name}' does not exist.")
        self._model.load_state_dict(torch.load(file_name, map_location=self._device))
