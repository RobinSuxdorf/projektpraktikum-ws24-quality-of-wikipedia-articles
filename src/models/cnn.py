from .base import Model
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(Model, nn.Module):
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

    def fit(self, features, labels):
        pass

    def predict(self, features):
        pass