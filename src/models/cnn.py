import os
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import tiktoken

from src.wikipedia_article_dataset import text_to_tensor, WikipediaArticleDataset
from .base import Model


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
        # x: (batch_size, sequence_length)
        x = self.embedding(x) # (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1) # (batch_size, embedding_dim, sequence_length)

        conv_x = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs] # list of (batch_size, num_filters)
        x = torch.cat(conv_x, dim=1) # (batch_size, num_filters * len(filter_sizes))

        x = self.dropout(x)
        x = self.fc(x) # (batch_size, num_classes)

        return x

class BaseCNNModel(Model):
    def __init__(
        self,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: list[int],
        max_length: int,
        num_classes: int,
        dropout: float,
        criterion: nn.Module,
        predict_fn: Callable[[torch.Tensor], torch.Tensor],
        label_dtype: torch.dtype
    ) -> None:
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._model = CNN(
            vocab_size=self._tokenizer.n_vocab,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            num_classes=num_classes,
            dropout=dropout
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._max_length = max_length
        self._criterion = criterion
        self._predict_fn = predict_fn
        self._label_dtype = label_dtype

    def _train_one_epoch(
        self,
        train_dataloader: DataLoader,
        optimizer: optim.Optimizer
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
        features,
        labels,
        learning_rate: float,
        num_epochs: int,
        batch_size: int
    ) -> None:
        train_dataset = WikipediaArticleDataset(
            features,
            labels,
            self._tokenizer.encode,
            self._max_length,
            device=self._device
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            avg_loss = self._train_one_epoch(train_dataloader, optimizer)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def predict(self, features) -> list:
        tensors = [
            text_to_tensor(article, self._tokenizer.encode, self._max_length, self._device)
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

def binary_predict_fn(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)

class CNNModel(BaseCNNModel):
    def __init__(
        self,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: list[int],
        max_length: int,
        dropout: float = 0.5
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            max_length=max_length,
            num_classes=2,
            dropout=dropout,
            criterion=nn.CrossEntropyLoss(),
            predict_fn=binary_predict_fn,
            label_dtype=torch.long
        )

def multilabel_predict_fn(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > 0.5).int()

class MultilabelCNNModel(BaseCNNModel):
    def __init__(
        self,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: list[int],
        max_length: int,
        dropout: float = 0.5
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            max_length=max_length,
            num_classes=5,
            dropout=dropout,
            criterion=nn.BCEWithLogitsLoss(),
            predict_fn=multilabel_predict_fn,
            label_dtype=torch.float
        )
