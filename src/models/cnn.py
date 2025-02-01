from .base import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from src.wikipedia_article_dataset import text_to_tensor, WikipediaArticleDataset
import tiktoken
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
        # x: (batch_size, sequence_length)
        x = self.embedding(x) # (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1) # (batch_size, embedding_dim, sequence_length)

        conv_x = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs] # list of (batch_size, num_filters)
        x = torch.cat(conv_x, dim=1) # (batch_size, num_filters * len(filter_sizes))

        x = self.dropout(x)
        x = self.fc(x) # (batch_size, num_classes)

        return torch.sigmoid(x)

class CNNModel(Model):
    def __init__(
        self,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: list[int],
        num_classes: int,
        max_length: int,
        dropout: float = 0.5
    ) -> None:
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

        self._model = CNN(
            self._tokenizer.n_vocab,
            embedding_dim,
            num_filters,
            filter_sizes,
            num_classes,
            dropout
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._max_length = max_length

    def _train_one_epoch(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> float:
        model.train()
        total_loss = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)

            labels = labels.long()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_dataloader)

    def fit(
        self, 
        features: any, 
        labels: any,
        learning_rate: float,
        num_epochs: int, 
        batch_size: int
    ) -> None:

        train_dataset = WikipediaArticleDataset(
            features,
            labels,
            self._tokenizer.encode,
            self._max_length
        )

        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

        for i in range(num_epochs):
            avg_loss = self._train_one_epoch(self._model, train_dataloader, criterion, optimizer)
            print(f"Epoch: {i}, loss: {avg_loss}")

    def predict(self, features: any) -> any:
        tensors = [text_to_tensor(article, self._tokenizer.encode, self._max_length, self._device) for article in features]
        input_batch = torch.stack(tensors)

        self._model.eval()

        with torch.no_grad():
            predictions = self._model(input_batch)

        predicted_classes = torch.argmax(predictions, dim=1)

        return predicted_classes.cpu().tolist()

    def save(self, file_name: str) -> None:
        torch.save(self._model.state_dict(), file_name)

    def load(self, file_name: str) -> None:
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Model file '{file_name}' does not exist.")
        self._model.load_state_dict(torch.load(file_name, map_location=self._device))
