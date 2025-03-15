import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNeuralNetworkModel


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def multilabel_predict_fn(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > 0.5).int()

class MultilabelNeuralNetworkModel(BaseNeuralNetworkModel):
    def __init__(
        self,
        input_dim: int
    ) -> None:
        super().__init__(
            NeuralNetwork(input_dim, 5),
            criterion=nn.BCEWithLogitsLoss(),
            predict_fn=multilabel_predict_fn,
            label_dtype=torch.float,
        )
