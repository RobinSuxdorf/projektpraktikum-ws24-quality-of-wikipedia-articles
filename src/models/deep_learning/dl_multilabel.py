"""Model class definitions for a neural network for multilabel classification.

Author: Robin Suxdorf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNeuralNetworkModel


class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        """
        Initializes the neural network.

        Args:
            input_dim (int): The number of input features.
            num_classes (int): The number of output classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = F.relu(self.fc1(x))  # (batch_size, 512)
        x = self.dropout(x)  # (batch_size, 512)
        x = self.fc2(x)  # (batch_size, num_classes)
        return x


def multilabel_predict_fn(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts logits to binary class predictions for multilabel classification.

    Each output entry represents a separate label, and sigmoid activation is
    applied to get probabilities, which are thresholded at 0.5

    Args:
        logits (torch.Tensor): Logits tensor of shape (batch_size, num_classes)

    Returns:
        torch.Tensor: Binary predictions of shape (batch_size, num_classes)
        where each element is either 0 or 1.
    """
    probs = torch.sigmoid(logits)
    return (probs > 0.5).int()


class MultilabelNeuralNetworkModel(BaseNeuralNetworkModel):
    """
    A multilabel classification model using a feedforward neural network.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Intializes the multilabel classification model.

        Args:
            input_dim (int): The number of input features.
        """
        super().__init__(
            NeuralNetwork(input_dim, 5),
            criterion=nn.BCEWithLogitsLoss(),
            predict_fn=multilabel_predict_fn,
            label_dtype=torch.float,
        )
