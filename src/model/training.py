import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()
    train_loss = 0.0

    for articles, labels in train_dataloader:
        articles, labels = articles.to(device), labels.to(device)
        outputs = model(articles)
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * articles.size(0)

    train_loss /= len(train_dataloader.dataset)
    return train_loss


def evaluate_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for articles, labels in test_dataloader:
            articles, labels = articles.to(device), labels.to(device)

            outputs = model(articles)
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * articles.size(0)

            predicted_labels = (outputs >= 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

    test_loss /= len(test_dataloader.dataset)
    accuracy = correct_predictions / total_predictions
    return test_loss, accuracy
