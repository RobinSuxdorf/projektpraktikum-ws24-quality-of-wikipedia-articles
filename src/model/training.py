from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # Ensure labels are the right shape: [batch_size, 1]
        labels = labels.float().unsqueeze(1)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss
   


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
):
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid for probability
            
            # Collect predictions and true labels for accuracy calculation
            test_preds.extend(outputs.round().cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(test_labels, test_preds)
    return acc
