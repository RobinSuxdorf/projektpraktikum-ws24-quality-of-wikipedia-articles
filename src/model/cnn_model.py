import torch
import torch.nn as nn


# Update the model's output layer (remove sigmoid for BCEWithLogitsLoss)
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters, filter_sizes, max_length):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, max_length, embedding_dim)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, embedding_dim, max_length)
        
        # Convolution + Max Pooling
        conv_x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_x, dim=1)  # Concatenate all conv outputs
        
        # Dropout and fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return self.sigmoid(x)
