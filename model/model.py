import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        output_size: int,
        num_layers: int
    ) -> None:
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) # (batch_size, seq_length, embedding_dim)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #.to(x.device)  # (num_layers, batch_size, hidden_size)
        out, _ = self.rnn(x, h_0) # (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out) # (batch_size, output_size)
        out = self.sigmoid(out)
        return out
