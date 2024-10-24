import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        embedding_dim: int, 
        hidden_dim: int, 
        output_dim: int
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor): # (length, 1) TODO: Specify exact datatype
        embedded = self.embedding(x) # (length, 1, embed size)

        output, hidden = self.rnn(embedded) # (length, 1, hidden dim), (1, 1, hidden dim)

        out = self.fc(hidden) # (1, 1, 2)

        return out

    def predict(self, text: str) -> int:
        tokenized = tokenizer.encode(text)
        tensor = torch.LongTensor(tokenized)

        tensor = tensor.unsqueeze(1)
        prediction = self(tensor)
        preds, ind = torch.max(F.softmax(prediction.squeeze(0), dim=-1), 1)

        return ind.item()
