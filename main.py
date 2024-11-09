import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from src.data import load_data, get_data_loaders
from src.model import TextCNN, train_one_epoch, evaluate_model


number_of_elements = 100
embedding_dim = 256
num_filters = 100
filter_sizes = [3, 4, 5]
max_length = 400
num_classes = 2
tokenizer = tiktoken.get_encoding("cl100k_base")
num_epochs = 5
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    df = load_data("../good.csv", "../promotional.csv")
    df = df.head(number_of_elements)

    train_loader, test_loader = get_data_loaders(df, tokenizer.encode, max_length, batch_size=batch_size, device = device)

    model = TextCNN(
        vocab_size=tokenizer.n_vocab, 
        embedding_dim=embedding_dim, 
        num_classes=num_classes,
        num_filters=num_filters, 
        filter_sizes=filter_sizes, 
        max_length=max_length
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        acc = evaluate_model(model, test_loader, device=device)
        print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
