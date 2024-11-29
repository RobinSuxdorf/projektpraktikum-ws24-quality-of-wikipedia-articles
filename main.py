import pandas as pd
import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from src.data import load_data, get_data_loaders
from src.model import TextCNN, train_one_epoch, evaluate_model


# number_of_elements = 100
num_classes = 2
tokenizer = tiktoken.get_encoding("cl100k_base")
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_run(
    data: pd.DataFrame,
    max_length,
    batch_size,
    embedding_dim,
    num_filters,
    filter_sizes
):
    df = data.copy(deep=True)
    # df = df.head(number_of_elements)

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

    for _ in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
        
        acc = evaluate_model(model, test_loader, device=device)

    return avg_loss, acc

def main() -> None:
    df = load_data("../good.csv", "../promotional.csv")

    embedding_dim_list = [256, 512, 1024]
    num_filters_list = [50, 100, 150, 200]
    filter_sizes_list = [[3, 4, 5], [3, 5, 7]]
    max_length_list = [200, 400, 800]
    batch_size_list = [32, 64]

    for embedding_dim in embedding_dim_list:
        for num_filters in num_filters_list:
            for filter_sizes in filter_sizes_list:
                for max_length in max_length_list:
                    for batch_size  in batch_size_list:
                        loss, acc = train_run(df, max_length, batch_size, embedding_dim, num_filters, filter_sizes)

                        print(loss, acc)

                        with open("results.txt", "a") as f:
                            f.write(f"Embedding dimension: {embedding_dim}\n")
                            f.write(f"Num filters: {num_filters}\n")
                            f.write(f"Filter sizes: {filter_sizes}\n")
                            f.write(f"Max length: {max_length}\n")
                            f.write(f"Batch size: {batch_size}\n")
                            f.write(f"Loss: {loss}\n")
                            f.write(f"Accuracy: {acc}\n")
                            f.write("\n")
    

if __name__ == "__main__":
    main()
