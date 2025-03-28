"""Prepares data for training and executes the training

Authors: Emmanuelle Steenhof"""


import datasets


def split_and_reduce_data(tokenized_datasets, amount_of_entries):
    """This functions reduced the data"""
    if amount_of_entries == "ALL":
        testing_data = tokenized_datasets["test"].shuffle(seed=42)
        training_data = tokenized_datasets["train"].shuffle(seed=42)
    else:
        testing_data = tokenized_datasets["test"].shuffle(seed=42).select(range(amount_of_entries))
        training_data = tokenized_datasets["train"].shuffle(seed=42).select(range(amount_of_entries))
    return testing_data, training_data

def training_steps_for_model(batch, model, optimizer, lr_scheduler, loss_function, device):
    """This function executes one training step for the model"""
    batch = {k: v.to(device) for k, v in batch.items()}
    predictions = model(**batch)
    loss = predictions.loss
    loss.backward()

    optimizer.step()

    lr_scheduler.step()

    optimizer.zero_grad()

def train_model(num_epochs, progress_bar, train_dataloader, model, optimizer, lr_scheduler, loss_function, device):
    """Executes the training procedure of the model"""
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            training_steps_for_model(batch, model, optimizer, lr_scheduler, loss_function, device)
            progress_bar.update(1)