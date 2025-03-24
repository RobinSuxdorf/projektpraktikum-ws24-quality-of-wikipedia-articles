"""Creates all model components

Authors: Emmanuelle Steenhof"""


from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from transformers import get_scheduler
from accelerate.test_utils.testing import get_backend
import torch

def create_dataloaders(training_data, testing_data, batch_size):
    """This function creates the dataloaders"""
    train_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(testing_data, batch_size=batch_size)
    return train_dataloader, eval_dataloader


def create_model_and_functions(model_type_transformer, optimizer_type, num_training_steps, num_of_classes, model_config):
    """This function creates the whole model except for the dataloader"""
    model = DistilBertForSequenceClassification.from_pretrained(model_type_transformer, config=model_config,
                                                                ignore_mismatched_sizes=True)
    optimizer = optimizer_type(model.parameters(), lr=5e-7, weight_decay=0.00000000000000001)
    lr_scheduler = get_scheduler(

        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps

    )
    global device
    device, _, _ = get_backend()
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, loss_function, device
