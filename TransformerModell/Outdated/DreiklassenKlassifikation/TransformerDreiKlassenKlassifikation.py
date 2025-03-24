import pandas as pd
import torch
from transformers import pipeline
from datasets import load_dataset, Dataset
#https://huggingface.co/learn/nlp-course/chapter4/2
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel
import pandas
import datasets


from DataPreprocessingTest import preprocess_data

def read_data_for_triple_classification(promotional_path, neutral_path, good_path):
    """Here the data is being read in"""
    df_promo = pd.read_csv(promotional_path)
    df_good = pd.read_csv(good_path)
    df_neutral = pd.read_csv(neutral_path)
    df_promo["labels"] = 2
    df_neutral["labels"] = 1
    df_good["labels"] = 0
    df_promo = df_promo[["text", "labels"]]
    df_neutral = df_neutral[["text", "labels"]]
    df_good = df_good[["text", "labels"]]
    df_promo["text"] = preprocess_data(df_promo["text"])
    df_neutral["text"] = preprocess_data(df_neutral["text"])

    df_good["text"] = preprocess_data(df_good["text"])
    concat_prep = [df_promo, df_good, df_neutral]
    df = pd.concat(concat_prep)
    ds = datasets.Dataset.from_pandas(df)
    ds = ds.remove_columns("__index_level_0__")
    ds = ds.shuffle(seed=42)
    ds = ds.train_test_split(0.2)
    return ds, 3



def tokenizer_function(examples):
    """Is used to tokenize the data"""
    return tokenizer(examples["text"], padding = "max_length", truncation = True)



#model = BertModel.from_pretrained(model_name)

from datasets import load_dataset, concatenate_datasets


def tokenize_data(dataset_final):
    """Here the data is being tokenized"""
    tokenized_dataset = dataset_final.map(tokenizer_function, batched =True)
    tokenized_datasets = tokenized_dataset.remove_columns(["text"])
    tokenized_datasets.set_format("torch")
    return tokenized_datasets



def split_and_reduce_data(tokenized_datasets, amount_of_entries):
    """This functions reduced the data"""
    if amount_of_entries == "ALL":
        testing_data = tokenized_datasets["test"].shuffle(seed=42)
        training_data = tokenized_datasets["train"].shuffle(seed=42)
    else:
        testing_data = tokenized_datasets["test"].shuffle(seed=42).select(range(amount_of_entries))
        training_data = tokenized_datasets["train"].shuffle(seed=42).select(range(amount_of_entries))
    return testing_data, training_data

from torch.utils.data import DataLoader

def create_dataloaders(training_data, testing_data):
    """Creates the dataloaders"""
    train_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(testing_data, batch_size=batch_size)
    return train_dataloader, eval_dataloader



from CustomModelTest import DistilBertExtensionFirstVersion

from torch import optim

from transformers import get_scheduler

from accelerate.test_utils.testing import get_backend

import torch

from transformers import DistilBertConfig


model_config = DistilBertConfig()

model_config.num_labels = 3
model_config.add_cross_attention =True


from transformers import DistilBertForSequenceClassification
def create_model_and_functions(model_type_transformer, optimizer_type, num_training_steps, num_of_classes):
    """Creates the model except for the tokenizer"""
    model =  DistilBertForSequenceClassification.from_pretrained(model_type_transformer, config= model_config, ignore_mismatched_sizes=True, attn_implementation ="sdpa")
    optimizer = optimizer_type(model.parameters(), lr=5e-7, weight_decay=0.00000000000000001)
    lr_scheduler = get_scheduler(

        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps

    )
    global device
    device, _, _ = get_backend()
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, loss_function, device




def training_steps_for_model(batch, model, optimizer, lr_scheduler, loss_function):
    """One training step for the model"""
    batch = {k: v.to(device) for k, v in batch.items()}
    predictions = model(**batch)
    loss = predictions.loss
    loss.backward()

    optimizer.step()

    lr_scheduler.step()

    optimizer.zero_grad()


from metrics_test import calculate_metrics

from EvaluationTest import evaluate_model as ev

from metrics_test import calculate_multilabel
def evaluation_step_for_model(batch, batch_real_results_list, batch_predictions_list, model):
    """One evaluation step for the model"""
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():

        predictions = model(**batch)


    real_values = batch["labels"]
    for value in real_values:
        batch_real_results_list.append(value.cpu())
    for single_prediction in predictions.logits:
        batch_predictions_list.append(torch.argmax(single_prediction, dim= -1).cpu())
    return batch_real_results_list, batch_predictions_list

import matplotlib.pyplot as plt
import numpy as np
def evaluate_model(eval_dataloader, model):
    """Evaluates the model and gathers the results"""
    batch_real_results_list = []
    batch_predictions_list = []
    for batch in eval_dataloader:
        batch_real_results_list, batch_predictions_list = evaluation_step_for_model(batch, batch_real_results_list, batch_predictions_list, model)
    ev(np.array(batch_predictions_list), np.array(batch_real_results_list))
    plt.show()


from tqdm.auto import tqdm
def execute_all_methods():
    """Executes the whole flow"""
    good_path = "wikipedia_samples/good_sample.csv"
    neutral_path = "wikipedia_samples/neutral_sample.csv"
    promotional_path = "wikipedia_samples/promotional_sample.csv"


    amount_of_entries = 10000

    model_type_transformer = "distilbert-base-uncased"
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type_transformer)
    dataset_final, amount_of_classes = read_data_for_triple_classification(promotional_path, neutral_path, good_path)

    tokenized_dataset_test = tokenize_data(dataset_final)
    testing_data, training_data = split_and_reduce_data(tokenized_dataset_test, amount_of_entries)
    train_dataloader, eval_dataloader = create_dataloaders(training_data, testing_data)

    optimizer_type = optim.AdamW
    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    model, optimizer, lr_scheduler, loss_function, device = create_model_and_functions(model_type_transformer, optimizer_type, num_training_steps, amount_of_classes)



    progress_bar = tqdm(range(num_training_steps))

    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
             training_steps_for_model(batch, model, optimizer, lr_scheduler, loss_function)
             progress_bar.update(1)
        #evaluate_model(eval_dataloader)


    model.eval()
    evaluate_model(eval_dataloader, model)



batch_size = 2
execute_all_methods()




