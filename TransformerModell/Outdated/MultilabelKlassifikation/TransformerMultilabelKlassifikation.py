import pandas as pd
# https://huggingface.co/learn/nlp-course/chapter4/2
# classifier = pipeline("text-classification", model= "bert-base-uncased")
from transformers import AutoTokenizer
import datasets

from DataPreprocessingTest import preprocess_data




def read_data_with_multilabel(promotional_path):
    df_promo = pd.read_csv(promotional_path)
    df_promo["text"] = preprocess_data(df_promo["text"])
    labels_promo = []
    """The labels are being concatenated"""
    for i in range(len(df_promo)):
        labels_of_i = []
        labels_of_i.append(float(df_promo.iloc[i]["advert"]))
        labels_of_i.append(float(df_promo.iloc[i]["coi"]))
        labels_of_i.append(float(df_promo.iloc[i]["fanpov"]))
        labels_of_i.append(float(df_promo.iloc[i]["pr"]))
        labels_of_i.append(float(df_promo.iloc[i]["resume"]))
        labels_promo.append(labels_of_i)
    """The labels column is being assigned"""
    df_promo["labels"] = labels_promo
    df_promo = df_promo.drop("advert", axis=1)
    df_promo = df_promo.drop("coi", axis=1)
    df_promo = df_promo.drop("fanpov", axis=1)
    df_promo = df_promo.drop("pr", axis=1)
    df_promo = df_promo.drop("resume", axis=1)
    df_promo = df_promo.drop("url", axis=1)
    """The dataframe is being turned into a dataset"""
    dataset_final = datasets.Dataset.from_pandas(df_promo)
    dataset_final = dataset_final.shuffle()
    dataset_final = dataset_final.train_test_split(0.2)
    print(dataset_final)
    return dataset_final, 5


def tokenizer_function(examples):
    """Is used to tokenize the articles"""
    return tokenizer(examples["text"], padding="max_length", truncation=True)


from datasets import load_dataset, concatenate_datasets

def tokenize_data(dataset_final):
    """This function is used to tokenize the data"""
    tokenized_dataset = dataset_final.map(tokenizer_function, batched=True)
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
    """This function creates the dataloaders"""
    train_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(testing_data, batch_size=batch_size)
    return train_dataloader, eval_dataloader


from torch import optim

from transformers import get_scheduler

from accelerate.test_utils.testing import get_backend

import torch

from transformers import DistilBertConfig

model_config = DistilBertConfig()

model_config.num_labels = 5
model_config.add_cross_attention = True
model_config.activation = 'gelu'

from transformers import DistilBertForSequenceClassification


def create_model_and_functions(model_type_transformer, optimizer_type, num_training_steps, num_of_classes):
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


def training_steps_for_model(batch, model, optimizer, lr_scheduler):
    """This function executes one training step for the model"""
    batch = {k: v.to(device) for k, v in batch.items()}
    predictions = model(**batch)
    loss = predictions.loss
    loss.backward()

    optimizer.step()

    lr_scheduler.step()

    optimizer.zero_grad()


from EvaluationTest import evaluate_model as ev


def evaluation_step_for_model(batch, batch_real_results_list, batch_predictions_list, backup_list, model):
    """This function executes one evaluation step for the model"""
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():

        outputs = model(**batch)

    predictions = outputs.logits

    real_values = batch["labels"]
    real_values = real_values.cpu().tolist()
    predictions_list = predictions.cpu().tolist()
    for single_label_set in real_values:
        batch_real_results_list.append(single_label_set)
    for prediction_set in predictions_list:
        pred_list = []
        current_max = 1
        for single_prediction in range(len(prediction_set)):
            x = 1 / (1 + np.exp(-prediction_set[single_prediction]))
            print("sigmoid_of_prediction")
            print(x)
            if (single_prediction == 0 and x > 0.5):
                pred_list.append(1.0)
            elif (single_prediction == 0):
                pred_list.append(0.0)
            else:
                if prediction_set[current_max] < prediction_set[single_prediction]:
                    current_max = single_prediction
                pred_list.append(x)
            backup_list.append(current_max)
        batch_predictions_list.append(pred_list)
    return batch_real_results_list, batch_predictions_list


import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(eval_dataloader, model):
    """Here the result get gathered further computed and evaluated"""
    batch_real_results_list = []
    batch_predictions_list = []
    # In case all are assigned to 0
    backup_list = []
    for batch in eval_dataloader:
        batch_real_results_list, batch_predictions_list = evaluation_step_for_model(batch, batch_real_results_list,
                                                                                    batch_predictions_list, backup_list,
                                                                                    model)
    label_coi = extract_label_of_one_type(batch_predictions_list, 1)
    label_coi = calculate_gauss(label_coi)
    for i in range(len(batch_predictions_list)):
        if label_coi[i] < np.mean(label_coi) - calculate_standardabweichung(label_coi, np.mean(label_coi)):
            batch_predictions_list[i][1] = 1.0
        else:
            batch_predictions_list[i][1] = 0.0

    label_fanpov = extract_label_of_one_type(batch_predictions_list, 2)
    label_fanpov = calculate_gauss(label_fanpov)
    for i in range(len(batch_predictions_list)):
        if label_fanpov[i] < np.mean(label_fanpov) - calculate_standardabweichung(label_fanpov, np.mean(label_fanpov)):
            batch_predictions_list[i][2] = 1.0
        else:
            batch_predictions_list[i][2] = 0.0

    label_pr = extract_label_of_one_type(batch_predictions_list, 3)
    label_pr = calculate_gauss(label_pr)
    for i in range(len(batch_predictions_list)):
        if label_pr[i] < np.mean(label_pr) - calculate_standardabweichung(label_pr, np.mean(label_pr)):
            batch_predictions_list[i][3] = 1.0
        else:
            batch_predictions_list[i][3] = 0.0

    label_resume = extract_label_of_one_type(batch_predictions_list, 4)
    label_resume = calculate_gauss(label_resume)
    for i in range(len(batch_predictions_list)):
        if label_resume[i] < np.mean(label_resume) - calculate_standardabweichung(label_resume, np.mean(label_resume)):
            batch_predictions_list[i][4] = 1.0
        else:
            batch_predictions_list[i][4] = 0.0
    for i in range(len(batch_predictions_list)):
        value_assigned_to_one = False
        for j in range(5):
            if j == 1:
                value_assigned_to_one = True
        if not value_assigned_to_one:
            batch_predictions_list[i][backup_list[i]] = 1

    ev(np.array(batch_predictions_list), np.array(batch_real_results_list))
    plt.show()


from tqdm.auto import tqdm


def calculate_gauss(values):
    """The gauss value gets calculated here"""
    mean = np.mean(values)
    print("mean")
    print(mean)
    s = calculate_standardabweichung(values, mean)
    print("standard")
    print(s)
    gauss_values = []
    for i in values:
        exponent_e = (1 / 2) * (((i - mean) * (i - mean)) / (2 * s))
        print("exponent_e")
        print(exponent_e)
        e_value = np.exp(-exponent_e)
        x = 2 * np.pi * s
        print("2*np.pi*s")
        print(2 * np.pi * s)
        gauss_value = (1 / np.sqrt(x)) * e_value
        gauss_values.append(gauss_value)
    return gauss_values


def calculate_standardabweichung(values, mean):
    """Here the variance gets calculated"""
    sum_standard = 0
    for i in values:
        sum_standard = sum_standard + ((i - mean) * (i - mean))
    standardabweichung = sum_standard / float(len(values))
    return standardabweichung


def extract_label_of_one_type(labels, position):
    """Here a label can be chosen and it is being extracted for all predictions"""
    extracted_labels = []
    for i in labels:
        extracted_labels.append(i[position])
    return extracted_labels


def execute_all_methods():
    """This method executes the whole flow"""
    promotional_path = "promotional.csv"
    amount_of_entries = 1000
    model_type = "distilbert-base-uncased"
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type, attention_type='sdpa')
    dataset_final, amount_of_classes = read_data_with_multilabel(promotional_path)

    tokenized_dataset_test = tokenize_data(dataset_final)
    testing_data, training_data = split_and_reduce_data(tokenized_dataset_test, amount_of_entries)
    train_dataloader, eval_dataloader = create_dataloaders(training_data, testing_data)

    optimizer_type = optim.AdamW
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    model, optimizer, lr_scheduler, loss_function, device = create_model_and_functions(model_type_transformer,
                                                                                       optimizer_type,
                                                                                       num_training_steps,
                                                                                       amount_of_classes)

    progress_bar = tqdm(range(num_training_steps))

    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            training_steps_for_model(batch, model, optimizer, lr_scheduler, loss_function)
            progress_bar.update(1)
        # evaluate_model(eval_dataloader)

    model.eval()
    evaluate_model(eval_dataloader, model)


batch_size = 1
execute_all_methods()
