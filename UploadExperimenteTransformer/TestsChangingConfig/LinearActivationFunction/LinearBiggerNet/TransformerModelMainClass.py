import pandas as pd
import torch
from transformers import pipeline
from datasets import load_dataset, Dataset
#https://huggingface.co/learn/nlp-course/chapter4/2
#classifier = pipeline("text-classification", model= "bert-base-uncased")
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel
import pandas
import datasets

#Es waere in Ordnung ein BERT Model zu verwenden. Alle die beim Gespraech waren waren einverstanden.


def read_in_data(promotional_path, good_path):
    df_promo = pd.read_csv(promotional_path)
    df_promo_aggregated = df_promo.groupby(["advert","coi","fanpov","pr","resume"]).count()
    df_promo_aggregated['label'] = range(1, len(df_promo_aggregated)+1)

    df_final = df_promo_aggregated.merge(df_promo, left_on = ["advert","coi","fanpov","pr","resume"],
                                     right_on=["advert","coi","fanpov","pr","resume"], how = 'inner')
    df_promo_final = df_final[['label', 'text_y', 'url_y']]
    #print(df_promo_final)
###df_good = pd.read_csv("Daten/good.csv")
    df_good = pd.read_csv(good_path)
    df_good["label"] = 0
    df_promo2 = df_promo_final[["text_y", "label"]]
    print(df_promo2)
    df_promo3= df_promo2.rename(columns = {"text_y" : "text"})
    #df_promo3 = df_promo2.rename(columns = {'text_y':'text', 'label':'label'}, inplace = True)
    print("df_promo3")
    print(df_promo2)
    df_good2 = df_good[["text", "label"]]
    test = [df_good2, df_promo3]
    df_temp = pd.concat(test)
    print("this is what df_temp looks like")
    print(df_temp)
    dataset_final = datasets.Dataset.from_pandas(df_temp)
    dataset_final = dataset_final.remove_columns("__index_level_0__")
    dataset_final = dataset_final.shuffle()
    dataset_final = dataset_final.train_test_split(0.3)
    print(dataset_final)
    return dataset_final, len(df_promo_aggregated)+1



def tokenizer_function(examples):
    return tokenizer(examples["text"], padding = "max_length", truncation = True)



#model = BertModel.from_pretrained(model_name)

from datasets import load_dataset, concatenate_datasets

def prepare_dataset_binary_classification(good_path, promotional_path):
    dataset = load_dataset("csv", data_files=good_path, split="train")
    dataset2 = load_dataset("csv", data_files=promotional_path, split="train")

    label_column2 = [1] * len(dataset2)
    print(label_column2)
    len_1 =len(dataset)
    dataset = dataset.remove_columns("url")
    label_column = [0] * len(dataset)
    dataset = dataset.add_column("label", label_column)


    label_column = [1] * len(dataset2)
    dataset2= dataset2.remove_columns("advert")
    dataset2= dataset2.remove_columns("coi")
    dataset2= dataset2.remove_columns("fanpov")
    dataset2= dataset2.remove_columns("pr")
    dataset2= dataset2.remove_columns("resume")
    dataset2= dataset2.remove_columns("url")
    df = pd.read_csv("promotional.csv")
#label_list =[]
    df_promo = pd.read_csv("promotional.csv")
    df_promo_aggregated = df_promo.groupby(["advert", "coi", "fanpov", "pr", "resume"]).count()
    df_promo_aggregated["label"] = range(1, len(df_promo_aggregated) + 1)

    df_final = df_promo_aggregated.merge(df_promo, left_on=["advert", "coi", "fanpov", "pr", "resume"],
                                     right_on=["advert", "coi", "fanpov", "pr", "resume"], how='inner')

    label_list= df_final["label"]
    print(label_list)
    dataset2 = dataset2.add_column("label", label_column2)
    print(dataset2[0])
    #dataset2 = dataset2.train_test_split(0.3)

    #dataset = dataset.train_test_split(0.3)
    dataset_final= concatenate_datasets([dataset, dataset2])
    dataset_final = dataset_final.shuffle()
    dataset_final = dataset_final.train_test_split(0.3)
    print(dataset_final)
    return dataset_final




def tokenize_data(dataset_final):
    tokenized_dataset = dataset_final.map(tokenizer_function, batched =True)
    tokenized_datasets = tokenized_dataset.remove_columns(["text"])
#tokenized_datasets = tokenized_datasets.remove_columns(["label"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets)
    return tokenized_datasets



def split_and_reduce_data(tokenized_datasets, amount_of_entries):
    testing_data = tokenized_datasets["test"].shuffle(seed=42).select(range(amount_of_entries))
    training_data= tokenized_datasets["train"].shuffle(seed=42).select(range(amount_of_entries))
    return testing_data, training_data




from torch.utils.data import DataLoader

def create_dataloaders(training_data, testing_data):
    train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(testing_data, batch_size=16)
    return train_dataloader, eval_dataloader



from CustomModelTest import DistilBertExtensionFirstVersion

from torch import optim

from transformers import get_scheduler

from accelerate.test_utils.testing import get_backend

import torch

from transformers import DistilBertConfig


model_config = DistilBertConfig()

print("Config Before:")
print(model_config)

model_config.activation= 'linear'
model_config.hidden_dim = 6144
model_config.n_layers = 12
model_config.dim =1536

print("Config After:")
print(model_config)

def create_model_and_functions(model_type_transformer, optimizer_type, num_training_steps, num_of_classes):
    model = DistilBertExtensionFirstVersion.from_pretrained(model_type_transformer, num_of_classes= num_of_classes, config= model_config, ignore_mismatched_sizes=True)
    optimizer = optimizer_type(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(

        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps

    )
    device, _, _ = get_backend()
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, loss_function, device




def training_steps_for_model(batch, model, optimizer, lr_scheduler):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    # outputs = model2(outputs1)
    logit = outputs
    # outputs = outputs.long().to("cuda")
    loss = loss_function(outputs, batch["labels"].long())
    loss.backward()

    optimizer.step()

    lr_scheduler.step()

    optimizer.zero_grad()


from metrics_test import calculate_metrics
def evaluation_step_for_model(batch, batch_real_results_list, batch_predictions_list):
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():

        outputs = model(**batch)

    predictions = torch.argmax(outputs, dim=-1)
    values_test = batch["labels"].cpu().tolist()
    prediction_test = predictions.cpu().tolist()
    for test_v in values_test:
        batch_real_results_list.append(test_v)
    for test_v in prediction_test:
        batch_predictions_list.append(test_v)
    return batch_real_results_list, batch_predictions_list

def evaluate_model(eval_dataloader):
    batch_real_results_list = []
    batch_predictions_list = []
    for batch in eval_dataloader:
        batch_real_results_list, batch_predictions_list = evaluation_step_for_model(batch, batch_real_results_list, batch_predictions_list)
    calculate_metrics(batch_predictions_list, batch_real_results_list)


good_path = "good.csv"
promotional_path = "promotional.csv"



amount_of_entries = 1000

model_name = "google-bert/bert-base-cased"##"bert-base-uncased"
model_type_transformer = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_type_transformer)
dataset_final = prepare_dataset_binary_classification(good_path, promotional_path)
amount_of_classes =2
print("binary_dataset")

print(dataset_final)
dataset_final, amount_of_classes =read_in_data(promotional_path, good_path)
print("multiclass_dataset")


print(dataset_final)
tokenized_dataset_test = tokenize_data(dataset_final)
print(tokenized_dataset_test)
testing_data, training_data = split_and_reduce_data(tokenized_dataset_test, amount_of_entries)
train_dataloader, eval_dataloader = create_dataloaders(training_data, testing_data)

optimizer_type = optim.AdamW
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
model, optimizer, lr_scheduler, loss_function, device = create_model_and_functions(model_type_transformer, optimizer_type, num_training_steps, amount_of_classes)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        training_steps_for_model(batch, model, optimizer, lr_scheduler)
        progress_bar.update(1)
        #evaluate_model(eval_dataloader)


model.eval()



evaluate_model(eval_dataloader)









