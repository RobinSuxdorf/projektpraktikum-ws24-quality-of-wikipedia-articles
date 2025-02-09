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
good_path = "good.csv"
promotional_path = "promotional.csv"

def read_in_data():
    df_promo = pd.read_csv(promotional_path)
    df_promo_aggregated = df_promo.groupby(["advert","coi","fanpov","pr","resume"]).count()
    df_promo_aggregated['label'] = range(1, len(df_promo_aggregated)+1)

    df_final = df_promo_aggregated.merge(df_promo, left_on = ["advert","coi","fanpov","pr","resume"],
                                     right_on=["advert","coi","fanpov","pr","resume"], how = 'inner')

    print("here a limitation happens")
    df_final = df_final.loc[df_final["label"]== 1]
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
    return df_temp, 2 #len(df_promo_aggregated)+1



def tokenizer_function(examples):
    return tokenizer(examples["text"], padding = "max_length", truncation = True)


model_name = "google-bert/bert-base-cased"##"bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = BertModel.from_pretrained(model_name)

from datasets import load_dataset, concatenate_datasets

dataset = load_dataset("csv", data_files="good.csv", split="train")
dataset2 = load_dataset("csv", data_files="promotional.csv", split="train")
#label_column = [1] * len(dataset)
#dataset = dataset.add_column("good", label_column)


label_column2 = [1] * len(dataset2)
print(label_column2)
len_1 =len(dataset)
#dataset2 = dataset2.add_column("good", label_column2)
"""dataset = dataset.add_column("advert", label_column)
dataset = dataset.add_column("coi", label_column)
dataset = dataset.add_column("fanpov", label_column)
dataset = dataset.add_column("pr", label_column)
dataset = dataset.add_column("resume", label_column)"""
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





tokenized_dataset = dataset_final.map(tokenizer_function, batched =True)
tokenized_datasets = tokenized_dataset.remove_columns(["text"])
#tokenized_datasets = tokenized_datasets.remove_columns(["label"])

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets)
testing_data = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
training_data= tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
print("testing data")
print(testing_data)

print("is it using this")

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, shuffle=True, batch_size=8)

eval_dataloader = DataLoader(testing_data, batch_size=8)

from transformers import AutoModelForSequenceClassification
import CustomModelTest
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
model2 = CustomModelTest.NeuralNetwork()
from transformers import DistilBertModel
model = DistilBertModel.from_pretrained("distilbert-base-cased", torch_dtype=torch.float16, attn_implementation="sdpa")

from CustomModelTest import My_extended_DistilBertForSequenceClassification

model = My_extended_DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")


from torch.optim import AdamW

optimizer = AdamW(list(model.parameters()) + list(model2.parameters()), lr=5e-5)



from transformers import get_scheduler

num_epochs = 3

num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(

    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps

)



import torch

from accelerate.test_utils.testing import get_backend

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)

model.to(device)





from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()

import torch

loss_function = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        #outputs = model2(outputs1)
        logit = outputs
        #outputs = outputs.long().to("cuda")
        loss = loss_function(outputs, batch["labels"].long())
        loss.backward()

        optimizer.step()

        lr_scheduler.step()

        optimizer.zero_grad()

        progress_bar.update(1)





import evaluate

"""metric = evaluate.load("accuracy")
metric_f1= evaluate.load("f1")
metric_mcc = evaluate.load("matthews_correlation")
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")"""


model.eval()

"""print(metric.compute())
print(metric_f1.compute())
print(metric_mcc.compute())
print(metric_precision.compute())
print(metric_recall.compute())"""


from metrics_test import calculate_metrics

batch_real_results_list = []
batch_predictions_list =[]

for batch in eval_dataloader:

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
    print(batch_real_results_list)
    print(batch_predictions_list)
    """metric_f1.add_batch(predictions=predictions, references=batch["labels"])
    metric_mcc.add_batch(predictions=predictions, references=batch["labels"])
    metric_precision.add_batch(predictions=predictions, references=batch["labels"])
    metric_recall.add_batch(predictions=predictions, references=batch["labels"])"""


"""print(metric.compute())

print(metric_mcc.compute())
print(metric_precision.compute())
print(metric_recall.compute())"""
calculate_metrics(batch_predictions_list, batch_real_results_list)






