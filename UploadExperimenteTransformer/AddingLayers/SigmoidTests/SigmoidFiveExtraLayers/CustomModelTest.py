"""from transformers import AutoModelForSequenceClassification
from transformers import BertConfig, BertModel

config = BertConfig()
print(config)


#model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
"""
import torch
from torch import nn

#https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Linear(10, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def build_final_model(model):
    second_part = NeuralNetwork()
    my_new_layer = torch.nn.Linear(64, 64)
    model = torch.nn.Sequential(model, second_part)
    return model


from transformers import DistilBertForSequenceClassification
from transformers import DistilBertConfig

class DistilBertExtensionFirstVersion(DistilBertForSequenceClassification):
    def __init__(self, config, num_of_classes):
        print(DistilBertConfig())
        super().__init__(config)
        self.bertPart = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 256)
        self.first_custom_layer = nn.Linear(256 ,1024)
        self.act_first_custom_layer =nn.Sigmoid()
        self.second_custom_layer = nn.Linear(1024 ,1024)
        self.act_second_custom_layer =nn.Sigmoid()
        self.third_custom_layer = nn.Linear(1024, 1024)
        self.act_third_custom_layer = nn.Sigmoid()
        self.fourth_custom_layer = nn.Linear(1024, 2048)
        self.act_fourth_custom_layer = nn.Sigmoid()
        self.fifth_custom_layer = nn.Linear(2048, num_of_classes)
        self.act_fifth_custom_layer = nn.Sigmoid()
        self.s_test = nn.Softmax()

    def forward(self, input_ids, attention_mask, labels):
        # Feed input to BERT
        outputsBert = self.bertPart(input_ids=input_ids,
                            attention_mask=attention_mask)
        #print(outputs[0])
        hidden = self.first_custom_layer(outputsBert[0])
        hidden= self.act_first_custom_layer(hidden)
        hidden = self.second_custom_layer(hidden)
        hidden = self.act_second_custom_layer(hidden)
        hidden = self.third_custom_layer(hidden)
        hidden = self.act_third_custom_layer(hidden)
        hidden = self.fourth_custom_layer(hidden)
        hidden = self.act_fourth_custom_layer(hidden)
        hidden = self.fifth_custom_layer(hidden)
        outputs = self.act_fifth_custom_layer(hidden)
        #outputs = self.s_test(outputs)
        outputs = self.s_test(outputs)
        logits = outputs

        return logits
