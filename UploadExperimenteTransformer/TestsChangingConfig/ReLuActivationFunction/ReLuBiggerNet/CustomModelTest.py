"""from transformers import AutoModelForSequenceClassification
from transformers import BertConfig, BertModel

config = BertConfig()
print(config)


#model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
"""
import torch
from torch import nn

from transformers import BertConfig, BertModel

config = BertConfig()

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
        self.bertPart = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = num_of_classes)
        """self.first_custom_layer = nn.Linear(256 ,1024)
        self.second_custom_layer = nn.Linear(1024 ,1024)
        self.third_custom_layer = nn.Linear(1024, num_of_classes)"""

        self.s_test = nn.Softmax()

    def forward(self, input_ids, attention_mask, labels):
        # Feed input to BERT
        outputsBert = self.bertPart(input_ids=input_ids,
                            attention_mask=attention_mask)
        """hidden = self.first_custom_layer(outputsBert[0])
        hidden = self.second_custom_layer(hidden)"""
        """outputs = self.third_custom_layer(hidden)
        outputs = self.s_test(outputsBert)"""
        logits = outputsBert[0]

        return logits
