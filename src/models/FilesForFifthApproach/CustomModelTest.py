"""from transformers import AutoModelForSequenceClassification
from transformers import BertConfig, BertModel

config = BertConfig()
print(config)


#model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
"""
import torch
from torch import nn

# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertConfig


class DistilBertExtensionFirstVersion(DistilBertForSequenceClassification):
    def __init__(self, config, num_of_classes):
        print(DistilBertConfig())
        super().__init__(config)
        self.bertPart = DistilBertForSequenceClassification.from_pretrained("google-bert/bert-base-cased",
                                                                            num_labels=256)
        self.first_custom_layer = nn.Linear(256, num_of_classes)
        self.act_first_custom_layer = nn.ReLU()
        self.s_test = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # Feed input to BERT
        outputs = self.bertPart(input_ids=input_ids,
                                attention_mask=attention_mask)
        # print(outputs[0])
        outputs = self.first_custom_layer(outputs[0])
        outputs = self.act_first_custom_layer(outputs)
        # outputs = self.s_test(outputs)
        outputs = self.s_test(outputs)

        return outputs
