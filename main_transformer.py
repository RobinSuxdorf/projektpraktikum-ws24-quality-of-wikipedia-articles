"""Defines the configurations and executes the transformer models

Author: Emmanuelle Steenhof"""

from src.transformer.ExecuteModel import execute_all_methods

from transformers import DistilBertConfig

"""Binary-Class Classification"""
model_config = DistilBertConfig()
model_config.activation = "gelu"
num_epochs = 5
augmented_promotional_path = ""
good_path = "good.csv"
neutral_path = ""
promotional_path = "promotional.csv"
model_type = "distilbert-base-uncased"
amount_of_entries = "ALL"
preprocess_data_flg = True
batch_size = 2
classification_type = "binary_classification"
execute_all_methods(
    classification_type,
    model_type,
    promotional_path,
    augmented_promotional_path,
    neutral_path,
    good_path,
    amount_of_entries,
    preprocess_data_flg,
    batch_size,
    model_config,
    num_epochs,
)


"""Three-Class Classification"""
model_config = DistilBertConfig()
model_config.add_cross_attention = True
model_config.activation = "gelu"
num_epochs = 5
augmented_promotional_path = ""
good_path = "good.csv"
neutral_path = "wikipedia_samples/neutral_sample.csv"
promotional_path = "promotional.csv"
model_type = "distilbert-base-uncased"
amount_of_entries = "ALL"
preprocess_data_flg = True
batch_size = 2
classification_type = "three_class_classification"
execute_all_methods(
    classification_type,
    model_type,
    promotional_path,
    augmented_promotional_path,
    neutral_path,
    good_path,
    amount_of_entries,
    preprocess_data_flg,
    batch_size,
    model_config,
    num_epochs,
)


"""Multilabel Classification"""
model_config = DistilBertConfig()
model_config.add_cross_attention = True
model_config.activation = "gelu"
num_epochs = 3
augmented_promotional_path = ""
good_path = ""
neutral_path = ""
promotional_path = "promotional.csv"
model_type = "distilbert-base-uncased"
amount_of_entries = "ALL"
preprocess_data_flg = True
batch_size = 1
classification_type = "multilabel_normal"
execute_all_methods(
    "multilabel_normal",
    model_type,
    promotional_path,
    augmented_promotional_path,
    neutral_path,
    good_path,
    amount_of_entries,
    preprocess_data_flg,
    batch_size,
    model_config,
    num_epochs,
)


"""Multilabel Classification Augmented"""
model_config = DistilBertConfig()
model_config.add_cross_attention = True
model_config.activation = "gelu"
num_epochs = 3
augmented_promotional_path = "augmented_promotional.csv"
good_path = ""
neutral_path = ""
promotional_path = "promotional.csv"
model_type = "distilbert-base-uncased"
amount_of_entries = "ALL"
preprocess_data_flg = True
batch_size = 1
classification_type = "multilabel_augmented"
execute_all_methods(
    classification_type,
    model_type,
    promotional_path,
    augmented_promotional_path,
    neutral_path,
    good_path,
    amount_of_entries,
    preprocess_data_flg,
    batch_size,
    model_config,
    num_epochs,
)
