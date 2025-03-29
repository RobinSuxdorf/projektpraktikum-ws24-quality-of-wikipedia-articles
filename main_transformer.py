"""Defines the configurations and executes the transformer models
Below are the settings used for the use cases

This tutorial has been used to learn implementing transformers
Fine-tune a pretrained model, https://huggingface.co/docs/transformers/en/training, 18.03.2025

Author: Emmanuelle Steenhof
"""

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
save_file_path = "trained_models/binary_classification"
# load_trained_model_flg means whether one wants to evaluate a pretrained model wiwthout further training.
load_trained_model_flg = 0
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
    save_file_path,
    load_trained_model_flg,
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
save_file_path = "trained_models/three_class_classification"
# load_trained_model_flg means whether one wants to evaluate a pretrained model wiwthout further training.
load_trained_model_flg = 0
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
    save_file_path,
    load_trained_model_flg,
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
save_file_path = "trained_models/multilabel_normal_classification"
# load_trained_model_flg means whether one wants to evaluate a pretrained model wiwthout further training.
load_trained_model_flg = 0
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
    save_file_path,
    load_trained_model_flg,
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
save_file_path = "trained_models/multilabel_augmented_classification"
# load_trained_model_flg means whether one wants to evaluate a pretrained model wiwthout further training.
load_trained_model_flg = 0
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
    save_file_path,
    load_trained_model_flg,
)
