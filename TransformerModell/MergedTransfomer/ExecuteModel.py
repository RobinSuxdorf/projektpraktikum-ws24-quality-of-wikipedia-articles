from transformers import AutoTokenizer
import datasets
from DataLoaders import read_data_with_multilabel, read_data_with_multilabel_augmented, read_data_for_binary_classification, read_data_for_three_class_classification
from TokenizeData import tokenize_data
from TrainModel import split_and_reduce_data
from CreateAllModelComponents import create_dataloaders, create_model_and_functions
from torch import optim
from transformers import get_scheduler
from tqdm.auto import tqdm
from TrainModel import train_model
from Evaluation import evaluate_model_multilabel, evaluate_model_binary

"""This tutorial has been used to learn implementing transformers
Fine-tune a pretrained model, https://huggingface.co/docs/transformers/en/training, 18.03.2025"""


def execute_all_methods(classification_type, model_type,  promotional_path, augmented_promotional_path, neutral_path,  good_path, amount_of_entries, preprocess_data_flg, batch_size, model_config, num_epochs):
    """This method executes the whole flow"""

    if classification_type == "multilabel_normal":
        model_config.num_labels = 5
        dataset_final, amount_of_classes = read_data_with_multilabel(promotional_path, preprocess_data_flg)
    elif classification_type == "multilabel_augmented":
        model_config.num_labels = 5
        dataset_final, amount_of_classes = read_data_with_multilabel_augmented(promotional_path, augmented_promotional_path, preprocess_data_flg)
    elif classification_type == "binary_classification":
        model_config.num_labels = 2
        dataset_final, amount_of_classes = read_data_for_binary_classification(promotional_path,
                                                                               good_path,
                                                                               preprocess_data_flg)
    elif classification_type == "three_class_classification":
        model_config.num_labels = 3
        dataset_final, amount_of_classes = read_data_for_three_class_classification(promotional_path,
                                                                                    neutral_path,
                                                                               good_path,
                                                                               preprocess_data_flg)
    """Creates the tokenizer and tokenizes the data"""
    tokenizer = AutoTokenizer.from_pretrained(model_type, attention_type='sdpa')
    tokenized_dataset_test = tokenize_data(dataset_final, tokenizer)
    """Prepares the data for training"""
    testing_data, training_data = split_and_reduce_data(tokenized_dataset_test, amount_of_entries)
    train_dataloader, eval_dataloader = create_dataloaders(training_data, testing_data, batch_size)

    """Creates the model"""
    optimizer_type = optim.AdamW
    num_training_steps = num_epochs * len(train_dataloader)
    model, optimizer, lr_scheduler, loss_function, device = create_model_and_functions(model_type,
                                                                                       optimizer_type,
                                                                                       num_training_steps,
                                                                                       amount_of_classes, model_config)

    """The model get trained"""
    progress_bar = tqdm(range(num_training_steps))
    train_model(num_epochs, progress_bar, train_dataloader, model, optimizer, lr_scheduler, loss_function, device)

    "The model gets evaluated"
    model.eval()
    if classification_type =="multilabel_normal" or classification_type == "multilabel_augmented":
        evaluate_model_multilabel(eval_dataloader, model, device)
    elif classification_type == "binary_classification" or classification_type == "three_class_classification":
        evaluate_model_binary(eval_dataloader, model, device)


