"""Reads in the Batches and evaluates the model

Author: Emmanuelle Steenhof
"""

from src.transformer.EvaluationTest import evaluate_model
import torch
import numpy as np
from src.transformer.HilfsfunktionenGauss import extract_label_of_one_type
from src.transformer.HilfsfunktionenGauss import calculate_gauss
from src.transformer.HilfsfunktionenGauss import calculate_variance
from matplotlib import pyplot as plt


def evaluation_step_for_model_binary(
    batch, batch_real_results_list, batch_predictions_list, model, device
):
    """This function is one evaluation step of the model"""
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        predictions = model(**batch)
    real_values = batch["labels"]
    for value in real_values:
        batch_real_results_list.append(value.cpu())
    for single_prediction in predictions.logits:
        batch_predictions_list.append(torch.argmax(single_prediction, dim=-1).cpu())
    return batch_real_results_list, batch_predictions_list


def evaluate_model_binary(eval_dataloader, model, device):
    """This function gathers the results in a list for further evaluations"""
    batch_real_results_list = []
    batch_predictions_list = []
    for batch in eval_dataloader:
        batch_real_results_list, batch_predictions_list = (
            evaluation_step_for_model_binary(
                batch, batch_real_results_list, batch_predictions_list, model, device
            )
        )
    evaluate_model(np.array(batch_predictions_list), np.array(batch_real_results_list))
    plt.show()


def evaluation_step_for_model_multilabel(
    batch, batch_real_results_list, batch_predictions_list, backup_list, model, device
):
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
            if single_prediction == 0 and x > 0.5:
                pred_list.append(1.0)
            elif single_prediction == 0:
                pred_list.append(0.0)
            else:
                if prediction_set[current_max] < prediction_set[single_prediction]:
                    current_max = single_prediction
                pred_list.append(x)
            backup_list.append(current_max)
        batch_predictions_list.append(pred_list)
    return batch_real_results_list, batch_predictions_list


def evaluate_model_multilabel(eval_dataloader, model, device):
    """Here the result get gathered further computed and evaluated"""
    batch_real_results_list = []
    batch_predictions_list = []
    backup_list = []
    for batch in eval_dataloader:
        batch_real_results_list, batch_predictions_list = (
            evaluation_step_for_model_multilabel(
                batch,
                batch_real_results_list,
                batch_predictions_list,
                backup_list,
                model,
                device,
            )
        )
    batch_predictions_list = set_label_via_anomaly_detection(batch_predictions_list, 1)

    batch_predictions_list = set_label_via_anomaly_detection(batch_predictions_list, 2)

    batch_predictions_list = set_label_via_anomaly_detection(batch_predictions_list, 3)

    batch_predictions_list = set_label_via_anomaly_detection(batch_predictions_list, 4)

    """In case nothing has been assigned the biggest logit of the labels except for advert is being set to 1"""
    for i in range(len(batch_predictions_list)):
        value_assigned_to_one = False
        for j in range(5):
            if j == 1:
                value_assigned_to_one = True
        if not value_assigned_to_one:
            batch_predictions_list[i][backup_list[i]] = 1

    evaluate_model(np.array(batch_predictions_list), np.array(batch_real_results_list))
    plt.show()


def set_label_via_anomaly_detection(batch_predictions_list, position):
    """Sets the labels to 1 if they are considered an outlier"""
    labels_to_set = extract_label_of_one_type(batch_predictions_list, position)
    labels_to_set = calculate_gauss(labels_to_set)
    for i in range(len(batch_predictions_list)):
        if labels_to_set[i] < np.mean(labels_to_set) - calculate_variance(
            labels_to_set, np.mean(labels_to_set)
        ):
            batch_predictions_list[i][position] = 1.0
        else:
            batch_predictions_list[i][position] = 0.0
    return batch_predictions_list
