"""Provides supporting functions for anomaly detection

Authors: Emmanuelle Steenhof"""

import numpy as np
def calculate_gauss(values):
    """The gauss value gets calculated here"""
    mean = np.mean(values)
    s = calculate_variance(values, mean)
    gauss_values = []
    for i in values:
        exponent_e = (1 / 2) * (((i - mean) * (i - mean)) / (2 * s))
        e_value = np.exp(-exponent_e)
        x = 2 * np.pi * s
        gauss_value = (1 / np.sqrt(x)) * e_value
        gauss_values.append(gauss_value)
    return gauss_values


def calculate_variance(values, mean):
    """Here the variance gets calculated"""
    sum_variance = 0
    for i in values:
        sum_variance = sum_variance + ((i - mean) * (i - mean))
    variance = sum_variance / float(len(values))
    return variance

def extract_label_of_one_type(labels, position):
    """Here a label can be chosen and it is being extracted for all predictions"""
    extracted_labels = []
    for i in labels:
        extracted_labels.append(i[position])
    return extracted_labels