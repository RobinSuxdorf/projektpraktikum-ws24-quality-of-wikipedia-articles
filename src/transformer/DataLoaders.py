"""Reads in the data
Based on files in pipeline adjusted to work for transformer
Author: Emmanuelle Steenhof
"""

import pandas as pd
from DataPreprocessing import preprocess_data
import datasets


def read_data_for_binary_classification(
    promotional_path, good_path, preprocess_data_flg
):
    """Reads in the data for binary classification"""
    df_promo = pd.read_csv(promotional_path)
    df_good = pd.read_csv(good_path)
    df_promo["labels"] = 1
    df_good["labels"] = 0
    df_promo = df_promo[["text", "labels"]]
    df_good = df_good[["text", "labels"]]
    if preprocess_data_flg:
        df_promo["text"] = preprocess_data(df_promo["text"])
        df_good["text"] = preprocess_data(df_good["text"])
    concat_prep = [df_promo, df_good]
    df = pd.concat(concat_prep)
    ds = datasets.Dataset.from_pandas(df)
    ds = ds.remove_columns("__index_level_0__")
    ds = ds.shuffle(seed=42)
    ds = ds.train_test_split(0.2)
    return ds, 2


def read_data_for_three_class_classification(
    promotional_path, neutral_path, good_path, preprocess_data_flg
):
    """Here the data is being read in"""
    df_promo = pd.read_csv(promotional_path)
    df_good = pd.read_csv(good_path)
    df_neutral = pd.read_csv(neutral_path)
    df_promo["labels"] = 2
    df_neutral["labels"] = 1
    df_good["labels"] = 0
    df_promo = df_promo[["text", "labels"]]
    df_neutral = df_neutral[["text", "labels"]]
    df_good = df_good[["text", "labels"]]
    if preprocess_data_flg:
        df_promo["text"] = preprocess_data(df_promo["text"])
        df_neutral["text"] = preprocess_data(df_neutral["text"])
        df_good["text"] = preprocess_data(df_good["text"])
    concat_prep = [df_promo, df_good, df_neutral]
    df = pd.concat(concat_prep)
    ds = datasets.Dataset.from_pandas(df)
    ds = ds.remove_columns("__index_level_0__")
    ds = ds.shuffle(seed=42)
    ds = ds.train_test_split(0.2)
    return ds, 3


def read_data_with_multilabel(promotional_path, preprocess_data_flg):
    df_promo = pd.read_csv(promotional_path)
    if preprocess_data_flg:
        df_promo["text"] = preprocess_data(df_promo["text"])
    labels_promo = []
    """The labels are being concatenated"""
    for i in range(len(df_promo)):
        labels_of_i = []
        labels_of_i.append(float(df_promo.iloc[i]["advert"]))
        labels_of_i.append(float(df_promo.iloc[i]["coi"]))
        labels_of_i.append(float(df_promo.iloc[i]["fanpov"]))
        labels_of_i.append(float(df_promo.iloc[i]["pr"]))
        labels_of_i.append(float(df_promo.iloc[i]["resume"]))
        labels_promo.append(labels_of_i)
    """The labels column is being assigned"""
    df_promo["labels"] = labels_promo
    df_promo = df_promo.drop("advert", axis=1)
    df_promo = df_promo.drop("coi", axis=1)
    df_promo = df_promo.drop("fanpov", axis=1)
    df_promo = df_promo.drop("pr", axis=1)
    df_promo = df_promo.drop("resume", axis=1)
    df_promo = df_promo.drop("url", axis=1)
    """The dataframe is being turned into a dataset"""
    dataset_final = datasets.Dataset.from_pandas(df_promo)
    dataset_final = dataset_final.shuffle()
    dataset_final = dataset_final.train_test_split(0.2)
    print(dataset_final)
    return dataset_final, 5


def read_data_with_multilabel_augmented(
    promotional_path, augmented_path, preprocess_data_flg
):
    """This function makes reads in the data"""
    df_promo = pd.read_csv(promotional_path)
    df_promo_augmented = pd.read_csv(augmented_path)
    if preprocess_data_flg:
        df_promo["text"] = preprocess_data(df_promo["text"])
        df_promo_augmented["text"] = preprocess_data(df_promo_augmented["text"])
    labels_promo = []
    for i in range(len(df_promo)):
        labels_of_i = []
        labels_of_i.append(float(df_promo.iloc[i]["advert"]))
        labels_of_i.append(float(df_promo.iloc[i]["coi"]))
        labels_of_i.append(float(df_promo.iloc[i]["fanpov"]))
        labels_of_i.append(float(df_promo.iloc[i]["pr"]))
        labels_of_i.append(float(df_promo.iloc[i]["resume"]))
        labels_promo.append(labels_of_i)
    df_promo["labels"] = labels_promo
    df_promo = df_promo.drop("advert", axis=1)
    df_promo = df_promo.drop("coi", axis=1)
    df_promo = df_promo.drop("fanpov", axis=1)
    df_promo = df_promo.drop("pr", axis=1)
    df_promo = df_promo.drop("resume", axis=1)
    df_promo = df_promo.drop("url", axis=1)

    df_promo_augmented_list = []
    for i in range(len(df_promo_augmented)):
        labels_of_i = []
        labels_of_i.append(float(df_promo_augmented.iloc[i]["advert"]))
        labels_of_i.append(float(df_promo_augmented.iloc[i]["coi"]))
        labels_of_i.append(float(df_promo_augmented.iloc[i]["fanpov"]))
        labels_of_i.append(float(df_promo_augmented.iloc[i]["pr"]))
        labels_of_i.append(float(df_promo_augmented.iloc[i]["resume"]))
        df_promo_augmented_list.append(labels_of_i)
    df_promo_augmented["labels"] = df_promo_augmented_list
    df_promo_augmented = df_promo_augmented.drop("advert", axis=1)
    df_promo_augmented = df_promo_augmented.drop("coi", axis=1)
    df_promo_augmented = df_promo_augmented.drop("fanpov", axis=1)
    df_promo_augmented = df_promo_augmented.drop("pr", axis=1)
    df_promo_augmented = df_promo_augmented.drop("resume", axis=1)
    df_promo_augmented = df_promo_augmented.drop("url", axis=1)
    ds = datasets.DatasetDict()
    dataset_final = datasets.Dataset.from_pandas(df_promo)
    ds["test"] = dataset_final
    ds["train"] = datasets.Dataset.from_pandas(df_promo_augmented)
    print(ds)
    return ds, 5
