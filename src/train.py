# src/model.py

import logging
from sklearn.model_selection import train_test_split
from src.models import Model, NaiveBayes, MultilabelNaiveBayes

logging.getLogger(__name__)


def train_model(features, labels, model_config: dict) -> Model:
    """_summary_

    Args:
        features (_type_): _description_
        data (_type_): _description_
        model_config (dict): _description_

    Returns:
        Model: _description_
    """
    type = model_config.get("type")

    logging.info("Splitting data into training and testing sets.")
    x_train, _, y_train, _ = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    if type == "naive_bayes":
        if y_train.shape[1] == 1:
            logging.info("Training a binary Naive Bayes model.")
            model = NaiveBayes(model_config)
            y_train = y_train.values.ravel()
            model.fit(x_train, y_train)
        else:
            logging.info("Training a multilabel Naive Bayes model.")
            model = MultilabelNaiveBayes(model_config)
            model.fit(x_train, y_train)

    return model
