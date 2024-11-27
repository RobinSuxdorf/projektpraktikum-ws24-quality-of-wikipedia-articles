# src/model.py

import logging
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

logging.getLogger(__name__)


def train_model(features, labels, model_config: dict):
    """
    Train a model based on the provided configuration.

    Args:
        features: Feature matrix.
        labels: Target labels.
        model_config (dict): Configuration dictionary for the model.

    Returns:
        Model: Trained model.
    """
    type = model_config.get("type")
    alpha = model_config.get("alpha")

    logging.info("Splitting data into training and testing sets.")
    x_train, _, y_train, _ = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    if type == "naive_bayes":
        logging.info("Training a Naive Bayes model.")
        model = MultinomialNB(alpha=alpha)
        model.fit(x_train, y_train)

    return model
