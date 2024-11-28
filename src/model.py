# src/model.py

import logging
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

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
    oversampling = model_config.get("oversampling", False)

    logging.info("Splitting data into training and testing sets.")
    x_train, _, y_train, _ = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    if type == "naive_bayes":
        if y_train.shape[1] == 1:
            logging.info("Training a binary Naive Bayes model.")
            model = MultinomialNB(alpha=alpha)
            y_train = y_train.values.ravel()
            model.fit(x_train, y_train)
        else:
            logging.info("Training a multilabel Naive Bayes model.")
            if oversampling:
                model = OneVsRestClassifier(
                    Pipeline(
                        [
                            ("oversample", RandomOverSampler(random_state=42)),
                            ("classifier", MultinomialNB(alpha=alpha)),
                        ]
                    )
                )
            else:
                model = OneVsRestClassifier(MultinomialNB(alpha=alpha))
            model.fit(x_train, y_train)

    return model
