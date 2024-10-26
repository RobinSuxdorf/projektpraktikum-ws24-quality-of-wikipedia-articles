# src/model.py

import logging
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.getLogger(__name__)


def train_naive_bayes(features, labels, test_size=0.2, random_state=42):
    """
    Train a Naive Bayes classifier on the provided features and labels.

    Args:
        features (sparse matrix): The numerical features extracted from the text data.
        labels (pd.Series or array-like): The labels corresponding to the text data.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        MultinomialNB: The trained Naive Bayes model.
    """
    logging.info("Splitting data into training and testing sets.")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    logging.info("Training the Naive Bayes model.")
    model = MultinomialNB()
    model.fit(x_train, y_train)

    logging.info("Making predictions on the test set.")
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logging.info(f"Model accuracy: {accuracy:.2f}")
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"F1 Score: {f1:.2f}")

    return model
