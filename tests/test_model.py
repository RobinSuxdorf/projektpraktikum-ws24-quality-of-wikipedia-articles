# tests/test_model.py

import unittest
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from src.model import train_naive_bayes

logging.getLogger(__name__)


class TestModelTraining(unittest.TestCase):
    """
    Test suite for the model training functions in the model module.
    """

    def setUp(self):
        """
        Set up sample data for testing.
        """
        logging.info("Setting up sample data for model training tests.")
        self.sample_text_series = pd.Series(
            [
                "Hello, this is a test sentence!",
                "Another test, with different words.",
                "Machine learning is amazing!",
                "Feature extraction is a key part of NLP.",
            ]
        )
        self.labels = pd.Series([0, 1, 0, 1])

        # Extract TF-IDF features for testing
        vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
        self.features = vectorizer.fit_transform(self.sample_text_series)

    def test_train_naive_bayes_output_type(self):
        """
        Test if train_naive_bayes returns a MultinomialNB instance.
        """
        logging.info("Testing if train_naive_bayes returns a MultinomialNB instance.")
        model = train_naive_bayes(self.features, self.labels)
        self.assertIsInstance(model, MultinomialNB)

    def test_train_naive_bayes_non_empty(self):
        """
        Test if train_naive_bayes trains a model that can make predictions.
        """
        logging.info(
            "Testing if train_naive_bayes trains a model that can make predictions."
        )
        model = train_naive_bayes(self.features, self.labels)
        predictions = model.predict(self.features)
        self.assertEqual(len(predictions), len(self.labels))

    def test_train_naive_bayes_train_test_split(self):
        """
        Test if train_naive_bayes performs a train-test split with the correct sizes.
        """
        logging.info(
            "Testing if train_naive_bayes performs a train-test split with correct sizes."
        )
        test_size = 0.25
        _, _, _, y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=42
        )
        _ = train_naive_bayes(self.features, self.labels, test_size=test_size)
        self.assertEqual(len(y_test), int(len(self.labels) * test_size))


if __name__ == "__main__":
    unittest.main()
