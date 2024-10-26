# tests/test_feature_engineering.py

import unittest
import logging
import pandas as pd
from src.feature_engineering import extract_features
from sklearn.feature_extraction.text import TfidfVectorizer

logging.getLogger(__name__)


class TestFeatureEngineering(unittest.TestCase):
    """
    Test suite for the feature engineering functions in the feature_engineering module.
    """

    def setUp(self):
        """
        Set up sample data for testing.
        """
        logging.info("Setting up sample data for feature engineering tests.")
        self.sample_text_series = pd.Series(
            [
                "Hello, this is a test sentence!",
                "Another test, with different words.",
                "Machine learning is amazing!",
                "Feature extraction is a key part of NLP.",
            ]
        )

    def test_extract_features_output_format(self):
        """
        Test if extract_features returns the correct output format.
        """
        logging.info("Testing output format of extract_features function.")
        features, vectorizer = extract_features(self.sample_text_series)
        self.assertIsInstance(features, type(TfidfVectorizer().fit_transform(["test"])))
        self.assertIsInstance(vectorizer, TfidfVectorizer)

    def test_extract_features_non_empty(self):
        """
        Test if extract_features returns non-empty features for non-empty input.
        """
        logging.info("Testing non-empty output of extract_features function.")
        features, _ = extract_features(self.sample_text_series)
        self.assertGreater(features.shape[0], 0)
        self.assertGreater(features.shape[1], 0)

    def test_extract_features_max_features(self):
        """
        Test if extract_features respects the max_features argument.
        """
        logging.info("Testing max_features parameter of extract_features function.")
        max_features = 2
        features, _ = extract_features(
            self.sample_text_series, max_features=max_features
        )
        self.assertEqual(features.shape[1], max_features)


if __name__ == "__main__":
    unittest.main()
