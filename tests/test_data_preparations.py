# tests/test_data_preparation.py

import unittest
import pandas as pd
import os
from src.data_preparation import preprocess_text_series, prepare_data


class TestDataPreparation(unittest.TestCase):
    """
    Test suite for the data preparation functions in the data_preparation module.
    """

    PROMO_FILE = "tests/sample_promotional.csv"
    GOOD_FILE = "tests/sample_good.csv"
    EMPTY_FILE = "tests/empty.csv"

    def setUp(self):
        """
        Set up sample data for testing.

        Creates sample text series and dataframes, and saves them to CSV files.
        """
        self.sample_text_series = pd.Series(
            ["Hello, this is a TEST sentence!", "It contains symbols and stopwords."]
        )
        self.expected_cleaned_text_series = pd.Series(
            ["hello test sentence", "contains symbols stopwords"]
        )

        self.promo_data = pd.DataFrame(
            {
                "text": [
                    "This is a promotional message!",
                    "Check out our new product now!",
                ]
            }
        )
        self.good_data = pd.DataFrame(
            {"text": ["This is a regular message.", "Hope you enjoy the content!"]}
        )

        self.promo_data.to_csv(self.PROMO_FILE, index=False)
        self.good_data.to_csv(self.GOOD_FILE, index=False)

    def tearDown(self):
        """
        Clean up any files created after each test.

        Removes the sample CSV files created during testing.
        """
        if os.path.exists(self.PROMO_FILE):
            os.remove(self.PROMO_FILE)
        if os.path.exists(self.GOOD_FILE):
            os.remove(self.GOOD_FILE)
        if os.path.exists(self.EMPTY_FILE):
            os.remove(self.EMPTY_FILE)

    def test_preprocess_text_series(self):
        """
        Test if preprocess_text_series correctly cleans a series of text.

        Checks that the returned series matches the expected cleaned series.
        """
        cleaned_series = preprocess_text_series(self.sample_text_series)
        pd.testing.assert_series_equal(
            cleaned_series, self.expected_cleaned_text_series
        )

    def test_prepare_data_output_format(self):
        """
        Test if prepare_data returns a DataFrame with expected columns.

        Checks that the returned DataFrame has 'cleaned_text' and 'label' columns.
        """
        data = prepare_data(self.PROMO_FILE, self.GOOD_FILE)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn("cleaned_text", data.columns)
        self.assertIn("label", data.columns)

    def test_prepare_data_row_count(self):
        """
        Test if prepare_data loads and processes the correct number of rows.

        Verifies that the DataFrame has the correct number of rows based on the nrows argument.
        """
        data = prepare_data(self.PROMO_FILE, self.GOOD_FILE, nrows=1)
        self.assertEqual(len(data), 2)

    def test_prepare_data_labels(self):
        """
        Test if prepare_data correctly assigns labels.

        Checks that promotional data is labeled with 1 and non-promotional data with 0.
        """
        data = prepare_data(self.PROMO_FILE, self.GOOD_FILE)
        self.assertIn(1, data["label"].values)
        self.assertIn(0, data["label"].values)

    def test_prepare_data_content(self):
        """
        Test if prepare_data correctly applies preprocess_text_series to the text data.

        Verifies that the cleaned text is in lowercase and contains no special characters.
        """
        data = prepare_data(self.PROMO_FILE, self.GOOD_FILE)
        for text in data["cleaned_text"]:
            self.assertTrue(text.islower())
            self.assertNotRegex(text, r"[^\w\s]")

    def test_prepare_data_file_not_found(self):
        """
        Test if prepare_data gracefully handles missing files.

        Expects a FileNotFoundError when a non-existent file is provided.
        """
        with self.assertRaises(FileNotFoundError):
            prepare_data("tests/non_existent_file.csv", self.GOOD_FILE)

    def test_prepare_data_empty_file(self):
        """
        Test if prepare_data gracefully handles empty files.

        Expects an EmptyDataError when an empty file is provided.
        """
        open(self.EMPTY_FILE, "w").close()
        with self.assertRaises(pd.errors.EmptyDataError):
            prepare_data(self.EMPTY_FILE, self.GOOD_FILE)


if __name__ == "__main__":
    unittest.main()
