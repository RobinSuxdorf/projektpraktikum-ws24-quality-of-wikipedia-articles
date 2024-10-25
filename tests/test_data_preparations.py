# tests/test_data_preparation.py

import unittest
import pandas as pd
from src.data_preparation import preprocess_text_series, prepare_data


class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_text_series = pd.Series(
            ["Hello, this is a TEST sentence!", "It contains symbols and stopwords."]
        )
        self.expected_cleaned_text_series = pd.Series(
            ["hello test sentence", "contains symbols stopwords"]
        )

        # Create sample CSV files with minimal data
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

        # Save mock data to CSV for testing the prepare_data function
        self.promo_data.to_csv("tests/sample_promotional.csv", index=False)
        self.good_data.to_csv("tests/sample_good.csv", index=False)

    def tearDown(self):
        # Clean up any files created after each test
        import os

        os.remove("tests/sample_promotional.csv")
        os.remove("tests/sample_good.csv")

    def test_preprocess_text_series(self):
        # Test if preprocess_text_series correctly cleans a series of text
        cleaned_series = preprocess_text_series(self.sample_text_series)
        pd.testing.assert_series_equal(
            cleaned_series, self.expected_cleaned_text_series
        )

    def test_prepare_data_output_format(self):
        # Test if prepare_data returns a DataFrame with expected columns
        data = prepare_data("tests/sample_promotional.csv", "tests/sample_good.csv")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn("cleaned_text", data.columns)
        self.assertIn("label", data.columns)

    def test_prepare_data_row_count(self):
        # Test if prepare_data loads and processes the correct number of rows
        data = prepare_data(
            "tests/sample_promotional.csv", "tests/sample_good.csv", nrows=1
        )
        self.assertEqual(len(data), 2)  # 1 row from each file

    def test_prepare_data_labels(self):
        # Test if prepare_data correctly assigns labels
        data = prepare_data("tests/sample_promotional.csv", "tests/sample_good.csv")
        self.assertIn(1, data["label"].values)
        self.assertIn(0, data["label"].values)

    def test_prepare_data_content(self):
        # Test if prepare_data correctly applies preprocess_text_series to the text data
        data = prepare_data("tests/sample_promotional.csv", "tests/sample_good.csv")
        for text in data["cleaned_text"]:
            self.assertTrue(text.islower())
            self.assertNotRegex(text, r"[^\w\s]")  # Check no special characters

    def test_prepare_data_file_not_found(self):
        # Test if prepare_data gracefully handles missing files
        with self.assertRaises(FileNotFoundError):
            prepare_data("tests/non_existent_file.csv", "tests/sample_good.csv")

    def test_prepare_data_empty_file(self):
        # Test if prepare_data gracefully handles empty files
        open("tests/empty.csv", "w").close()  # Create an empty file
        with self.assertRaises(pd.errors.EmptyDataError):
            prepare_data("tests/empty.csv", "tests/sample_good.csv")
        import os

        os.remove("tests/empty.csv")


if __name__ == "__main__":
    unittest.main()
