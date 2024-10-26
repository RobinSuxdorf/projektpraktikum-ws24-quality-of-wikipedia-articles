# Text Classification Project

## Overview
This project provides an overview of the text classification pipeline using a Naive Bayes model. The workflow is designed for classifying promotional vs. non-promotional text data by leveraging feature extraction techniques and model training steps. The setup is modular to allow easy integration or adaptation with different models.

## What's Here
- **Data Preparation (`src/data_preparation.py`)**: Loads data, cleans text, removes stopwords using NLTK, and labels text as promotional (1) or non-promotional (0).
- **Feature Engineering (`src/feature_engineering.py`)**: Extracts features from the cleaned text using TF-IDF vectorization, converting text into numerical data for model training.
- **Model Training (`src/model.py`)**: Trains a Naive Bayes classifier using the TF-IDF features. Logging is implemented to track the training process and model performance.
- **Tests (`tests/`)**: Includes unit tests for each major module to ensure reliability.
  - `test_data_preparation.py`, `test_feature_engineering.py`, `test_model.py`.
- **Main Pipeline (`main.py`)**: Integrates all steps, from loading and processing data to training and saving the model.
- **Models (`models/`)**: Stores saved models and vectorizers for future use.

## Running the Project
1. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
   Dependencies include pandas, scikit-learn, joblib, and nltk.

2. **Run the Main Pipeline**:
   ```sh
   python main.py
   ```
   This script will prepare the data, extract features, train the model, and save the resulting model and vectorizer to `models/`.

3. **Run Tests**:
   ```sh
   python -m unittest discover tests/
   ```
   Runs the unit tests to verify that each module works as expected.

## Logging
Logging is centrally configured in `main.py`, and each module uses `logging.getLogger(__name__)` to maintain consistent log formatting throughout the project.

## Guidelines
- **PEP 8**: Follow PEP 8 coding standards and black formmatter to maintain code consistency.
- **Logging**: Use `logging.getLogger(__name__)` for logging in each module to integrate with the centralized logging configuration.
- **Testing**: Add or update tests for any new features or modifications to maintain robustness.
