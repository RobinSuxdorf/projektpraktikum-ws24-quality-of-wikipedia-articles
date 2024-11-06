<<<<<<< HEAD
# Projektpraktikum: Qualität von Wikipedia-Artikelns

## Overview

This project provides an overview of the text classification pipeline using a Naive Bayes model. The workflow is designed for classifying promotional vs. non-promotional text data by leveraging feature extraction techniques and model training steps. The setup is modular to allow easy integration or adaptation with different models.

## What's Here

- **Data (`data/raw/`)**: Stores raw data for input.
- **Data Preparation (`src/data_preparation.py`)**: Loads data, cleans text, removes stopwords using NLTK, and labels text as promotional (1) or non-promotional (0).
- **Feature Engineering (`src/feature_engineering.py`)**: Extracts features from the cleaned text using TF-IDF vectorization, converting text into numerical data for model training.
- **Model Training (`src/model.py`)**: Trains a Naive Bayes classifier using the TF-IDF features. Logging is implemented to track the training process and model performance.
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

## Logging

Logging is centrally configured in `main.py`, and each module uses `logging.getLogger(__name__)` to maintain consistent log formatting throughout the project.

## Guidelines

- **PEP 8 & black**: Follow PEP 8 coding standards and use black formatter to maintain code consistency.
- **Logging**: Use `logging.getLogger(__name__)` for logging in each module to integrate with the centralized logging configuration.
=======
# Quality of Wikipedia Articles

This project aims to analyze and classify Wikipedia articles into promotional and non-promotional categories.

## Table of Contents
- [Installation](#installation)
- [Code Formatter](#code-formatter)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Logging](#logging)

## Installation

### Prerequisites
- Python 3.x

To install the required packages, run the following command:
```sh
pip install -r requirements.txt
```

## Code Formatter
We use Ruff for code formatting. To format the code, run:
```sh
ruff check .
```

## Usage
To run the script, use the following command:
```sh
python main.py
```

You can also specify additional arguments:
```sh
python main.py -g data/raw/good.csv -p data/raw/promotional.csv -n 1000 -s
```
* -g, --good_file: Path to the CSV file containing non-promotional text data.
* -p, --promo_file: Path to the CSV file containing promotional text data.
* -n, --nrows: Number of rows to read from each CSV file.
* -s, --shuffle: Whether to shuffle the combined dataset.

## Project Structure
* Praktikumsbericht/: Contains the LaTeX code for the report.
* src/: Contains the source code.
    * data_loader.py: Functions for loading and processing data.
    * utils.py: Utility functions including argument parsing and file validation.
* main.py: Main script to run the project.

## Logging
Logs are saved to app.log and also printed to the console.
>>>>>>> main
