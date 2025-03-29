# Quality of Wikipedia Articles

This project aims to analyze and classify Wikipedia articles into promotional and non-promotional categories.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
  - [Data Loader](#data-loader)
  - [Preprocessing](#preprocessing)
  - [Features](#features)
  - [Models](#models)
  - [Grid Search](#grid-search)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Logging](#logging)
- [Convert Wikipedia Dump](#convert-wikipedia-dump)

## Installation

### Prerequisites

- Python 3.12.x (gensim does not compile in 3.13.x as of 2025-03-21)

To install the required packages, run the following command:

```sh
pip install -r requirements.txt
```

## Usage

To run a machine learning pipeline, use the command:

```sh
python main.py -c <config-name>
```

To run a deep learning pipeline, use the `main_deep_learning.ipynb` notebook.

To run the transformer models, use the `main_transformer.py` script.

## Configuration

Specify the configuration settings in the YAML files located in the configs/ directory. Below are the possible options for each section of the configuration.

### Data Loader

```yaml
data_loader:
  good_file: "data/raw/good.csv" # Path to the CSV file containing non-promotional text data.
  promo_file: "data/raw/promotional.csv" # Path to the CSV file containing promotional text data.
  neutral_file: "data/raw/neutral.csv" # (optional) Path to the CSV file containing neutral text data.
  nrows: 1000 # (optional) Number of rows to read from each CSV file.
  shuffle: false # Whether to shuffle the combined dataset.
  label_change_frac: 0.1 # (optional) Fraction of labels to randomly change.
  save: "loaded_data.csv" # Path to save the loaded data.
```

### Preprocessing

```yaml
preprocessing:
  remove_non_word: true # Remove non-word characters.
  convert_lowercase: true # Convert text to lowercase.
  remove_stopwords: true # Remove stopwords.
  apply_stemming: true # Apply stemming.
  remove_numbers: false # Remove numbers.
  remove_whitespace: true # Remove leading and trailing whitespace.
  save: "preprocessed_data.csv" # Path to save the preprocessed data.
```

### Features

#### TF-IDF Vectorizer

```yaml
features:
  type: tfidf
  ngram_range: [1, 1] # N-gram range for the vectorizer.
  max_df: 0.9 # Maximum document frequency for the vectorizer.
  min_df: 0.001 # Minimum document frequency for the vectorizer.
  max_features: 10_000 # Maximum number of features for the vectorizer.
  sublinear_tf: true # Apply sublinear term frequency scaling.
  save: "features_tfidf.pkl" # Path to save the extracted features.
```

#### Count Vectorizer

```yaml
features:
  type: count
  ngram_range: [1, 1] # N-gram range for the vectorizer.
  max_df: 0.9 # Maximum document frequency for the vectorizer.
  min_df: 0.001 # Minimum document frequency for the vectorizer.
  max_features: 10_000 # Maximum number of features for the vectorizer.
  binary: false # If True, all non-zero term counts are set to 1.
  save: "features_count.pkl" # Path to save the extracted features.
```

#### Bag of Words Vectorizer

```yaml
features:
  type: bagofwords
  ngram_range: [1, 1] # N-gram range for the vectorizer.
  max_df: 0.9 # Maximum document frequency for the vectorizer.
  min_df: 0.001 # Minimum document frequency for the vectorizer.
  max_features: 10_000 # Maximum number of features for the vectorizer.
  save: "features_bagofwords.pkl" # Path to save the extracted features.
```

#### Word2Vec Vectorizer

```yaml
features:
  type: word2vec
  workers: 4 # Number of workers for word2vec.
  vector_size: 100 # Vector size for word2vec.
  window: 5 # Window size for word2vec.
  min_count: 5 # Minimum count for word2vec.
  sg: 0 # Skip-gram (1) or CBOW (0) for word2vec.
  hs: 0 # Hierarchical softmax for word2vec.
  negative: 5 # Negative sampling for word2vec.
  alpha: 0.025 # Initial learning rate for word2vec.
  epochs: 5 # Number of epochs for word2vec.
  save: "features_word2vec.pkl" # Path to save the extracted features.
```

#### GloVe Vectorizer

```yaml
features:
  type: glove
  model_name: "glove-wiki-gigaword-100" # GloVe model name.
  save: "features_glove.pkl" # Path to save the extracted features.
```

### Models

#### Logistic Regression

```yaml
model:
  type: logistic_regression # Type of model (logistic_regression).
  test_size: 0.2 # Fraction of data to use for testing.
  random_state: 42 # (optional) Random state for reproducibility.
  grid_search: false # Whether to perform grid search for hyperparameter tuning.
  penalty: "l2" # Penalty for logistic regression (l1 or l2).
  solver: "liblinear" # Solver for logistic regression (liblinear, saga, etc.).
  max_iter: 100 # Maximum number of iterations for logistic regression.
  save: "logistic_regression_model.pkl" # Path to save the trained model.
```

#### Naive Bayes

```yaml
model:
  type: naive_bayes # Type of model (naive_bayes).
  test_size: 0.2 # Fraction of data to use for testing.
  random_state: 42 # (optional) Random state for reproducibility.
  grid_search: false # Whether to perform grid search for hyperparameter tuning.
  alpha: 1.0 # Smoothing parameter for Naive Bayes.
  fit_prior: false # Whether to learn class prior probabilities.
  save: "naive_bayes_model.pkl" # Path to save the trained model.
```

#### Support Vector Machine (SVM)

```yaml
model:
  type: linear_svm # Type of model (linear_svm).
  test_size: 0.2 # Fraction of data to use for testing.
  random_state: 42 # (optional) Random state for reproducibility.
  grid_search: false # Whether to perform grid search for hyperparameter tuning.
  C: 1.0 # Regularization parameter for SVM.
  loss: hinge # Loss function parameter for SVM.
  save: "svm_model.pkl" # Path to save the trained model.
```

### Grid Search

Grid search is a technique used to find the optimal hyperparameters for a model by exhaustively searching through a specified parameter grid. When `grid_search` is set to `true`, the model will perform grid search using the provided `param_grid`.

Example configuration for Naive Bayes with grid search:

```yaml
model:
  type: naive_bayes
  test_size: 0.2
  random_state: 42
  grid_search: true
  param_grid:
    alpha: [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]
    fit_prior: [true, false]
  save: "naive_bayes_model.pkl"
```

### Evaluation

```yaml
evaluation:
  test_data: "data/raw/promotional.csv" # (optional) Path to a test dataset to be used instead of the test data from a train test split
  save: "evaluation.png" # Path to save the evaluation results.
```

## Project Structure

- Berichte/: LaTeX code for presentations and report associated with the project
- configs/: YAML configuration files
- graphics/: Visualizations generated during data analysis and model evaluation
- preserved_experiments/: Preserved artifacts from exploratory data analysis, prototyping experiments, and interactive development
- scripts/: Auxillary scripts intended to be run directly rather than imported
  - graph_label_distribution.py: Graph the distribution of the labels in the promotional dataset
  - graph_results_bar.py: Graph the results of the model performances using bar plots
  - main_wp_dump_convert.py: Convert a Wikipedia XML dump into CSV files
  - main_wp_dump_sampling.py: Sample Wikipedia dump with reservoir sampling
- src/: Main source code of the project
  - models/: Model implementations
    - deep_learning/: Deep learning implementations
      - base.py: Base class for neural network models
      - dl_binary.py: Model class definitions for a neural network for binary classification
      - dl_multiclass.py: Model class definitions for a neural network for multiclass classification
      - dl_multilabel.py: Model class definitions for a neural network for multilabel classification
    - base.py: Abstract base class for machine learning models
    - logistic_regression.py: Model class definitions for Logistic Regression models
    - naive_bayes.py: Model class definitions for Naive Bayes models
    - support_vector_machine.py: Model class definitions for Support Vector Machine models
  - transformer/: Transformer model implementation with all its submodules
  - vectorizer/: Vectorizer implementations
    - base.py: Abstract base class for vectorizers
    - gensim.py: Vectorizer class definitions for vectorizers using Gensim models
    - sklearn.py: Vectorizer class definitions for vectorizers using Scikit-learn models
  - wp/: Wikipedia dump processing code
    - categorized_page.py: Module for categorizing Wikipedia pages with quality and type classification functionalities
    - wp_dump.py: Module for processing Wikipedia dump files and categorizing pages into CSV outputs
  - data_loader.py: Module for loading data for binary and multilabel classification tasks
  - evaluation.py: Module for evaluating classification models by computing metrics and visualizing them using bar plots
  - features.py: Module for extracting features from text using various vectorization techniques
  - preprocessing.py: Module for preprocessing text data using various techniques
  - train.py: Module for training models using various algorithms with optional grid search optimization
  - utils.py: Module for utility functions and pipeline step definitions
- main_deep_learning.ipynb: Main script to run the pipeline for deep learning approach
- main.py: Main script for machine learning preprocessing and model training and evaluation pipeline
- README.md: Project documentation <-- YOU ARE HERE
- requirements.txt: List of required Python packages (experiments not included)

## Logging

Logs are saved to app.log and also printed to the console.

## Convert Wikipedia Dump

Download from https://dumps.wikimedia.org/enwiki/

- `enwiki-yyyymmdd-pages-articles-multistream.xml.bz2` (about 20 GB)
- `enwiki-yyyymmdd-pages-articles-multistream-index.txt.bz2` (about 250 MB)

To split the dump into 4 CSV files (good, promo, neutral, skipped) run:

```sh
python main_wp_dump_convert.py -c <config-name>
```

To extract samples from the CSV files run:

```sh
python main_wp_dump_sampling.py -c <config-name>
```
