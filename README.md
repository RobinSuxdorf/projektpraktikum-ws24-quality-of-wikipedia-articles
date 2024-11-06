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
